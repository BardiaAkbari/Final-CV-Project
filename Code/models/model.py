import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, k=16):
        super().__init__()
        self.k = k
        self.w_qs = nn.Linear(dim, dim, bias=False)
        self.w_ks = nn.Linear(dim, dim, bias=False)
        self.w_vs = nn.Linear(dim, dim, bias=False)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim)
        )
        
        self.linear_final = nn.Linear(dim, dim)

    def forward(self, xyz, feature):
        # xyz: B, N, 3
        # feature: B, N, C
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k] # B, N, K
        
        knn_xyz = index_points(xyz, knn_idx)
        
        q = self.w_qs(feature).unsqueeze(2)
        k = index_points(self.w_ks(feature), knn_idx)
        v = index_points(self.w_vs(feature), knn_idx)
        
        pos_enc = self.pos_encoder(xyz.unsqueeze(2) - knn_xyz)
        
        attn = self.attn_mlp(q - k + pos_enc) 
        attn = F.softmax(attn / np.sqrt(k.shape[-1]), dim=-2) # Softmax over K
        
        new_feature = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        new_feature = self.linear_final(new_feature) + feature
        return new_feature

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz, feature, npoint):
        # Sampling
        new_xyz_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, new_xyz_idx)
        
        # Grouping
        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) # B, npoint, nsample, 3
        grouped_feature = index_points(feature, idx) # B, npoint, nsample, C
        
        # MLP
        # (B, N, K, C) -> (B, C, N, K) for BatchNorm
        B, N_new, K, C = grouped_feature.shape
        grouped_feature = grouped_feature.permute(0, 3, 1, 2).contiguous().view(B, C, N_new*K)
        grouped_feature = self.mlp[0](grouped_feature.permute(0, 2, 1).view(B*N_new*K, C))
        grouped_feature = self.mlp[1](grouped_feature.permute(1, 0).view(1, -1, 1)).squeeze(2).view(B, -1, N_new, K).permute(0, 1, 2, 3) # Hacky BN
        grouped_feature = self.mlp[2](grouped_feature) # ReLU
        
        # Max Pooling
        new_feature = torch.max(grouped_feature, dim=-1)[0] # B, C, N_new
        new_feature = new_feature.permute(0, 2, 1) # B, N_new, C
        
        return new_xyz, new_feature

class PointTransformerCls(nn.Module):
    def __init__(self, blocks, c, k=16, num_classes=40):
        super().__init__()
        self.c = c
        self.in_planes = c
        self.k = k
        
        self.fc1 = nn.Sequential(
            nn.Linear(3+3, c), # XYZ + Normals (if available) or XYZ+XYZ
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True)
        )
        
        self.enc1 = self._make_enc(blocks[0], c)
        self.td1 = TransitionDown(c, c*2, nsample=k)
        self.enc2 = self._make_enc(blocks[1], c*2)
        self.td2 = TransitionDown(c*2, c*4, nsample=k)
        self.enc3 = self._make_enc(blocks[2], c*4)
        self.td3 = TransitionDown(c*4, c*8, nsample=k)
        self.enc4 = self._make_enc(blocks[3], c*8)
        
        self.fc2 = nn.Sequential(
            nn.Linear(c*8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_enc(self, block_num, planes):
        layers = []
        for _ in range(block_num):
            layers.append(PointTransformerLayer(planes, k=self.k))
        return nn.Sequential(*layers)

    def forward(self, xyz, x):
        # xyz: B, N, 3
        # x: B, N, C (input features)
        
        # Initial Embedding
        # If input has no features, use xyz as features
        if x is None:
            x = xyz
            
        # Helper to apply linear to (B, N, C)
        B, N, C = x.shape
        x = self.fc1(x.view(-1, C)).view(B, N, -1)
        
        # Stage 1
        for layer in self.enc1:
            x = layer(xyz, x)
            
        # Stage 2
        xyz, x = self.td1(xyz, x, npoint=512)
        for layer in self.enc2:
            x = layer(xyz, x)
            
        # Stage 3
        xyz, x = self.td2(xyz, x, npoint=128)
        for layer in self.enc3:
            x = layer(xyz, x)
            
        # Stage 4
        xyz, x = self.td3(xyz, x, npoint=32)
        for layer in self.enc4:
            x = layer(xyz, x)
            
        # Global Avg Pooling
        x = torch.mean(x, dim=1) # B, C
        
        x = self.fc2(x)
        return x

def point_transformer_38(num_classes=40, **kwargs):
    # ~38 layers: blocks [2, 2, 6, 2] * 3 (layers/block) + stems/trans ~ 38
    return PointTransformerCls([2, 2, 6, 2], c=32, num_classes=num_classes, **kwargs)

def point_transformer_50(num_classes=40, **kwargs):
    # ~50 layers: blocks [3, 4, 6, 3] usually
    return PointTransformerCls([3, 4, 6, 3], c=32, num_classes=num_classes, **kwargs)