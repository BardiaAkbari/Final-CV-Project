import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pure PyTorch Helper Functions (Replaces lib.pointops)
# ---------------------------------------------------------------------------
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

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

# ---------------------------------------------------------------------------
# Point Transformer Modules
# ---------------------------------------------------------------------------

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
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k] 
        knn_xyz = index_points(xyz, knn_idx)
        
        q = self.w_qs(feature).unsqueeze(2)
        k = index_points(self.w_ks(feature), knn_idx)
        v = index_points(self.w_vs(feature), knn_idx)
        
        pos_enc = self.pos_encoder(xyz.unsqueeze(2) - knn_xyz)
        
        attn = self.attn_mlp(q - k + pos_enc) 
        attn = F.softmax(attn / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32)), dim=-2)
        
        new_feature = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        new_feature = self.linear_final(new_feature) + feature
        return new_feature

class PointTransformerBlock(nn.Module):
    def __init__(self, c, k=16):
        super().__init__()
        self.layer = PointTransformerLayer(c, k=k)
        
    def forward(self, xyz, feature):
        return self.layer(xyz, feature)

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        
        # Flattened MLP approach to avoid view errors
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz, feature, npoint):
        # 1. Sampling
        if self.stride > 1:
            new_xyz_idx = farthest_point_sample(xyz, npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
        else:
            new_xyz = xyz
            npoint = xyz.shape[1]
        
        # 2. Grouping
        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_feature = index_points(feature, idx) # (B, N_new, K, C)
        
        # 3. MLP (Flatten -> Apply -> Reshape)
        B, N_new, K, C = grouped_feature.shape
        grouped_feature = grouped_feature.reshape(B * N_new * K, C)
        grouped_feature = self.mlp(grouped_feature)
        grouped_feature = grouped_feature.reshape(B, N_new, K, -1)
        
        # 4. Pooling
        new_feature = torch.max(grouped_feature, dim=2)[0] 
        
        return new_xyz, new_feature

class PointTransformerCls(nn.Module):
    def __init__(self, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        
        # Architecture Config matching your request [1, 2, 2, 2, 2]
        self.planes = [32, 64, 128, 256, 512]
        self.nsample = [8, 16, 16, 16, 16]
        # Number of points at each stage (approx downsampling)
        # N, N/4, N/16, N/64, N/256
        self.npoints = [1024, 256, 64, 16, 4] 
        
        self.enc1 = self._make_enc(self.in_channels, self.planes[0], blocks[0], self.nsample[0], self.npoints[0])
        self.enc2 = self._make_enc(self.planes[0], self.planes[1], blocks[1], self.nsample[1], self.npoints[1])
        self.enc3 = self._make_enc(self.planes[1], self.planes[2], blocks[2], self.nsample[2], self.npoints[2])
        self.enc4 = self._make_enc(self.planes[2], self.planes[3], blocks[3], self.nsample[3], self.npoints[3])
        self.enc5 = self._make_enc(self.planes[3], self.planes[4], blocks[4], self.nsample[4], self.npoints[4])

        self.cls = nn.Sequential(
            nn.Linear(self.planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_enc(self, in_planes, out_planes, blocks, nsample, npoint):
        layers = []
        # Transition Down (Standard stride logic handled by npoint passed to forward)
        # Note: We calculate stride dynamically based on npoint
        stride = 1 # Initial definition, logic handled in TransitionDown wrapper
        layers.append(TransitionDown(in_planes, out_planes, stride, nsample))
        
        # Transformer Blocks
        for _ in range(blocks):
            layers.append(PointTransformerBlock(out_planes, nsample))
        return nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, N, C_in)
        xyz = x[..., :3].contiguous()
        
        # If input is XYZ only
        if self.in_channels == 3:
            features = xyz
        else:
            features = x
            
        # Encoder 1
        xyz, features = self.enc1[0](xyz, features, npoint=self.npoints[0]) # TD
        for layer in self.enc1[1:]: features = layer(xyz, features)
            
        # Encoder 2
        xyz, features = self.enc2[0](xyz, features, npoint=self.npoints[1]) # TD
        for layer in self.enc2[1:]: features = layer(xyz, features)

        # Encoder 3
        xyz, features = self.enc3[0](xyz, features, npoint=self.npoints[2]) # TD
        for layer in self.enc3[1:]: features = layer(xyz, features)
        
        # Encoder 4
        xyz, features = self.enc4[0](xyz, features, npoint=self.npoints[3]) # TD
        for layer in self.enc4[1:]: features = layer(xyz, features)
        
        # Encoder 5
        xyz, features = self.enc5[0](xyz, features, npoint=self.npoints[4]) # TD
        for layer in self.enc5[1:]: features = layer(xyz, features)

        # Global Pooling
        res = self.cls(features.mean(dim=1))
        return res

def point_transformer_38(num_classes=40, in_channels=6, **kwargs):
    return PointTransformerCls([1, 2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, **kwargs)

def point_transformer_50(num_classes=40, in_channels=6, **kwargs):
    return PointTransformerCls([1, 2, 3, 5, 2], in_channels=in_channels, num_classes=num_classes, **kwargs)