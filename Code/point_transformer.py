import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
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
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
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
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = TransformerBlock(channels, 512, k)
        self.k = k
        self.nneighbor = nneighbor

    def forward(self, xyz, points):
        return self.sa(xyz, points)

class PointTransformerCls(nn.Module):
    def __init__(self, num_class, num_point=1024, input_dim=3, k=16):
        super().__init__()
        self.num_point = num_point
        self.k = k
        
        # Initial embedding
        self.input_dim = input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Transformer Block 1
        self.transformer1 = TransformerBlock(32, 512, k)
        
        # Transition Down 1: N -> N/4, 32 -> 64
        self.trans1_stride = 4
        self.trans1_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Transformer Block 2
        self.transformer2 = TransformerBlock(64, 512, k)
        
        # Transition Down 2: N/4 -> N/16, 64 -> 128
        self.trans2_stride = 4
        self.trans2_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Transformer Block 3
        self.transformer3 = TransformerBlock(128, 512, k)
        
        # Transition Down 3: N/16 -> N/64, 128 -> 256
        self.trans3_stride = 4
        self.trans3_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Transformer Block 4
        self.transformer4 = TransformerBlock(256, 512, k)
        
        # Transition Down 4: N/64 -> N/256, 256 -> 512
        self.trans4_stride = 4
        self.trans4_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Transformer Block 5
        self.transformer5 = TransformerBlock(512, 512, k)
        
        # Classification Head
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )

    def forward(self, x, return_attn=False):
        # x: (B, C, N) -> permute to (B, N, C) for transformer
        # In this implementation, we expect input (B, C, N) from dataloader usually, 
        # but the transformer code uses (B, N, C). Let's standardise.
        
        # Usually dataset returns (B, N, 3) or (B, N, 6). 
        # We assume x is (B, C, N) coming in.
        
        xyz = x[:, :3, :].permute(0, 2, 1) # B, N, 3
        features = x.permute(0, 2, 1) # B, N, C
        
        if self.input_dim == 3:
            features = features[:, :, :3]
        
        # Initial Embedding
        features = self.fc1(features) # B, N, 32
        
        # Stage 1
        features, _ = self.transformer1(xyz, features)
        
        # Transition Down 1
        n_pt1 = xyz.shape[1] // self.trans1_stride
        fps_idx1 = farthest_point_sample(xyz, n_pt1)
        new_xyz1 = index_points(xyz, fps_idx1)
        
        # kNN and Pooling for features
        knn_idx1 = knn_point(self.k, xyz, new_xyz1) # B, N_new, k
        grouped_features1 = index_points(features, knn_idx1) # B, N_new, k, C
        
        # Local Max Pooling
        # Apply linear transform before pooling
        grouped_features1 = self.trans1_fc(grouped_features1)
        new_features1 = torch.max(grouped_features1, dim=2)[0] # B, N_new, C_new
        
        xyz = new_xyz1
        features = new_features1
        
        # Stage 2
        features, _ = self.transformer2(xyz, features)
        
        # Transition Down 2
        n_pt2 = xyz.shape[1] // self.trans2_stride
        fps_idx2 = farthest_point_sample(xyz, n_pt2)
        new_xyz2 = index_points(xyz, fps_idx2)
        
        knn_idx2 = knn_point(self.k, xyz, new_xyz2)
        grouped_features2 = index_points(features, knn_idx2)
        grouped_features2 = self.trans2_fc(grouped_features2)
        new_features2 = torch.max(grouped_features2, dim=2)[0]
        
        xyz = new_xyz2
        features = new_features2
        
        # Stage 3
        features, _ = self.transformer3(xyz, features)
        
        # Transition Down 3
        n_pt3 = xyz.shape[1] // self.trans3_stride
        fps_idx3 = farthest_point_sample(xyz, n_pt3)
        new_xyz3 = index_points(xyz, fps_idx3)
        
        knn_idx3 = knn_point(self.k, xyz, new_xyz3)
        grouped_features3 = index_points(features, knn_idx3)
        grouped_features3 = self.trans3_fc(grouped_features3)
        new_features3 = torch.max(grouped_features3, dim=2)[0]
        
        xyz = new_xyz3
        features = new_features3
        
        # Stage 4
        features, _ = self.transformer4(xyz, features)
        
        # Transition Down 4
        n_pt4 = xyz.shape[1] // self.trans4_stride
        fps_idx4 = farthest_point_sample(xyz, n_pt4)
        new_xyz4 = index_points(xyz, fps_idx4)
        
        knn_idx4 = knn_point(self.k, xyz, new_xyz4)
        grouped_features4 = index_points(features, knn_idx4)
        grouped_features4 = self.trans4_fc(grouped_features4)
        new_features4 = torch.max(grouped_features4, dim=2)[0]
        
        xyz = new_xyz4
        features = new_features4
        
        # Stage 5
        features, _ = self.transformer5(xyz, features)
        
        # Global Average Pooling
        features = torch.mean(features, dim=1) # B, C
        
        # MLP Head
        x = self.fc_layer(features)
        
        return x, None