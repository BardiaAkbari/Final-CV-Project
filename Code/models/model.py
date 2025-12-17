import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# BatchNorm helpers
# ============================================================

class BatchNorm1d_P(nn.BatchNorm1d):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm2d_P(nn.BatchNorm2d):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = super().forward(x)
        return x.permute(0, 2, 3, 1)

# ============================================================
# Point cloud utility functions
# ============================================================

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.reshape(*raw_shape, -1)


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch_idx = torch.arange(B, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    return dist.topk(k, dim=-1, largest=False)[1]

# ============================================================
# Point Transformer Layer
# ============================================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.share_planes = share_planes
        self.mid_planes = out_planes
        self.attn_planes = out_planes // share_planes
        self.k = nsample

        self.linear_q = nn.Linear(in_planes, self.mid_planes)
        self.linear_k = nn.Linear(in_planes, self.mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, 3),
            BatchNorm2d_P(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes)
        )

        self.fc_gamma = nn.Sequential(
            BatchNorm2d_P(self.mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_planes, self.attn_planes),
            BatchNorm2d_P(self.attn_planes),
            nn.ReLU(inplace=True),
            nn.Linear(self.attn_planes, self.attn_planes)
        )

    def forward(self, xyz, features):
        idx = knn_point(self.k, xyz, xyz)            
        knn_xyz = index_points(xyz, idx)                

        q = self.linear_q(features)                   
        k = index_points(self.linear_k(features), idx)  
        v = index_points(self.linear_v(features), idx)  

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz) 

        attn = self.fc_gamma(
            q[:, :, None] - k +
            pos_enc.view(
                pos_enc.shape[0],
                pos_enc.shape[1],
                pos_enc.shape[2],
                self.share_planes,
                -1
            ).sum(dim=3)
        ) 

        attn = F.softmax(attn, dim=-2)

        v = (v + pos_enc).view(
            v.shape[0],
            v.shape[1],
            v.shape[2],
            self.share_planes,
            -1
        )  

        out = torch.einsum(
            "bnksa,bnka->bnsa",
            v, attn
        )

        return out.reshape(out.shape[0], out.shape[1], -1)


# ============================================================
# Point Transformer Block
# ============================================================

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()

        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = BatchNorm1d_P(planes)

        self.transformer = PointTransformerLayer(
            planes, planes, share_planes, nsample
        )
        self.bn2 = BatchNorm1d_P(planes)

        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = BatchNorm1d_P(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, px):
        xyz, features = px
        identity = features

        features = self.relu(self.bn1(self.linear1(features)))
        features = self.relu(self.bn2(self.transformer(xyz, features)))
        features = self.bn3(self.linear3(features))
        features = self.relu(features + identity)

        return [xyz, features]

# ============================================================
# Transition Down
# ============================================================

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample

        if stride != 1:
            self.linear = nn.Linear(in_planes + 3, out_planes, bias=False)
            self.bn = BatchNorm2d_P(out_planes)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.bn = BatchNorm1d_P(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, px):
        xyz, features = px

        if self.stride == 1:
            features = self.relu(self.bn(self.linear(features)))
            return [xyz, features]

        npoint = xyz.shape[1] // self.stride
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)

        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

        grouped_features = index_points(features, idx)
        grouped_features = torch.cat(
            [grouped_xyz_norm, grouped_features], dim=-1
        )

        grouped_features = self.relu(self.bn(self.linear(grouped_features)))
        new_features = grouped_features.max(dim=2)[0]

        return [new_xyz, new_features]

# ============================================================
# Point Transformer Classification Network
# ============================================================

class PointTransformerCls(nn.Module):
    def __init__(self, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_channels
        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        self.cls = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(
                PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        xyz = x[..., :3].contiguous()

        xyz1, features1 = self.enc1([xyz, x])
        xyz2, features2 = self.enc2([xyz1, features1])
        xyz3, features3 = self.enc3([xyz2, features2])
        xyz4, features4 = self.enc4([xyz3, features3])
        xyz5, features5 = self.enc5([xyz4, features4])

        return self.cls(features5.mean(dim=1))

# ============================================================
# Model Variants
# ============================================================

class PointTransformerCls26(PointTransformerCls):
    def __init__(self, **kwargs):
        super().__init__([1, 1, 1, 1, 1], **kwargs)


class PointTransformerCls38(PointTransformerCls):
    def __init__(self, **kwargs):
        super().__init__([1, 2, 2, 2, 2], **kwargs)


class PointTransformerCls50(PointTransformerCls):
    def __init__(self, **kwargs):
        super().__init__([1, 2, 3, 5, 2], **kwargs)
