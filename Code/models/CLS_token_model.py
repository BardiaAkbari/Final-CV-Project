import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pointops.functions import pointops
import einops

# ============================================================
# BatchNorm helpers
# ============================================================

class BatchNorm1d_P(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class BatchNorm2d_P(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = super().forward(x)
        return x.permute(0, 2, 3, 1)

# ============================================================
# Utility functions
# ============================================================

def index_points(points, idx):
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1).type(torch.int64)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

# ============================================================
# Point Transformer Layer
# ============================================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes, nsample):
        super().__init__()
        self.mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
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
        knn_idx = pointops.knnquery(self.k, xyz)
        if xyz.size(1) < self.k:
            knn_idx = knn_idx[:, :, :xyz.size(1)]
        knn_xyz = index_points(xyz, knn_idx)

        q = self.linear_q(features)
        k = index_points(self.linear_k(features), knn_idx)
        v = index_points(self.linear_v(features), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)

        attn = self.fc_gamma(
            q[:, :, None] - k +
            einops.reduce(pos_enc, "b n k (s a) -> b n k a", reduction="sum", a=self.mid_planes)
        )
        attn = F.softmax(attn, dim=-2)

        res = torch.einsum(
            "b n k s a, b n k a -> b n s a",
            einops.rearrange(v + pos_enc, "b n k (s a) -> b n k s a", s=self.share_planes),
            attn
        )
        res = einops.rearrange(res, "b n s a -> b n (s a)")
        return res

# ============================================================
# Point Transformer Block
# ============================================================

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = BatchNorm1d_P(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
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
        features += identity
        features = self.relu(features)
        return [xyz, features]

# ============================================================
# Transition Down
# ============================================================

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride, nneighbor=16):
        super().__init__()
        self.stride, self.nneighbor = stride, nneighbor
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.bn = BatchNorm2d_P(out_planes)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.bn = BatchNorm1d_P(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, px):
        xyz, features = px
        if self.stride == 1:
            new_features = self.relu(self.bn(self.linear(features)))
            return [xyz, new_features]

        npoint = xyz.size(1) // self.stride
        fps_idx = pointops.furthestsampling(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        knn_idx = pointops.knnquery(self.nneighbor, xyz, new_xyz)
        grouped_xyz = index_points(xyz, knn_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]
        grouped_features = torch.cat([grouped_xyz_norm, index_points(features, knn_idx)], dim=-1)
        grouped_features = self.relu(self.bn(self.linear(grouped_features)))
        new_features = torch.max(grouped_features, dim=2)[0]
        return [new_xyz, new_features]

# ============================================================
# Transition Up
# ============================================================

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes),
                                         BatchNorm1d_P(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes),
                                         nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes),
                                         BatchNorm1d_P(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes),
                                         BatchNorm1d_P(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, px1, px2=None):
        if px2 is None:
            p1, x1 = px1
            x_mean = self.linear2(x1.mean(dim=1, keepdim=True)).repeat(1, x1.size(1), 1)
            x = torch.cat([x1, x_mean], dim=2)
            x = self.linear1(x)
        else:
            p1, x1 = px1
            p2, x2 = px2
            dist, idx = pointops.nearestneighbor(p1, p2)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_x = pointops.interpolation(self.linear2(x2).permute(0, 2, 1), idx, weight).permute(0, 2, 1)
            x = interpolated_x + self.linear1(x1)
        return x

# ============================================================
# Point Transformer Classification Network
# ============================================================

class PointTransformerCls(nn.Module):
    def __init__(self, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes, share_planes = in_channels, [32, 64, 128, 256, 512], 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))

        # Encoders
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, N, in_channels)
        """
        B = x.size(0)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, in_channels)
        xyz_cls = torch.zeros(B, 1, 3, device=x.device)  # dummy xyz for CLS token

        x = torch.cat([cls_tokens, x], dim=1)           # (B, N+1, in_channels)
        xyz = torch.cat([xyz_cls, x[..., :3]], dim=1)   # (B, N+1, 3)

        # Encoder forward pass
        xyz1, features1 = self.enc1([xyz, x])
        xyz2, features2 = self.enc2([xyz1, features1])
        xyz3, features3 = self.enc3([xyz2, features2])
        xyz4, features4 = self.enc4([xyz3, features3])
        xyz5, features5 = self.enc5([xyz4, features4])

        # Use CLS token for classification
        cls_features = features5[:, 0, :]               # first token
        res = self.cls(cls_features)
        return res

# ============================================================
# Variants
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


