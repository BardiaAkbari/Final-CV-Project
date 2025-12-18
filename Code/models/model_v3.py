import torch
import torch.nn as nn
from lib.pointops.functions import pointops


# ============================================================
# Point Transformer Layer (UNCHANGED)
# ============================================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, self.mid_planes)
        self.linear_k = nn.Linear(in_planes, self.mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes)
        )

        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(self.mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_planes, self.mid_planes // share_planes),
            nn.BatchNorm1d(self.mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo  # (n,3), (n,c), (b)

        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)

        x_k = pointops.queryandgroup(
            self.nsample, p, p, x_k, None, o, o, use_xyz=True
        )
        x_v = pointops.queryandgroup(
            self.nsample, p, p, x_v, None, o, o, use_xyz=False
        )

        p_r, x_k = x_k[:, :, :3], x_k[:, :, 3:]

        for i, layer in enumerate(self.linear_p):
            p_r = (
                layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                if i == 1 else layer(p_r)
            )

        w = x_k - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0], p_r.shape[1],
            self.out_planes // self.mid_planes,
            self.mid_planes
        ).sum(2)

        for i, layer in enumerate(self.linear_w):
            w = (
                layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                if i % 3 == 0 else layer(w)
            )

        w = self.softmax(w)

        n, nsample, c = x_v.shape
        s = self.share_planes

        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1)
        return x.view(n, c)


# ============================================================
# Transition Down (UNCHANGED)
# ============================================================

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample

        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)

        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo

        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)

            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]

            x = pointops.queryandgroup(
                self.nsample, p, n_p, x, None, o, n_o, use_xyz=True
            )
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)

            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))

        return [p, x, o]


# ============================================================
# Point Transformer Block (UNCHANGED)
# ============================================================

class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.transformer = PointTransformerLayer(
            planes, planes, share_planes, nsample
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x

        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))

        x += identity
        x = self.relu(x)
        return [p, x, o]


# ============================================================
# Point Transformer Classification Network
# ============================================================

class PointTransformerCls(nn.Module):
    def __init__(self, blocks, c=6, num_classes=40):
        super().__init__()
        self.c = c
        self.in_planes = c

        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        self.cls_head = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes, stride, nsample))
        self.in_planes = planes
        for _ in range(blocks):
            layers.append(PointTransformerBlock(
                self.in_planes, self.in_planes, share_planes, nsample
            ))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p, x, o = pxo
        x = p if self.c == 3 else torch.cat((p, x), 1)

        p, x, o = self.enc1([p, x, o])
        p, x, o = self.enc2([p, x, o])
        p, x, o = self.enc3([p, x, o])
        p, x, o = self.enc4([p, x, o])
        p, x, o = self.enc5([p, x, o])

        # Global average pooling per batch
        feats = []
        for i in range(o.shape[0]):
            s = 0 if i == 0 else o[i-1]
            e = o[i]
            feats.append(x[s:e].mean(0, keepdim=True))
        x = torch.cat(feats, 0)

        return self.cls_head(x)


# ============================================================
# Model Variants
# ============================================================

def pointtransformer_cls26(**kwargs):
    return PointTransformerCls([1, 1, 1, 1, 1], **kwargs)

def pointtransformer_cls38(**kwargs):
    return PointTransformerCls([1, 2, 2, 2, 2], **kwargs)

def pointtransformer_cls50(**kwargs):
    return PointTransformerCls([1, 2, 3, 5, 2], **kwargs)
