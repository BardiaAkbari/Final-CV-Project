# models/pytorch_pointops.py
import torch


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _pad_idx(idx, k):
    """
    idx: (M, K0), K0 <= k
    return: (M, k)
    """
    M, K0 = idx.shape
    if K0 == k:
        return idx
    pad = idx[:, :1].repeat(1, k - K0)
    return torch.cat([idx, pad], dim=1)


def _batch_offsets_to_ranges(offset):
    starts = torch.cat([offset.new_zeros(1), offset[:-1]])
    ends = offset
    return starts.tolist(), ends.tolist()


# ------------------------------------------------------------
# Furthest Sampling (pure PyTorch)
# ------------------------------------------------------------

def furthestsampling(xyz, offset, new_offset):
    """
    xyz: (N, 3)
    offset: (B,)
    new_offset: (B,)
    return: (sum(new_offset),)
    """
    device = xyz.device
    idx_all = []

    src_starts, src_ends = _batch_offsets_to_ranges(offset)
    dst_starts, dst_ends = _batch_offsets_to_ranges(new_offset)

    for s0, s1, d0, d1 in zip(src_starts, src_ends, dst_starts, dst_ends):
        pts = xyz[s0:s1]
        npoint = d1 - d0

        N = pts.shape[0]
        centroids = torch.zeros(npoint, dtype=torch.long, device=device)
        dist = torch.full((N,), 1e10, device=device)

        farthest = torch.randint(0, N, (1,), device=device).item()
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pts[farthest:farthest+1]
            d = ((pts - centroid) ** 2).sum(-1)
            mask = d < dist
            dist[mask] = d[mask]
            farthest = dist.argmax().item()

        idx_all.append(centroids + s0)

    return torch.cat(idx_all, dim=0)


# ------------------------------------------------------------
# KNN Query (SAFE, CUDA-equivalent)
# ------------------------------------------------------------

def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    """
    xyz: (N, 3)
    new_xyz: (M, 3)
    offset: (B,)
    new_offset: (B,)
    return:
        idx: (M, nsample)
        dist: (M, nsample)
    """
    device = xyz.device
    idx_all, dist_all = [], []

    src_starts, src_ends = _batch_offsets_to_ranges(offset)
    dst_starts, dst_ends = _batch_offsets_to_ranges(new_offset)

    for s0, s1, d0, d1 in zip(src_starts, src_ends, dst_starts, dst_ends):
        src = xyz[s0:s1]        # (Ns, 3)
        dst = new_xyz[d0:d1]    # (Nd, 3)

        dist = torch.cdist(dst, src)  # (Nd, Ns)
        Ns = src.shape[0]

        k = min(nsample, Ns)
        d_k, idx_k = dist.topk(k, largest=False)

        if k < nsample:
            idx_k = _pad_idx(idx_k, nsample)
            d_k = _pad_idx(d_k, nsample)

        idx_all.append(idx_k + s0)
        dist_all.append(d_k)

    return torch.cat(idx_all, 0), torch.cat(dist_all, 0)


# ------------------------------------------------------------
# Grouping
# ------------------------------------------------------------

def grouping(input, idx):
    """
    input: (N, C)
    idx: (M, nsample)
    return: (M, nsample, C)
    """
    return input[idx.view(-1)].view(idx.shape[0], idx.shape[1], -1)


# ------------------------------------------------------------
# Query and Group (USED BY YOUR MODEL)
# ------------------------------------------------------------

def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    output:
      (M, nsample, C + 3) if use_xyz
      (M, nsample, C)     otherwise
    """
    if new_xyz is None:
        new_xyz = xyz

    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)

    grouped_xyz = xyz[idx.view(-1)].view(idx.shape[0], nsample, 3)
    grouped_xyz = grouped_xyz - new_xyz.unsqueeze(1)

    if feat is not None:
        grouped_feat = feat[idx.view(-1)].view(idx.shape[0], nsample, -1)
        if use_xyz:
            return torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            return grouped_feat
    else:
        return grouped_xyz


# ------------------------------------------------------------
# Subtraction
# ------------------------------------------------------------

def subtraction(input1, input2, idx):
    """
    input1, input2: (N, C)
    idx: (N, nsample)
    return: (N, nsample, C)
    """
    return input1.unsqueeze(1) - input2[idx]


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

def aggregation(input, position, weight, idx):
    """
    input: (N, C)
    position: (N, nsample, C)
    weight: (N, nsample, Cw)
    idx: (N, nsample)
    return: (N, C)
    """
    weighted = (input[idx] + position) * weight.sum(-1, keepdim=True)
    return weighted.sum(1)


# ------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------

def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)
    dist = torch.clamp(dist, min=1e-8)
    w = 1.0 / dist
    w = w / w.sum(1, keepdim=True)
    return (feat[idx] * w.unsqueeze(-1)).sum(1)



# ------------------------------------------------------------
# Ball Query (pure PyTorch, SAFE)
# ------------------------------------------------------------

def ball_query(radius, nsample, xyz, new_xyz, offset, new_offset):
    """
    xyz: (N, 3)
    new_xyz: (M, 3)
    offset: (B,)
    new_offset: (B,)
    return:
        idx: (M, nsample)
    """
    device = xyz.device
    idx_all = []

    src_starts, src_ends = _batch_offsets_to_ranges(offset)
    dst_starts, dst_ends = _batch_offsets_to_ranges(new_offset)

    radius2 = radius * radius

    for s0, s1, d0, d1 in zip(src_starts, src_ends, dst_starts, dst_ends):
        src = xyz[s0:s1]          # (Ns, 3)
        dst = new_xyz[d0:d1]      # (Nd, 3)
        Ns = src.shape[0]
        Nd = dst.shape[0]

        # squared distances: (Nd, Ns)
        dist2 = torch.cdist(dst, src, p=2) ** 2

        idx_batch = torch.zeros((Nd, nsample), dtype=torch.long, device=device)

        for i in range(Nd):
            mask = dist2[i] <= radius2
            idx_in_ball = torch.nonzero(mask, as_tuple=False).view(-1)

            if idx_in_ball.numel() == 0:
                # no neighbors → default to first point
                idx_batch[i].fill_(0)
            elif idx_in_ball.numel() >= nsample:
                idx_batch[i] = idx_in_ball[:nsample]
            else:
                # pad with first valid neighbor
                pad = idx_in_ball[0].repeat(nsample - idx_in_ball.numel())
                idx_batch[i] = torch.cat([idx_in_ball, pad], dim=0)

        idx_all.append(idx_batch + s0)

    return torch.cat(idx_all, dim=0)
