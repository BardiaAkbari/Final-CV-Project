# pointops_pytorch.py
import torch


def batched_index_select(x, idx):
    """
    x: (N, C)
    idx: (M, K)
    return: (M, K, C)
    """
    M, K = idx.shape
    idx_flat = idx.reshape(-1)
    out = x[idx_flat]
    return out.view(M, K, -1)


def furthest_point_sampling(x, npoint):
    """
    x: (N, 3)
    npoint: int
    return: (npoint,) indices
    """
    device = x.device
    N = x.shape[0]
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.full((N,), 1e10, device=device)
    farthest = torch.randint(0, N, (1,), device=device).item()

    for i in range(npoint):
        centroids[i] = farthest
        centroid = x[farthest].view(1, 3)
        dist = torch.sum((x - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=0)[1].item()

    return centroids


def furthestsampling(p, o, n_o):
    """
    p: (N, 3)
    o: (B,)
    n_o: (B,)
    """
    idx_all = []
    start_src = 0
    start_dst = 0

    for i in range(o.shape[0]):
        end_src = o[i].item()
        end_dst = n_o[i].item()

        pts = p[start_src:end_src]
        npoint = end_dst - start_dst

        fps_idx = furthest_point_sampling(pts, npoint)
        idx_all.append(fps_idx + start_src)

        start_src = end_src
        start_dst = end_dst

    return torch.cat(idx_all, dim=0)


def knn_query(k, src_p, dst_p):
    """
    src_p: (Ns, 3)
    dst_p: (Nd, 3)
    return: (Nd, k) indices
    """
    dist = torch.cdist(dst_p, src_p)  # (Nd, Ns)
    idx = dist.topk(k, largest=False)[1]
    return idx


def queryandgroup(nsample, src_p, dst_p, src_x, _, src_o, dst_o, use_xyz=True):
    """
    src_p: (Ns, 3)
    dst_p: (Nd, 3)
    src_x: (Ns, C) or None
    """
    device = src_p.device
    grouped_feats = []

    src_start = 0
    dst_start = 0

    for i in range(src_o.shape[0]):
        src_end = src_o[i].item()
        dst_end = dst_o[i].item()

        sp = src_p[src_start:src_end]
        dp = dst_p[dst_start:dst_end]

        idx = knn_query(nsample, sp, dp)  # (Nd_i, nsample)
        idx = idx + src_start

        grouped_p = batched_index_select(src_p, idx)  # (Nd_i, nsample, 3)
        grouped_p = grouped_p - dp.unsqueeze(1)

        if src_x is not None:
            grouped_x = batched_index_select(src_x, idx)
            if use_xyz:
                grouped = torch.cat([grouped_p, grouped_x], dim=-1)
            else:
                grouped = grouped_x
        else:
            grouped = grouped_p

        grouped_feats.append(grouped)

        src_start = src_end
        dst_start = dst_end

    return torch.cat(grouped_feats, dim=0)
