"""Coordinate primitives shared by all models and losses."""

import itertools

import torch
import torch.nn.functional as F


def centers_from_w2c(ext):
    return -(ext[..., :3, :3].transpose(-1, -2) @ ext[..., :3, 3, None]).squeeze(-1)


def edges(ext):
    ii, jj = torch.triu_indices(len(ext), len(ext), 1, device=ext.device)
    r, t = ext[:, :3, :3], ext[:, :3, 3]
    q = r[ii] @ r[jj].transpose(-1, -2)
    b = t[ii] - (q @ t[jj, :, None]).squeeze(-1)
    return q, b, ii, jj


def rotation_angle(a, b):
    d = ((a - b).square().sum((-2, -1)) / 8).clamp(0, 1).sqrt()
    return 2 * torch.asin(d)


def rkd_statistics(centers):
    n = len(centers)
    ii, jj = torch.triu_indices(n, n, 1, device=centers.device)
    dist = (centers[ii] - centers[jj]).norm(dim=-1)
    valid_dist = torch.isfinite(dist) & (dist > 1e-8)
    safe = torch.where(valid_dist, dist, torch.zeros_like(dist))
    norm = safe.sum() / valid_dist.sum().clamp_min(1)
    normalized = safe / norm.clamp_min(1e-8)
    triples = [
        (k, a, b)
        for k in range(n)
        for a, b in itertools.combinations([j for j in range(n) if j != k], 2)
    ]
    idx = torch.tensor(triples, device=centers.device, dtype=torch.long).reshape(-1, 3)
    u = centers[idx[:, 1]] - centers[idx[:, 0]]
    v = centers[idx[:, 2]] - centers[idx[:, 0]]
    valid_angle = (
        torch.isfinite(u).all(-1)
        & torch.isfinite(v).all(-1)
        & (u.norm(dim=-1) > 1e-8)
        & (v.norm(dim=-1) > 1e-8)
    )
    angle = (F.normalize(u, dim=-1, eps=1e-8) * F.normalize(v, dim=-1, eps=1e-8)).sum(
        -1
    )
    return normalized, angle, valid_dist, valid_angle, idx


def couple_stat(centers, depth, support):
    if not bool(support.any()):
        raise ValueError("couple: empty teacher support")
    selected = depth[support]
    if not bool(
        torch.isfinite(centers).all()
        and torch.isfinite(selected).all()
        and (selected > 0).all()
    ):
        raise ValueError("couple: invalid geometry/depth on fixed teacher support")
    variance = (centers - centers.mean(0)).square().sum(-1).mean()
    if float(variance.detach()) <= 1e-12:
        raise ValueError("couple: degenerate camera spread")
    return 0.5 * variance.log() - selected.mean().log()
