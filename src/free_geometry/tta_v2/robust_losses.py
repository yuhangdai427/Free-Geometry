"""Protocol-v2 robust loss terms, shared by the DA3 and VGGT TTA pipelines.

Pure torch, no model deps. All terms are GT-free (student vs frozen teacher).
Unless noted, computations keep the input dtype and nothing here detaches —
the caller decides where to stop grad (probe.evaluate runs under
torch.no_grad() anyway).
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# Huber knee in d = |sin(phi/2)| space; phi ~ 10 deg of relative rotation.
ROT_HUBER_DELTA = math.sin(math.radians(10.0))


def huber_cos_weighted(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(w * (huber + 2*(1-cos))).mean() over all leading positions.

    hs/ht: [..., C] with any leading dims; w: [...] matching leading dims
    (pass a mean-1 weight to recover the usual average). Per position:
    SmoothL1(beta=1) meaned over C plus 2*(1 - cosine similarity).

    Semantic fix vs losses._huber_cos: an all-zero w returns EXACTLY 0 here
    (the old form leaves a constant 2 from the cos term). When mean(w) == 1
    this is numerically identical to losses._huber_cos.
    """
    huber_t = F.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(dim=-1)
    cos_t = F.cosine_similarity(hs, ht, dim=-1)
    per = huber_t + 2.0 * (1.0 - cos_t)
    return (w * per).mean()


def edges_from_w2c(ext: torch.Tensor) -> Dict[str, torch.Tensor]:
    """All C(S,2) undirected edges (i<j) of the view graph from w2c extrinsics.

    ext: [S,3,4] or [1,S,3,4] (a leading batch dim is squeezed, matching
    losses.loss_rel_pose). Returns:
      R_rel      [E,3,3]  R_i @ R_j^T   (T_{i<-j} = E_i @ inv(E_j))
      t_rel      [E,3]    t_i - R_rel @ t_j
      baseline   [E]      camera-center distance (center = -R^T t)
      edge_index [E,2]    int64 (i, j) pairs, i < j, loop order (i outer)
    Convention identical to losses.loss_rel_pose (:114-118).
    """
    if ext.dim() == 4:
        ext = ext[0]
    R, t = ext[..., :3, :3], ext[..., :3, 3]
    S = R.shape[0]
    ii, jj = torch.triu_indices(S, S, offset=1, device=ext.device)
    R_rel = R[ii] @ R[jj].transpose(-1, -2)
    t_rel = t[ii] - torch.einsum("eij,ej->ei", R_rel, t[jj])
    centers = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    baseline = (centers[ii] - centers[jj]).norm(dim=-1)
    return {"R_rel": R_rel, "t_rel": t_rel, "baseline": baseline,
            "edge_index": torch.stack([ii, jj], dim=-1)}


def rot_edges_huber(Rr_s: torch.Tensor, Rr_t: torch.Tensor,
                    delta: float = ROT_HUBER_DELTA,
                    edge_w: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Huber-saturated per-edge relative-rotation loss (robust replacement for
    the raw Frobenius term in losses.loss_rel_pose).

    Rr_s/Rr_t: [E,3,3] relative rotations from edges_from_w2c (NOT detached
    here). Per edge: z = ||Rr_s - Rr_t||^2_F (sum over last two dims),
    d = sqrt(z/8) (= |sin(phi/2)| for a phi-angle relative rotation),
    l = 16 * H_delta(d) with the standard Huber H (0.5 d^2 for d <= delta,
    delta (d - 0.5 delta) above). In the small-residual regime l == z exactly
    (16 * 0.5 * d^2 = z); with the default delta the Huber knee sits at ~10 deg
    so a single gross outlier saturates instead of dominating the mean.

    edge_w: optional [E] weights (mean-~1 style, like conf weights); the loss
    is the plain weighted mean — None -> equal weights, all-zero w -> exactly 0.

    Returns {"loss": scalar, "per_edge": [E], "angle_deg": [E]} where
    angle_deg = 2*arcsin(clamp(d,0,1)) in degrees (the relative rotation
    angle, for probe records).
    """
    z = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1))
    d = (z / 8.0).clamp_min(1e-12).sqrt()  # clamp: sqrt'(0)=inf -> NaN grad at z=0
    l = torch.where(d <= delta, 0.5 * d ** 2, delta * (d - 0.5 * delta)) * 16.0
    angle_deg = 2.0 * torch.asin(d.clamp(0.0, 1.0)) * (180.0 / math.pi)
    if l.numel() == 0:
        loss = z.sum() * 0.0
    elif edge_w is None:
        loss = l.mean()
    else:
        loss = (l * edge_w).mean()
    return {"loss": loss, "per_edge": l, "angle_deg": angle_deg}


def tdir_cos_loss(t_s: torch.Tensor, t_t: torch.Tensor, baseline_t: torch.Tensor,
                  skip_ratio: float = 1e-3,
                  edge_w: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Per-edge translation-direction loss: 1 - cos(t_s_hat, t_t_hat).

    t_s/t_t: [E,3] relative translations (edges_from_w2c "t_rel"); baseline_t:
    [E] teacher camera-center distances. Edges whose teacher baseline <
    skip_ratio * mean(baseline_t) are dropped from the mean — this implements
    the intent of the losses.loss_rel_pose skip comment (:120) that the old
    code never actually enforced. NaN baselines are dropped too (the
    comparison is False), which is how an invalid teacher pose edge gets out.

    edge_w: optional [E] weights, same semantics as rot_edges_huber.

    Returns {"loss": scalar (exactly 0 if every edge is dropped), "per_edge":
    [E] weighted per-edge values (defined for dropped edges as well),
    "n_kept", "n_skipped"}.
    """
    keep = baseline_t >= skip_ratio * torch.nanmean(baseline_t)  # NaN-safe: one NaN baseline must not drop all edges
    tn_s = F.normalize(t_s, dim=-1, eps=1e-8)
    tn_t = F.normalize(t_t, dim=-1, eps=1e-8)
    per = 1.0 - (tn_s * tn_t).sum(dim=-1)
    if edge_w is not None:
        per = per * edge_w
    n_kept = int(keep.sum())
    n_skipped = baseline_t.shape[0] - n_kept
    loss = per[keep].mean() if n_kept else per.sum() * 0.0
    return {"loss": loss, "per_edge": per, "n_kept": n_kept, "n_skipped": n_skipped}


def couple_robust(centers_s: torch.Tensor, centers_t: torch.Tensor,
                  depth_s: torch.Tensor, depth_t: torch.Tensor,
                  valid_t: torch.Tensor, huber_delta: Optional[float] = None,
                  eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Gauge-coupling scalar, robustified: Huber(diff) instead of diff^2.

    stat(c, d) = log(spr) - log(md) with spr = RMS spread of the camera
    centers (sqrt(mean_s ||c_s - c_bar||^2)) and md = mean depth. BOTH depths
    are masked by valid_t (the fixed losses.loss_couple_centers semantics:
    the teacher-derived valid-pixel mask applies to the student side too).
    diff = stat_s - stat_t; huber_delta=None -> diff^2, else the standard
    Huber on diff (linear tails, so one bad step cannot dominate the trace).

    Guards (caller skips the term when "skipped"): spr < eps on either side,
    or an empty valid mask on either side -> {"loss": None, "skipped": True,
    "reason": ...}. The teacher side is NOT detached here.
    """
    def stat(centers, depth):
        spr = (centers - centers.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
        m = valid_t & torch.isfinite(depth) & (depth > 0)
        md = depth[m].mean().clamp_min(eps)
        return spr, md, m

    spr_s, md_s, m_s = stat(centers_s, depth_s)
    spr_t, md_t, m_t = stat(centers_t, depth_t)
    if not bool(m_s.any()):
        return {"loss": None, "skipped": True, "reason": "empty student valid mask"}
    if not bool(m_t.any()):
        return {"loss": None, "skipped": True, "reason": "empty teacher valid mask"}
    if float(spr_s) < eps or float(spr_t) < eps:
        return {"loss": None, "skipped": True,
                "reason": f"degenerate camera spread spr_s={float(spr_s):.2e} "
                          f"spr_t={float(spr_t):.2e}"}
    stat_s = torch.log(spr_s) - torch.log(md_s)
    stat_t = torch.log(spr_t) - torch.log(md_t)
    diff = stat_s - stat_t
    if huber_delta is None:
        loss = diff ** 2
    else:
        loss = torch.where(diff.abs() <= huber_delta, 0.5 * diff ** 2,
                           huber_delta * (diff.abs() - 0.5 * huber_delta))
    return {"loss": loss, "skipped": False, "reason": None,
            "stat_s": stat_s, "stat_t": stat_t, "diff": diff}
