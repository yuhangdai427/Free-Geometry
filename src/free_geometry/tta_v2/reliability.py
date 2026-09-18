"""Protocol-v2 phase-C teacher-target reliability weights (pure torch).

Cross-checking the frozen teacher against itself: every scene task trains on one
shared frame group S seen under TWO different teacher contexts A and B (see
scripts/build_ab_context_manifests.py). The two frozen teacher caches of S carry
independent context noise; where they disagree, the teacher target for that
patch / view-graph edge is unreliable. This module turns cache-vs-cache
disagreement into frozen reliability weights.

Frozen semantics: every weight here is computed ONCE from the two teacher caches
BEFORE any student adaptation step and held fixed for the whole scene run.
Nothing reads student outputs or student residuals, and nothing writes
global/module state — all functions are pure computations of their arguments.

Weight form (both levels): u = raw disagreement, tau = median(u) (recorded),
q = 1 / (1 + (u/tau)^2) in (0, 1]. A zero median (half or more entries agree
exactly) falls back to tau = mean(u) so the disagreeing entries still get
down-weighted instead of dividing by zero; when even the mean is zero (total
agreement) q is identically 1.
"""
import torch
import torch.nn.functional as F

from .robust_losses import edges_from_w2c

_TAU_EPS = 1e-8
# Disagreement at or below float32 cosine/matmul numerical noise carries no
# signal: treat it as exact agreement (u == 0) so identical caches give q == 1
# exactly instead of q ~ 0.5 from a noise-level median.
_U_FLOOR = 1e-6


def _floor_deviation(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u < _U_FLOOR, torch.zeros_like(u), u)


def _deviation_weights(u: torch.Tensor):
    """Cauchy reliability from a deviation map of any shape.

    Returns (q, tau_used, tau_median): q = 1/(1+(u/tau)^2) with
    tau = median(u) unless that is ~0, in which case tau = max(mean(u), eps).
    """
    tau_med = u.median()
    if float(tau_med) >= _TAU_EPS:
        tau = tau_med
    else:
        tau = u.mean().clamp_min(_TAU_EPS)
    q = 1.0 / (1.0 + (u / tau) ** 2)
    return q, tau, tau_med


def feature_reliability(feats_A: torch.Tensor, feats_B: torch.Tensor):
    """Per-patch teacher-feature reliability from the A/B cache disagreement.

    feats_A/feats_B: [S,P,C] or [1,S,P,C] frozen teacher features of the SAME
    S shared views (one tapped layer; call per layer). Per patch
    u = 1 - cos(z_A, z_B) in [0, 2]; q = 1/(1+(u/tau)^2) with tau = median(u)
    (recorded in stats["tau"]; stats["tau_used"] is the effective scale after
    the zero-median fallback, see module docstring).

    Returns (q [S,P], stats). Frozen before adaptation; student residuals never
    enter. Identical caches -> u == 0 -> q == 1 everywhere. Deviations below
    float32 noise (_U_FLOOR) are treated as exact agreement.
    """
    if feats_A.dim() == 4:
        feats_A = feats_A[0]
    if feats_B.dim() == 4:
        feats_B = feats_B[0]
    if feats_A.shape != feats_B.shape:
        raise ValueError(f"feats_A {tuple(feats_A.shape)} != feats_B {tuple(feats_B.shape)}")
    u = 1.0 - F.cosine_similarity(feats_A.float(), feats_B.float(), dim=-1)  # [S,P]
    u = _floor_deviation(u)
    q, tau, tau_med = _deviation_weights(u)
    stats = {"tau": float(tau_med), "tau_used": float(tau),
             "u_mean": float(u.mean()), "q_mean": float(q.mean())}
    return q, stats


def rotation_reliability(ext_A: torch.Tensor, ext_B: torch.Tensor):
    """Per-view-graph-edge relative-rotation reliability (gauge-free).

    ext_A/ext_B: w2c extrinsics of the SAME S shared views under context A /
    context B: [S,3,4], [S,4,4] or [1,S,...]. Per edge (i,j) the disagreement is
    u^R_ij = Angle(Q^A_ij, Q^B_ij) with Q = R_i @ R_j^T the relative rotation —
    a relative quantity, so a global gauge difference between the two caches
    cancels and only context noise remains. q = 1/(1+(u/tau)^2) with
    tau = median(u), same frozen semantics as feature_reliability.

    Alongside the rotation reliability, returns per-edge baseline-consistency
    statistics between the two caches (the relative translations t_i - Q t_j):
      tdir_inconsistency [E]  1 - cos(t_A_hat, t_B_hat)   (direction)
      baseline_log_ratio [E]  |log(||t_A|| / ||t_B||)|    (length)
    (camera-center baselines themselves are gauge-free; returned as
    baseline_t for reference.)

    Returns (q [E], stats) with tau/tau_used/angle_deg [E]/u_rad [E]/q_mean plus
    edge_index [E,2], baseline_t, tdir_inconsistency, baseline_log_ratio.
    Frozen before adaptation; student residuals never enter.
    """
    eA = edges_from_w2c(ext_A)
    eB = edges_from_w2c(ext_B)
    z = ((eA["R_rel"] - eB["R_rel"]) ** 2).sum(dim=(-2, -1))  # [E]
    d = (z / 8.0).clamp(0.0, 1.0).sqrt()  # |sin(phi/2)|, robust_losses convention
    u = 2.0 * torch.asin(d)  # rad
    u = _floor_deviation(u)
    q, tau, tau_med = _deviation_weights(u)
    tA, tB = eA["t_rel"], eB["t_rel"]
    tdir = 1.0 - (F.normalize(tA, dim=-1, eps=1e-8) * F.normalize(tB, dim=-1, eps=1e-8)).sum(dim=-1)
    len_ratio = torch.log(tA.norm(dim=-1).clamp_min(1e-12)
                          / tB.norm(dim=-1).clamp_min(1e-12)).abs()
    stats = {"tau": float(tau_med), "tau_used": float(tau),
             "u_rad": u, "angle_deg": u * (180.0 / 3.141592653589793),
             "q_mean": float(q.mean()),
             "edge_index": eA["edge_index"], "baseline_t": eA["baseline"],
             "tdir_inconsistency": tdir, "baseline_log_ratio": len_ratio}
    return q, stats


def camera_task_weight(q_R_per_edge: torch.Tensor) -> float:
    """Scene/task-level geometric reliability: plain mean of the per-edge
    rotation reliability from rotation_reliability. float in (0, 1]; frozen
    like its input. Empty input is a hard error (a task with no edges has no
    geometric signal at all)."""
    if q_R_per_edge.numel() == 0:
        raise ValueError("camera_task_weight: empty q_R_per_edge")
    return float(q_R_per_edge.float().mean())
