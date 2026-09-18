"""Unified GT-free TTA losses operating on the adapter interface.

All feature terms run in each model's NATIVE readout space (adapter-provided,
already post-norm / post-projection); all geometry terms consume camera
CENTERS only — gauge-free, so no per-model coordinate plumbing lives here
(the adapter is responsible for producing centers in a consistent world frame).
"""
from typing import Dict, List

import torch
import torch.nn.functional as F

HUBER_BETA = 1.0


def _huber_cos(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """hs/ht: [B,S,P,C]; w: [B,S,P] normalized to mean 1. Huber(beta=1) + 2(1-cos)."""
    huber_t = F.smooth_l1_loss(hs, ht, beta=HUBER_BETA, reduction="none").mean(dim=-1)
    cos_t = F.cosine_similarity(hs, ht, dim=-1)
    return (huber_t * w).mean() + 2.0 * (1.0 - (cos_t * w).mean())


def loss_maskdistill(readouts_s: Dict[str, torch.Tensor], readouts_t: Dict[str, torch.Tensor],
                     conf_patch: torch.Tensor) -> torch.Tensor:
    """ALL-position masked-input distillation (2026-09-16 ablation winner).

    readouts_*: {name: [1,S,P,C]} in native readout space (teacher detached).
    conf_patch: [1,S,P] teacher depth-confidence per patch, mean-normalized.
    """
    w = conf_patch / conf_patch.mean().clamp_min(1e-8)
    total = 0.0
    for name in readouts_s:
        total = total + _huber_cos(readouts_s[name].float(),
                                   readouts_t[name].float().detach(), w)
    return total / max(1, len(readouts_s))


def loss_rkd_centers(centers_s: torch.Tensor, centers_t: torch.Tensor,
                     delta: float = 0.2) -> torch.Tensor:
    """RKD on camera centers: mean-normalized pairwise distances + triangle
    angles, per-residual Huber(delta). centers_*: [S,3] (teacher detached)."""
    cs, ct = centers_s.float(), centers_t.float().detach()
    S = cs.shape[0]
    if S < 3:
        return (cs - ct).sum() * 0.0
    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
    dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
    dloss = F.huber_loss(ds / ds.mean().clamp_min(1e-8), dt / dt.mean().clamp_min(1e-8), delta=delta)
    if S < 4:
        return dloss
    aterms = []
    for k in range(S):
        rest = [x for x in range(S) if x != k]
        for a, b in [(rest[0], rest[1]), (rest[0], rest[2]), (rest[1], rest[2])]:
            vs1 = F.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
            vs2 = F.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
            vt1 = F.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
            vt2 = F.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
            aterms.append(F.huber_loss((vs1 * vs2).sum(), (vt1 * vt2).sum(), delta=delta))
    return dloss + torch.stack(aterms).mean()


def loss_couple_centers(centers_s: torch.Tensor, depth_s: torch.Tensor,
                        centers_t: torch.Tensor, depth_t: torch.Tensor,
                        valid_t: torch.Tensor) -> torch.Tensor:
    """Gauge coupling scalar: (log RMS(centers) - log mean_depth), student vs
    teacher. FIX (2026-09-17): both sides take mean depth over the SAME
    teacher-derived valid-pixel mask — removes the old asymmetric conf filter.

    depth_*: [S,H,W]; valid_t: [S,H,W] bool from teacher confidence quantile.
    """
    def stat(centers, depth, valid):
        spr = (centers - centers.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
        d = depth.float()
        m = torch.isfinite(d) & (d > 0) & valid
        md = d[m].mean().clamp_min(1e-6)
        return torch.log(spr.clamp_min(1e-6)) - torch.log(md)

    cs = stat(centers_s.float(), depth_s, valid_t)   # student depth on SAME mask
    ct = stat(centers_t.float().detach(), depth_t, valid_t)
    return (cs - ct) ** 2


def valid_mask_from_conf(conf: torch.Tensor, q: float = 0.05) -> torch.Tensor:
    """conf: [S,H,W] teacher confidence -> bool mask keeping 1-q fraction."""
    c = conf.float()
    thr = torch.quantile(c.flatten(), q)
    return torch.isfinite(c) & (c >= thr)


def loss_rel_pose(ext_s: torch.Tensor, ext_t: torch.Tensor) -> torch.Tensor:
    """Corrected relative-pose loss for w2c inputs (2026-09-18 fix).

    FIX: the relative transform from camera j to camera i is
      T_{i<-j} = E_i @ inv(E_j)  (NOT inv(E_i) @ E_j)
    For w2c [R|t]:
      R_rel = R_i @ R_j^T
      t_rel = t_i - R_i @ R_j^T @ t_j

    Loss form (unchanged): per-pair rotation Frobenius + translation-direction
    1-cos, scale-free. ext_*: [S,3,4] or [1,S,3,4] w2c (student grad, teacher
    detached)."""
    if ext_s.dim() == 4:
        ext_s = ext_s[0]
    if ext_t.dim() == 4:
        ext_t = ext_t[0]
    R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
    R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
    S = R_s.shape[0]
    rot_loss, tdir_loss, npairs = 0.0, 0.0, 0
    for i in range(S):
        for j in range(i + 1, S):
            # Correct: T_{i<-j} = E_i @ inv(E_j)
            Rr_s = R_s[i] @ R_s[j].transpose(-1, -2)
            Rr_t = R_t[i] @ R_t[j].transpose(-1, -2)
            tr_s = t_s[i] - Rr_s @ t_s[j]
            tr_t = t_t[i] - Rr_t @ t_t[j]
            rot_loss = rot_loss + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            # Skip direction for near-zero baselines (teacher criterion)
            tn_t = F.normalize(tr_t.squeeze(-1) if tr_t.dim() > 1 else tr_t,
                               dim=-1, eps=1e-8)
            tn_s = F.normalize(tr_s.squeeze(-1) if tr_s.dim() > 1 else tr_s,
                               dim=-1, eps=1e-8)
            tdir_loss = tdir_loss + (1.0 - (tn_s * tn_t).sum(-1)).mean()
            npairs += 1
    return rot_loss / npairs + tdir_loss / npairs


def compute_arm_loss(arm: str, student_out: Dict, teacher_cache: Dict) -> torch.Tensor:
    """arm: 'm_allpos' | 'rkdc_allpos' | 'maskrel_allpos' | 'rkdcr_allpos'.
    All-position maskdistill base."""
    feat = loss_maskdistill(student_out["readouts"], teacher_cache["readouts"],
                            teacher_cache["conf_patch"])
    if arm == "m_allpos":
        return feat
    if arm == "maskrel_allpos":
        return feat + 1.0 * loss_rel_pose(student_out["ext_w2c"], teacher_cache["ext_w2c"])
    rkd = loss_rkd_centers(student_out["centers"], teacher_cache["centers"])
    cp = loss_couple_centers(student_out["centers"], student_out["depth"],
                             teacher_cache["centers"], teacher_cache["depth"],
                             teacher_cache["valid"])
    base = feat + 1.5 * rkd + 1.0 * cp
    if arm == "rkdcr_allpos":
        # rkdc + corrected rel (2026-09-18: E_i @ inv(E_j) construction)
        return base + 1.0 * loss_rel_pose(student_out["ext_w2c"], teacher_cache["ext_w2c"])
    return base
