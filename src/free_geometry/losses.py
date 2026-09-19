"""Five independent terms, one scalar backward, fixed teacher-only supervision."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LOSS_NAMES, LossConfig, ReliabilityConfig
from .geometry import couple_stat, edges, rkd_statistics, rotation_angle


def _finite(x):
    return torch.isfinite(x)


def _require(mask, name):
    if not bool(mask.any()):
        raise ValueError(f"{name}: no valid teacher supervision")


def _safe(x):
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _q(disagreement, threshold, available):
    # Numerical cosine/matmul noise is not evidence of disagreement.
    d = torch.where(
        disagreement.abs() < 1e-6,
        torch.zeros_like(disagreement),
        disagreement.clamp_min(0),
    )
    return torch.where(
        available, 1 / (1 + (d / threshold).square()), torch.ones_like(d)
    ).detach()


def _weighted(per, mask, q):
    _require(mask, "loss")
    return (per[mask] * q[mask]).mean()


@dataclass
class LossBundle:
    total: torch.Tensor
    raw: dict
    weighted: dict
    contributions: dict
    counts: dict

    def record(self):
        return {
            "total": float(self.total.detach()),
            "raw": {k: float(v.detach()) for k, v in self.raw.items()},
            "weighted": {k: float(v.detach()) for k, v in self.weighted.items()},
            "contributions": {
                k: float(v.detach()) for k, v in self.contributions.items()
            },
            "counts": self.counts,
        }


@torch.no_grad()
def prepare_supervision(a, b=None, loss=None, reliability=None, informative=True):
    """Cache all masks and q once; CPU-compatible, no student-derived gating."""
    loss = loss or LossConfig()
    reliability = reliability or ReliabilityConfig()
    a.check()
    if b is not None:
        b.check()
        if (
            a.frame_ids != b.frame_ids
            or a.patch_hw != b.patch_hw
            or a.readouts.keys() != b.readouts.keys()
        ):
            raise ValueError("A/B frame/readout mapping mismatch")
    use_b = b is not None and informative and reliability.enabled
    cache = {
        "q": {},
        "masks": {},
        "targets": {},
        "availability": {},
        "ab_informative": bool(informative and b is not None),
    }
    q, masks, targets = cache["q"], cache["masks"], cache["targets"]
    ph, pw = a.patch_hw
    real = a.valid.bool()
    finite_conf = _finite(a.conf) & (a.conf >= 0) & real
    conf = torch.where(finite_conf, a.conf.float(), torch.zeros_like(a.conf.float()))
    cp = F.adaptive_avg_pool2d(conf[:, None], (ph, pw)).flatten(1)[None]
    # Exclude patches containing padding/nonfinite confidence, keeping positional semantics.
    patch_valid = (
        F.adaptive_avg_pool2d(finite_conf.float()[:, None], (ph, pw)).flatten(1)[None]
        >= 1 - 1e-6
    )
    if bool(patch_valid.any()):
        cp = cp / cp[patch_valid].mean().clamp_min(1e-8)
    cache["confidence"] = cp
    if loss.feature:
        q["feature"], masks["feature"], targets["feature"] = {}, {}, {}
        for name, fa in a.readouts.items():
            mask = patch_valid & _finite(fa).all(-1) & (cp > 0)
            _require(mask, "feature:" + name)
            masks["feature"][name] = mask
            targets["feature"][name] = _safe(fa)
            available = torch.zeros_like(mask)
            weight = torch.ones_like(cp)
            if use_b and reliability.feature:
                fb = b.readouts[name]
                if fb.shape != fa.shape:
                    raise ValueError("A/B feature shape mismatch")
                available = mask & _finite(fb).all(-1)
                d = 1 - F.cosine_similarity(_safe(fa), _safe(fb), dim=-1)
                weight = _q(d, reliability.feature_threshold, available)
            q["feature"][name] = weight
            cache["availability"]["feature:" + name] = int(available.sum())
    ext = _safe(a.ext_w2c)
    ra, ta, ii, jj = edges(ext)
    finite_pose = _finite(a.ext_w2c).all((-2, -1))
    pose_mask = finite_pose[ii] & finite_pose[jj]
    rb, tb = None, None
    bpose = torch.zeros_like(pose_mask)
    if use_b:
        rb, tb, _, _ = edges(_safe(b.ext_w2c))
        bp = _finite(b.ext_w2c).all((-2, -1))
        bpose = bp[ii] & bp[jj]
    if loss.rotation:
        _require(pose_mask, "rotation")
        targets["rotation"], masks["rotation"] = ra, pose_mask
        available = pose_mask & bpose
        q["rotation"] = (
            _q(
                rotation_angle(ra, rb),
                math.radians(reliability.rotation_threshold_deg),
                available,
            )
            if use_b and reliability.rotation
            else torch.ones_like(ta[:, 0])
        )
        cache["availability"]["rotation"] = (
            int(available.sum()) if use_b and reliability.rotation else 0
        )
    if loss.translation:
        baseline = ta.norm(dim=-1)
        ref = (
            baseline[pose_mask].mean()
            if bool(pose_mask.any())
            else baseline.new_tensor(0.0)
        )
        keep = pose_mask & (baseline > 1e-12) & (baseline >= 1e-3 * ref)
        _require(keep, "translation")
        targets["translation"], masks["translation"] = F.normalize(ta, dim=-1), keep
        available = keep & bpose
        weight = torch.ones_like(baseline)
        if use_b and reliability.translation:
            bn = tb.norm(dim=-1)
            bref = bn[bpose].mean() if bool(bpose.any()) else bn.new_tensor(0.0)
            available &= (bn > 1e-12) & (bn >= 1e-3 * bref)
            cosine = (
                (F.normalize(ta, dim=-1) * F.normalize(tb, dim=-1)).sum(-1).clamp(-1, 1)
            )
            cosine = torch.where(cosine > 1 - 1e-6, torch.ones_like(cosine), cosine)
            weight = _q(
                cosine.acos(),
                math.radians(reliability.translation_threshold_deg),
                available,
            )
        q["translation"] = weight
        cache["availability"]["translation"] = (
            int(available.sum()) if use_b and reliability.translation else 0
        )
    if loss.rkd:
        # Keep nonfinite centers as NaN while determining validity, sanitize only afterwards.
        da, aa, md, ma, _triples = rkd_statistics(a.centers.float())
        _require(md, "rkd_distance")
        if len(a.centers) >= 3:
            _require(ma, "rkd_angle")
        targets["rkd_distance"], targets["rkd_angle"] = _safe(da), _safe(aa)
        masks["rkd_distance"], masks["rkd_angle"] = md, ma
        for key, value in [("rkd_distance", da), ("rkd_angle", aa)]:
            q[key] = torch.ones_like(value)
            cache["availability"][key] = 0
        if use_b and reliability.rkd:
            db, ab, mbd, mba, _ = rkd_statistics(b.centers.float())
            for key, x, y, available, threshold in (
                ("rkd_distance", da, db, md & mbd, reliability.rkd_distance_threshold),
                ("rkd_angle", aa, ab, ma & mba, reliability.rkd_angle_threshold),
            ):
                q[key] = _q((_safe(x) - _safe(y)).abs(), threshold, available)
                cache["availability"][key] = int(available.sum())
    if loss.couple:
        support = real & finite_conf & _finite(a.depth) & (a.depth > 0)
        _require(support, "couple")
        threshold = torch.quantile(a.conf[support].float(), 0.05)
        support &= a.conf >= threshold
        stat = couple_stat(a.centers.float(), a.depth.float(), support)
        masks["couple"], targets["couple"] = support, stat
        q["couple"] = stat.new_ones(())
        cache["availability"]["couple"] = 0
        if use_b and reliability.couple:
            try:
                if not bool(b.valid[support].all()):
                    raise ValueError("B lacks A support")
                other = couple_stat(b.centers.float(), b.depth.float(), support)
                q["couple"] = _q(
                    (stat - other).abs(),
                    reliability.couple_threshold,
                    stat.new_tensor(True, dtype=torch.bool),
                )
                cache["availability"]["couple"] = 1
            except ValueError:
                pass  # q=1 means unavailable comparison, not reliable teacher
    return cache


def compute_losses(student, teacher, cache, config=None):
    config = config or LossConfig()
    student.check()
    if student.frame_ids != teacher.frame_ids or student.patch_hw != teacher.patch_hw:
        raise ValueError("student/teacher frame or patch correspondence mismatch")
    raw, weighted, counts = {}, {}, {}
    targets, masks, q = cache["targets"], cache["masks"], cache["q"]

    def record(name, per, mask, weight):
        if not bool(_finite(per[mask]).all()):
            raise FloatingPointError(f"{name}: invalid student output")
        raw[name] = per[mask].mean()
        weighted[name] = _weighted(per, mask, weight)
        counts[name] = int(mask.sum())

    if config.feature:
        raws, ws, n = [], [], 0
        if student.readouts.keys() != teacher.readouts.keys():
            raise ValueError("feature readout names mismatch")
        for name, fs in student.readouts.items():
            ft = targets["feature"][name]
            mask = masks["feature"][name]
            if fs.shape != ft.shape or not bool(_finite(fs[mask]).all()):
                raise FloatingPointError("feature shape mismatch/nonfinite student")
            # Index before nonlinear ops so excluded NaN patches cannot poison gradients.
            x, y = fs.float()[mask], ft[mask]
            per = F.smooth_l1_loss(x, y, reduction="none", beta=1).mean(-1) + 2 * (
                1 - F.cosine_similarity(x, y, dim=-1)
            )
            base = cache["confidence"][mask] * per
            raws.append(base.mean())
            ws.append((base * q["feature"][name][mask]).mean())
            n += int(mask.sum())
        raw["feature"], weighted["feature"], counts["feature"] = (
            torch.stack(raws).mean(),
            torch.stack(ws).mean(),
            n,
        )
    if config.rotation or config.translation:
        if not bool(_finite(student.ext_w2c).all()):
            raise FloatingPointError("nonfinite student camera")
        rs, ts, _, _ = edges(student.ext_w2c.float())
        if config.rotation:
            z = (rs - targets["rotation"]).square().sum((-2, -1))
            delta = math.sin(math.radians(config.rotation_knee_deg) / 2)
            per = torch.where(
                z <= 8 * delta**2,
                z,
                16 * delta * ((z / 8).clamp_min(delta**2).sqrt() - 0.5 * delta),
            )
            record("rotation", per, masks["rotation"], q["rotation"])
        if config.translation:
            per = 1 - (F.normalize(ts, dim=-1, eps=1e-8) * targets["translation"]).sum(
                -1
            )
            record("translation", per, masks["translation"], q["translation"])
    if config.rkd:
        if not bool(_finite(student.centers).all()):
            raise FloatingPointError("nonfinite student centers")
        ds, aa, _, _, _ = rkd_statistics(student.centers.float())
        for key, value in [("rkd_distance", ds), ("rkd_angle", aa)]:
            mask = masks[key]
            if bool(mask.any()):
                per = F.huber_loss(
                    value, targets[key], reduction="none", delta=config.rkd_delta
                )
                record(key, per, mask, q[key])
        raw["rkd"] = sum(raw[k] for k in ("rkd_distance", "rkd_angle") if k in raw)
        weighted["rkd"] = sum(
            weighted[k] for k in ("rkd_distance", "rkd_angle") if k in weighted
        )
        counts["rkd"] = sum(counts.get(k, 0) for k in ("rkd_distance", "rkd_angle"))
    if config.couple:
        stat = couple_stat(
            student.centers.float(), student.depth.float(), masks["couple"]
        )
        raw["couple"] = (stat - targets["couple"]).square()
        weighted["couple"] = raw["couple"] * q["couple"]
        counts["couple"] = int(masks["couple"].sum())
    contributions = {
        name: weighted[name] * getattr(config, name)
        for name in LOSS_NAMES
        if getattr(config, name) > 0
    }
    total = sum(contributions.values())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite total loss")
    return LossBundle(total, raw, weighted, contributions, counts)


def supervision_summary(cache):
    def stats(v):
        if isinstance(v, dict):
            return {k: stats(x) for k, x in v.items()}
        return (
            {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max())}
            if v.numel()
            else None
        )

    return {
        "ab_informative": cache["ab_informative"],
        "q": stats(cache["q"]),
        "available_comparisons": cache["availability"],
        "comparison_error": cache.get("comparison_error"),
    }
