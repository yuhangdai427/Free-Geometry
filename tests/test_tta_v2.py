import hashlib
import json
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from free_geometry.losses import _huber_cos, loss_rel_pose  # noqa: E402
from free_geometry.tta_v2 import (  # noqa: E402
    ControllerConfig, ProbeEvaluator, append_jsonl, couple_robust, edges_from_w2c,
    huber_cos_weighted, make_patch_mask, rel_change, rot_edges_huber, select,
    should_stop, stable_seed, tdir_cos_loss,
)


def _rand_rot(gen):
    q = torch.randn(4, generator=gen, dtype=torch.float64)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _axis_angle(axis, theta):
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    x, y, z = axis.tolist()
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64)
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _rand_ext(S, gen):
    R = torch.stack([_rand_rot(gen) for _ in range(S)])
    t = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    return torch.cat([R, t[..., None]], dim=-1)


# ---------------------------------------------------------------------------
# robust_losses
# ---------------------------------------------------------------------------
def test_rot_edges_huber_small_residual_matches_frobenius():
    gen = torch.Generator().manual_seed(0)
    E = 6
    R_t = torch.stack([_rand_rot(gen) for _ in range(E)])
    R_err = _axis_angle([0.3, -0.5, 1.0], math.radians(0.5))
    R_s = torch.matmul(R_err, R_t)
    out = rot_edges_huber(R_s, R_t)
    z = ((R_s - R_t) ** 2).sum(dim=(-2, -1))
    assert out["per_edge"].shape == (E,)
    torch.testing.assert_close(out["per_edge"], z, rtol=1e-5, atol=0.0)
    torch.testing.assert_close(out["angle_deg"], torch.full((E,), 0.5, dtype=torch.float64),
                               rtol=1e-6, atol=1e-8)


def test_rot_edges_huber_saturates_at_large_error():
    gen = torch.Generator().manual_seed(1)
    E = 4
    R_t = torch.stack([_rand_rot(gen) for _ in range(E)])
    R_err = _axis_angle([1.0, 0.0, 0.0], math.radians(90.0))
    R_s = torch.matmul(R_err, R_t)
    out = rot_edges_huber(R_s, R_t)
    z = ((R_s - R_t) ** 2).sum(dim=(-2, -1))
    torch.testing.assert_close(z, torch.full((E,), 4.0, dtype=torch.float64), rtol=0, atol=1e-10)
    delta = math.sin(math.radians(10.0))
    d = math.sqrt(4.0 / 8.0)
    manual = 16.0 * delta * (d - 0.5 * delta)
    assert manual == pytest.approx(1.7235, abs=1e-3)
    torch.testing.assert_close(out["per_edge"], torch.full((E,), manual, dtype=torch.float64),
                               rtol=1e-9, atol=1e-12)
    assert bool((out["per_edge"] < z).all())
    torch.testing.assert_close(out["angle_deg"], torch.full((E,), 90.0, dtype=torch.float64),
                               rtol=1e-9, atol=1e-10)
    # zero weights -> exactly 0
    zero_w = rot_edges_huber(R_s, R_t, edge_w=torch.zeros(E, dtype=torch.float64))
    assert zero_w["loss"] == 0.0


def test_huber_cos_zero_weight_exact_and_one_weight_handcalc():
    gen = torch.Generator().manual_seed(2)
    hs = torch.randn(2, 3, 8, generator=gen, dtype=torch.float64)
    ht = torch.randn(2, 3, 8, generator=gen, dtype=torch.float64)
    w0 = torch.zeros(2, 3, dtype=torch.float64)
    assert huber_cos_weighted(hs, ht, w0) == 0.0
    w1 = torch.ones(2, 3, dtype=torch.float64)
    huber_t = F.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(dim=-1)
    cos_t = F.cosine_similarity(hs, ht, dim=-1)
    hand = (huber_t + 2.0 * (1.0 - cos_t)).mean()
    assert huber_cos_weighted(hs, ht, w1) == hand
    # mean-1 weight: numerically identical to the legacy _huber_cos
    w = torch.rand(2, 3, generator=gen, dtype=torch.float64)
    w = w / w.mean()
    legacy = _huber_cos(hs[None], ht[None], w[None])
    assert float(legacy) == pytest.approx(float(huber_cos_weighted(hs, ht, w)), rel=1e-12)


def test_tdir_skips_near_zero_baseline_and_all_skipped():
    gen = torch.Generator().manual_seed(3)
    t_t = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    t_s = t_t + 0.1 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    baseline = torch.tensor([1.0, 2.0, 1e-9], dtype=torch.float64)
    out = tdir_cos_loss(t_s, t_t, baseline, skip_ratio=1e-3)
    assert out["n_skipped"] == 1 and out["n_kept"] == 2
    tn_s = F.normalize(t_s[:2], dim=-1, eps=1e-8)
    tn_t = F.normalize(t_t[:2], dim=-1, eps=1e-8)
    hand = (1.0 - (tn_s * tn_t).sum(dim=-1)).mean()
    assert float(out["loss"]) == pytest.approx(float(hand), rel=1e-12)
    # NaN teacher baseline (invalid teacher geometry) drops every edge
    nan_out = tdir_cos_loss(t_s, t_t, torch.full((3,), float("nan"), dtype=torch.float64))
    assert nan_out["n_skipped"] == 3
    assert nan_out["loss"] == 0.0


def test_couple_robust_guard_and_handcalc():
    gen = torch.Generator().manual_seed(4)
    S = 4
    centers_s = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    centers_t = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    depth_s = torch.rand(S, 5, 6, generator=gen, dtype=torch.float64) + 0.05
    depth_t = torch.rand(S, 5, 6, generator=gen, dtype=torch.float64) + 0.05
    valid = torch.rand(S, 5, 6, generator=gen) < 0.6

    def stat(c, d):
        spr = (c - c.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
        md = d[valid & torch.isfinite(d) & (d > 0)].mean()
        return math.log(float(spr)) - math.log(float(md))

    out = couple_robust(centers_s, centers_t, depth_s, depth_t, valid)
    assert not out["skipped"]
    hand = (stat(centers_s, depth_s) - stat(centers_t, depth_t)) ** 2
    assert float(out["loss"]) == pytest.approx(hand, rel=1e-12)
    # both sides masked by valid_t: perturbing only invalid pixels changes nothing
    depth_s2 = depth_s.clone()
    depth_s2[~valid] = 123.0
    out2 = couple_robust(centers_s, centers_t, depth_s2, depth_t, valid)
    assert float(out2["loss"]) == pytest.approx(hand, rel=1e-12)
    # huber variant: linear tail matches the closed form and bounds diff^2
    out_h = couple_robust(centers_s, centers_t, depth_s, depth_t, valid, huber_delta=1e-6)
    diff = stat(centers_s, depth_s) - stat(centers_t, depth_t)
    assert float(out_h["loss"]) == pytest.approx(1e-6 * (abs(diff) - 0.5e-6), rel=1e-9)
    assert float(out_h["loss"]) <= float(out["loss"]) + 1e-15
    # guards: empty valid mask / degenerate spread
    out_empty = couple_robust(centers_s, centers_t, depth_s, depth_t,
                              torch.zeros_like(valid))
    assert out_empty["skipped"] and out_empty["loss"] is None
    out_deg = couple_robust(torch.zeros(S, 3, dtype=torch.float64), centers_t,
                            depth_s, depth_t, valid)
    assert out_deg["skipped"] and out_deg["loss"] is None


def test_edges_from_w2c_matches_loss_rel_pose_convention():
    gen = torch.Generator().manual_seed(5)
    S = 5
    ext = _rand_ext(S, gen)
    edges = edges_from_w2c(ext[None])  # exercise the [1,S,3,4] squeeze
    E = S * (S - 1) // 2
    assert edges["R_rel"].shape == (E, 3, 3)
    assert edges["t_rel"].shape == (E, 3)
    assert edges["baseline"].shape == (E,)
    assert edges["edge_index"].shape == (E, 2)
    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    assert [tuple(p) for p in edges["edge_index"].tolist()] == pairs
    # brute-force the losses.loss_rel_pose loop
    R, t = ext[..., :3, :3], ext[..., :3, 3]
    centers = -(R.transpose(-1, -2) @ t[..., None]).squeeze(-1)
    Rr, tr, base = [], [], []
    for i, j in pairs:
        Rr_ij = R[i] @ R[j].transpose(-1, -2)
        Rr.append(Rr_ij)
        tr.append(t[i] - Rr_ij @ t[j])
        base.append((centers[i] - centers[j]).norm())
    torch.testing.assert_close(edges["R_rel"], torch.stack(Rr), rtol=1e-10, atol=0.0)
    torch.testing.assert_close(edges["t_rel"], torch.stack(tr), rtol=1e-10, atol=0.0)
    torch.testing.assert_close(edges["baseline"], torch.stack(base), rtol=1e-10, atol=0.0)
    # end-to-end: loss_rel_pose decomposes into the edges' Frobenius + 1-cos means
    ext_s = ext.clone()
    ext_s[..., :3, 3] += 0.01 * torch.randn(S, 3, generator=gen, dtype=torch.float64)
    es = edges_from_w2c(ext_s)
    z = ((es["R_rel"] - edges["R_rel"]) ** 2).sum(dim=(-2, -1))
    tn_s = F.normalize(es["t_rel"], dim=-1, eps=1e-8)
    tn_t = F.normalize(edges["t_rel"], dim=-1, eps=1e-8)
    hand = z.mean() + (1.0 - (tn_s * tn_t).sum(dim=-1)).mean()
    assert float(loss_rel_pose(ext_s, ext)) == pytest.approx(float(hand), rel=1e-10)


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------
def _mk_step(step, values):
    records = []
    for pair in ("p0", "p1"):
        for k in (0, 1):
            records.append({"pair_id": pair, "mask_id": k,
                            "components": {n: float(v) for n, v in values.items()},
                            "total": 0.0})
    return {"step": step, "records": records}


def test_select_monotonic_improvement_picks_last():
    cfg = ControllerConfig()
    trace = [_mk_step(s, {n: 1.0 - 0.05 * s for n in ("feature", "rkd", "couple", "rot_deg")})
             for s in range(5)]
    res = select(trace, cfg)
    assert res["selected_step"] == 4
    assert not res["fell_back_to_baseline"]
    assert res["improvement"] == pytest.approx(0.2, rel=1e-9)
    assert res["disqualified"] == {}


def test_select_falls_back_to_last_qualified_before_tau_violation():
    cfg = ControllerConfig()
    trace = [_mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(20, {n: 0.95 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(40, {"feature": 0.90, "rkd": 0.90, "couple": 0.90, "rot_deg": 2.0})]
    res = select(trace, cfg)
    assert res["selected_step"] == 20
    assert not res["fell_back_to_baseline"]
    assert 40 in res["disqualified"]
    assert "rot_deg" in res["disqualified"][40]


def test_select_plateau_falls_back_and_ties_pick_earliest():
    cfg = ControllerConfig()
    plateau = [_mk_step(s, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")})
               for s in (0, 10, 20, 30)]
    res = select(plateau, cfg)
    assert res["fell_back_to_baseline"]
    assert res["selected_step"] == 0
    assert res["improvement"] == 0.0
    tie = [_mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")}),
           _mk_step(5, {n: 0.95 for n in ("feature", "rkd", "couple", "rot_deg")}),
           _mk_step(9, {n: 0.95 for n in ("feature", "rkd", "couple", "rot_deg")})]
    assert select(tie, cfg)["selected_step"] == 5


def test_rel_change_and_should_stop():
    assert rel_change(2.0, 1.0, 1e-3) == pytest.approx(1.0)
    assert rel_change(0.0, 0.0, 1e-3) == 0.0
    assert rel_change(1e-4, 0.0, 1e-3) == pytest.approx(0.1)
    cfg = ControllerConfig()
    plateau = [_mk_step(s, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")})
               for s in (0, 10, 20, 30)]
    assert should_stop(plateau, cfg) is True
    assert should_stop(plateau[:3], cfg) is False          # not enough history
    early = [_mk_step(s, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")})
             for s in (0, 10, 20, 25)]
    assert should_stop(early, cfg) is False                # latest step < min_step
    improving = [_mk_step(s, {n: 1.0 - 0.05 * s for n in ("feature", "rkd", "couple", "rot_deg")})
                 for s in range(5)]
    assert should_stop(improving, cfg) is False
    with pytest.raises(ValueError, match="step-0"):
        select(plateau[1:], cfg)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _probe_context(gen, pair_id="pairA"):
    S, P, C, H, W = 4, 16, 8, 6, 6
    teacher = {
        "features": torch.randn(1, S, P, C, generator=gen, dtype=torch.float64),
        "feat_w": torch.rand(1, S, P, generator=gen, dtype=torch.float64),
        "ext_w2c": _rand_ext(S, gen),
        "centers": torch.randn(S, 3, generator=gen, dtype=torch.float64),
        "depth": torch.rand(S, H, W, generator=gen, dtype=torch.float64) + 0.1,
        "valid": torch.rand(S, H, W, generator=gen) < 0.8,
    }
    return {"pair_id": pair_id, "scene": "sceneX", "patch_grid": (4, 4),
            "teacher": teacher, "images_meta": {"tag": "opaque"}}


def test_probe_evaluate_is_deterministic_and_step_aware(tmp_path):
    gen = torch.Generator().manual_seed(7)
    ctx = _probe_context(gen)
    S = 4
    out = {
        "features": torch.randn(1, S, 16, 8, generator=gen, dtype=torch.float64),
        "ext_w2c": _rand_ext(S, gen),
        "centers": torch.randn(S, 3, generator=gen, dtype=torch.float64),
        "depth": torch.rand(S, 6, 6, generator=gen, dtype=torch.float64) + 0.1,
    }
    calls = []

    def forward_fn(images_meta, mask):
        assert images_meta == {"tag": "opaque"}
        assert mask.dtype == torch.bool and mask.shape == (S, 4, 4)
        calls.append(mask.clone())
        return out

    evaluator = ProbeEvaluator([ctx])
    r1 = evaluator.evaluate(0, forward_fn)
    r2 = evaluator.evaluate(0, forward_fn)
    assert r1 == r2
    assert len(r1["records"]) == 2  # one context x two fixed masks
    assert all(torch.equal(a, b) for a, b in zip(calls[:2], calls[2:]))
    r5 = evaluator.evaluate(5, forward_fn)
    assert r5["step"] == 5
    assert r5["records"] == r1["records"]
    rec = r1["records"][0]
    assert set(rec["components"]) == {"feature", "rot_deg", "rkd", "couple"}
    assert rec["total"] == pytest.approx(
        rec["components"]["feature"] + 1.5 * rec["components"]["rkd"]
        + rec["components"]["couple"] + rec["components"]["rot_deg"])
    assert "couple_skipped" not in rec

    # dict features (DA3 multi-tap style): mean over readout dicts
    feats_s = torch.randn(1, S, 16, 8, generator=gen, dtype=torch.float64)
    feats_t = ctx["teacher"]["features"]
    ctx_d = _probe_context(gen, pair_id="pairB")
    ctx_d["teacher"] = dict(ctx_d["teacher"], features={"L0": feats_t, "L1": feats_t * 2})
    out_d = dict(out, features={"L0": feats_s, "L1": feats_s * 2})
    ev_d = ProbeEvaluator([ctx_d])
    r_d = ev_d.evaluate(0, lambda meta, mask: out_d)
    hand = 0.5 * (float(huber_cos_weighted(feats_s, feats_t, ctx_d["teacher"]["feat_w"]))
                  + float(huber_cos_weighted(feats_s * 2, feats_t * 2, ctx_d["teacher"]["feat_w"])))
    assert r_d["records"][0]["components"]["feature"] == pytest.approx(hand, rel=1e-6)

    # model training mode is saved -> eval -> restored
    model = torch.nn.Linear(2, 2)
    model.train()
    modes = []

    def fn_with_model(meta, mask):
        modes.append(model.training)
        return out

    ev2 = ProbeEvaluator([ctx], model=model)
    ev2.evaluate(1, fn_with_model)
    assert modes == [False, False]
    assert model.training is True

    # JSONL roundtrip
    path = str(tmp_path / "probe" / "trace.jsonl")
    append_jsonl(path, r1)
    append_jsonl(path, r5)
    with open(path) as f:
        lines = f.read().strip().split("\n")
    assert [json.loads(l)["step"] for l in lines] == [0, 5]
    assert json.loads(lines[0]) == r1


def test_probe_evaluate_marks_skipped_couple():
    gen = torch.Generator().manual_seed(11)
    ctx = _probe_context(gen)
    ctx["teacher"]["valid"] = torch.zeros_like(ctx["teacher"]["valid"])
    S = 4
    out = {"features": ctx["teacher"]["features"], "ext_w2c": ctx["teacher"]["ext_w2c"],
           "centers": ctx["teacher"]["centers"], "depth": ctx["teacher"]["depth"]}
    evaluator = ProbeEvaluator([ctx])
    res = evaluator.evaluate(0, lambda meta, mask: out)
    for rec in res["records"]:
        assert rec["components"]["couple"] == 0.0
        assert rec["couple_skipped"] == "empty student valid mask"


def test_make_patch_mask_reproducible_and_rng_isolated():
    m1 = make_patch_mask(2, 4, 4, 0.5, 123)
    m2 = make_patch_mask(2, 4, 4, 0.5, 123)
    m3 = make_patch_mask(2, 4, 4, 0.5, 124)
    assert m1.shape == (2, 4, 4) and m1.dtype == torch.bool
    assert torch.equal(m1, m2)
    assert not torch.equal(m1, m3)
    assert make_patch_mask(1, 4, 4, 1.0, 0).all()
    assert not make_patch_mask(1, 4, 4, 0.0, 0).any()
    torch.manual_seed(99)
    before = torch.rand(4)
    torch.manual_seed(99)
    make_patch_mask(2, 4, 4, 0.5, 123)
    assert torch.equal(before, torch.rand(4))


def test_stable_seed_matches_protocol_v1_semantics():
    key = "::".join(["probemask", "sceneX", "pairA", "1"])
    assert stable_seed("probemask", "sceneX", "pairA", 1) == \
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
