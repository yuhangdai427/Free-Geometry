"""Regression tests for the protocol-v2 review fixes.

Covers:
  1. rot_edges_huber branch-precise form: identity -> loss exactly 0 with a
     finite, exactly-zero gradient; 90 deg -> 1.7234; NaN-at-z=0 is gone.
  2. tdir_cos_loss boundaries: all-zero baselines never kept; a single NaN
     drops only that edge; all-NaN -> teacher_unavailable.
  3. couple_robust support region: student statistics on the fixed teacher
     support; student_valid_fraction; student_invalid / teacher_unavailable.
  4. ProbeEvaluator record schema: couple=None (never fake 0.0) +
     couple_status + valid/invalid_reason.
  5. controller.select hardening: the reviewer's end-to-end counterexample
     (features/rot/centers unchanged, depth valid->all-zero must DISQUALIFY,
     never report improvement); schema mismatch rejection; non-finite
     handling; baseline-non-finite exclusion from the participating set.
  6. should_stop best+patience: plateau -> True, new-best reset -> False,
     disqualified entries skipped.
"""
import json
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from free_geometry.tta_v2 import (  # noqa: E402
    ControllerConfig, ProbeEvaluator, couple_robust, edges_from_w2c,
    rot_edges_huber, select, should_stop, tdir_cos_loss,
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
# 1. rot_edges_huber
# ---------------------------------------------------------------------------
def test_rot_identity_backward_finite_zero():
    gen = torch.Generator().manual_seed(0)
    R_t = _rand_rot(gen)
    R_s = R_t.clone().requires_grad_(True)
    out = rot_edges_huber(R_s, R_t)
    assert float(out["loss"].detach()) == 0.0
    assert float(out["per_edge"].sum()) == 0.0
    assert float(out["angle_deg"].sum()) == pytest.approx(0.0, abs=1e-8)
    out["loss"].backward()
    assert torch.isfinite(R_s.grad).all()
    assert float(R_s.grad.abs().sum()) == 0.0


def test_rot_ninety_deg_value():
    gen = torch.Generator().manual_seed(1)
    R_t = _rand_rot(gen)
    R_err = _axis_angle([1.0, 0.0, 0.0], math.radians(90.0))
    out = rot_edges_huber(R_err @ R_t, R_t)
    assert float(out["per_edge"].reshape(-1)[0]) == pytest.approx(1.7234, abs=1e-3)
    assert float(out["angle_deg"].reshape(-1)[0]) == pytest.approx(90.0, abs=1e-9)
    # backward through the large branch stays finite
    Rs = (R_err @ R_t).clone().requires_grad_(True)
    rot_edges_huber(Rs, R_t)["loss"].backward()
    assert torch.isfinite(Rs.grad).all()


def test_rot_knee_continuity():
    # exactly at z = 8 delta^2 both branches give the same value
    delta = math.sin(math.radians(10.0))
    phi_knee = 2.0 * math.asin(delta)  # 20 deg
    gen = torch.Generator().manual_seed(2)
    R_t = _rand_rot(gen)
    R_err = _axis_angle([0.0, 1.0, 0.0], phi_knee)
    out = rot_edges_huber(R_err @ R_t, R_t)
    z = 8.0 * delta * delta
    assert float(out["per_edge"].reshape(-1)[0]) == pytest.approx(z, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. tdir_cos_loss boundaries
# ---------------------------------------------------------------------------
def test_tdir_all_zero_baselines_never_kept():
    gen = torch.Generator().manual_seed(3)
    t_t = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    t_s = t_t + 0.1 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    out = tdir_cos_loss(t_s, t_t, torch.zeros(3, dtype=torch.float64))
    assert out["n_kept"] == 0 and out["n_skipped"] == 3
    assert float(out["loss"]) == 0.0
    assert out["teacher_unavailable"] is False  # finite, just unusable baselines


def test_tdir_single_nan_drops_only_that_edge():
    gen = torch.Generator().manual_seed(4)
    t_t = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    t_s = t_t + 0.1 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    baseline = torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float64)
    out = tdir_cos_loss(t_s, t_t, baseline)
    assert out["n_kept"] == 2 and out["n_skipped"] == 1
    assert out["teacher_unavailable"] is False
    import torch.nn.functional as F
    hand = (1.0 - (F.normalize(t_s[[0, 2]], dim=-1, eps=1e-8)
                   * F.normalize(t_t[[0, 2]], dim=-1, eps=1e-8)).sum(dim=-1)).mean()
    assert float(out["loss"]) == pytest.approx(float(hand), rel=1e-12)


def test_tdir_all_nan_teacher_unavailable():
    gen = torch.Generator().manual_seed(5)
    t_t = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    t_s = t_t + 0.1 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    out = tdir_cos_loss(t_s, t_t, torch.full((3,), float("nan"), dtype=torch.float64))
    assert out["n_kept"] == 0
    assert float(out["loss"]) == 0.0
    assert out["teacher_unavailable"] is True


# ---------------------------------------------------------------------------
# 3. couple_robust support region
# ---------------------------------------------------------------------------
def _couple_setup(n_support=10, gen_seed=6):
    gen = torch.Generator().manual_seed(gen_seed)
    S = 2
    centers_s = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    centers_t = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    valid = torch.zeros(S, 3, 4, dtype=torch.bool)
    valid.view(-1)[:n_support] = True
    depth_t = torch.ones(S, 3, 4, dtype=torch.float64)
    return centers_s, centers_t, depth_t, depth_t.clone(), valid


def test_couple_student_valid_fraction_exact():
    centers_s, centers_t, depth_t, depth_s, valid = _couple_setup(10)
    depth_s.view(-1)[7:] = 0.0  # 7/10 valid -> 0.7
    out = couple_robust(centers_s, centers_t, depth_s, depth_t, valid)
    assert not out["skipped"]
    assert out["student_valid_fraction"] == pytest.approx(0.7)
    assert out["teacher_unavailable"] is False and out["student_invalid"] is False

    depth_s.view(-1)[5:] = 0.0  # exactly 5/10 -> 0.5 passes (threshold is < 0.5)
    out = couple_robust(centers_s, centers_t, depth_s, depth_t, valid)
    assert not out["skipped"] and out["student_valid_fraction"] == pytest.approx(0.5)

    depth_s.view(-1)[4:] = 0.0  # 4/10 -> 0.4 -> student_invalid
    out = couple_robust(centers_s, centers_t, depth_s, depth_t, valid)
    assert out["skipped"] and out["loss"] is None
    assert out["student_invalid"] is True and out["teacher_unavailable"] is False
    assert out["reason"] == "student_invalid"
    assert out["student_valid_fraction"] == pytest.approx(0.4)


def test_couple_teacher_unavailable_and_full_zero_student():
    centers_s, centers_t, depth_t, depth_s, valid = _couple_setup(10)
    out = couple_robust(centers_s, centers_t, torch.zeros_like(depth_s), depth_t, valid)
    assert out["skipped"] and out["student_invalid"] and out["reason"] == "student_invalid"
    assert out["student_valid_fraction"] == 0.0

    out = couple_robust(centers_s, centers_t, depth_s, depth_t,
                        torch.zeros_like(valid))
    assert out["skipped"] and out["teacher_unavailable"]
    assert out["reason"] == "teacher_unavailable"
    assert out["student_valid_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 4/5. probe record schema + selector hardening (end-to-end counterexample)
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
            "teacher": teacher, "images_meta": None}


def _teacher_like_out(ctx):
    t = ctx["teacher"]
    return {"features": t["features"], "ext_w2c": t["ext_w2c"],
            "centers": t["centers"], "depth": t["depth"]}


def test_probe_record_marks_invalid_couple_none_not_zero():
    gen = torch.Generator().manual_seed(7)
    ctx = _probe_context(gen)
    ctx["teacher"]["valid"] = torch.zeros_like(ctx["teacher"]["valid"])
    ev = ProbeEvaluator([ctx])
    res = ev.evaluate(0, lambda meta, mask: _teacher_like_out(ctx))
    for rec in res["records"]:
        assert rec["components"]["couple"] is None
        assert rec["couple_status"] == "teacher_unavailable"
        assert rec["valid"] is False
        assert rec["invalid_reason"] == "couple:teacher_unavailable"


def test_selector_disqualifies_student_depth_collapse_e2e():
    """The reviewer's counterexample: features/rot/centers identical to the
    teacher; only the student depth goes from valid to all-zero. Step 5 must
    be DISQUALIFIED (never scored as an improvement), selection falls back."""
    gen = torch.Generator().manual_seed(8)
    ctx = _probe_context(gen)
    ev = ProbeEvaluator([ctx])
    state = {"collapse": False}

    def fwd(meta, mask):
        out = _teacher_like_out(ctx)
        if state["collapse"]:
            out["depth"] = torch.zeros_like(out["depth"])
        return out

    r0 = ev.evaluate(0, fwd)
    for rec in r0["records"]:
        assert rec["valid"] is True and rec["couple_status"] == "ok"
        assert abs(rec["components"]["feature"]) < 1e-6  # fp32 readout cast noise
        assert rec["components"]["rot_deg"] == 0.0
        assert rec["components"]["rkd"] == 0.0
        assert rec["components"]["couple"] == 0.0  # identical stats -> diff 0
    state["collapse"] = True
    r5 = ev.evaluate(5, fwd)
    for rec in r5["records"]:
        assert rec["components"]["couple"] is None
        assert rec["couple_status"] == "student_invalid"
        assert rec["valid"] is False
        # the other components did NOT move — a false "improvement" could only
        # come from mishandling the invalidated couple
        assert abs(rec["components"]["feature"]) < 1e-6
        assert rec["components"]["rot_deg"] == 0.0
    res = select([r0, r5], ControllerConfig())
    assert 5 in res["disqualified"]
    assert "invalid" in res["disqualified"][5]
    assert res["selected_step"] == 0
    assert res["fell_back_to_baseline"] is True
    assert res["improvement"] == 0.0
    assert res["n_comparable"] == 8  # 2 masks x 4 components, all valid at step 0


def _mk_step(step, values, pairs=("p0", "p1"), drop_component=None, extra_component=None):
    records = []
    for pair in pairs:
        for k in (0, 1):
            comps = {n: float(v) for n, v in values.items()}
            if drop_component:
                comps.pop(drop_component)
            if extra_component:
                comps[extra_component] = 1.0
            records.append({"pair_id": pair, "mask_id": k, "components": comps,
                            "valid": True, "invalid_reason": None, "total": 0.0})
    return {"step": step, "records": records}


def test_selector_rejects_schema_mismatch():
    cfg = ControllerConfig()
    base = _mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")})
    missing = _mk_step(5, {n: 0.9 for n in ("feature", "rkd", "couple", "rot_deg")},
                       drop_component="couple")
    res = select([base, missing], cfg)
    assert 5 in res["disqualified"]
    assert "component set mismatch" in res["disqualified"][5]
    extra = _mk_step(9, {n: 0.9 for n in ("feature", "rkd", "couple", "rot_deg")},
                     extra_component="new_metric")
    res = select([base, extra], cfg)
    assert 9 in res["disqualified"]
    assert "component set mismatch" in res["disqualified"][9]


def test_selector_rejects_nonfinite_candidate_and_excludes_nonfinite_baseline():
    cfg = ControllerConfig()
    base = _mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")})
    bad = _mk_step(5, {"feature": float("inf"), "rkd": 0.9, "couple": 0.9,
                       "rot_deg": 0.9})
    res = select([base, bad], cfg)
    assert 5 in res["disqualified"]
    assert "non-finite" in res["disqualified"][5]

    # non-finite BASELINE: the component is permanently excluded, the candidate
    # is not punished for it, and selection still works on the rest
    bad_base = _mk_step(0, {"feature": float("nan"), "rkd": 1.0, "couple": 1.0,
                            "rot_deg": 1.0})
    ok = _mk_step(5, {"feature": 123.0, "rkd": 0.9, "couple": 0.9, "rot_deg": 0.9})
    res = select([bad_base, ok], cfg)
    assert res["n_comparable"] == 12  # 4 records x 3 finite-baseline components
    assert 5 not in res["disqualified"]
    assert res["selected_step"] == 5  # 0.1 mean improvement on the valid comps


def test_selector_none_component_at_candidate_disqualifies():
    cfg = ControllerConfig()
    base = _mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")})
    recs = _mk_step(5, {"feature": 0.9, "rkd": 0.9, "couple": 0.9, "rot_deg": 0.9})
    for r in recs["records"]:
        r["components"]["couple"] = None  # became invalid at step 5
    res = select([base, recs], cfg)
    assert 5 in res["disqualified"]
    assert "invalid" in res["disqualified"][5]
    assert res["selected_step"] == 0


# ---------------------------------------------------------------------------
# 6. should_stop best+patience
# ---------------------------------------------------------------------------
def test_should_stop_plateau_after_min_step():
    cfg = ControllerConfig(min_step=30, patience=2, min_delta=0.005)
    trace = [_mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(30, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(40, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(50, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")})]
    assert should_stop(trace, cfg) is True
    assert should_stop(trace[:2], cfg) is False  # only one post-min_step check
    early = [_mk_step(0, {n: 1.0 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(10, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(20, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")}),
             _mk_step(25, {n: 0.999 for n in ("feature", "rkd", "couple", "rot_deg")})]
    assert should_stop(early, cfg) is False  # latest step < min_step


def test_should_stop_new_best_resets_stale_counter():
    cfg = ControllerConfig(min_step=30, patience=2, min_delta=0.005)
    names = ("feature", "rkd", "couple", "rot_deg")
    trace = [_mk_step(0, {n: 1.0 for n in names}),
             _mk_step(30, {n: 0.99 for n in names}),   # imp 0.01 -> best
             _mk_step(40, {n: 0.98 for n in names}),   # imp 0.02 -> new best, reset
             _mk_step(50, {n: 0.979 for n in names})]  # imp 0.021 < 0.02+0.005 -> stale 1
    assert should_stop(trace, cfg) is False
    trace.append(_mk_step(60, {n: 0.978 for n in names}))  # stale 2 -> stop
    assert should_stop(trace, cfg) is True


def test_should_stop_skips_disqualified_entries():
    cfg = ControllerConfig(min_step=30, patience=2, min_delta=0.005)
    names = ("feature", "rkd", "couple", "rot_deg")
    plateau = {n: 0.999 for n in names}
    trace = [_mk_step(0, {n: 1.0 for n in names}),
             _mk_step(30, plateau),
             _mk_step(40, {"feature": 0.9, "rkd": 0.9, "couple": 0.9, "rot_deg": 5.0}),
             _mk_step(50, plateau)]
    assert should_stop(trace, cfg) is True  # step40 disqualified -> not counted


# ---------------------------------------------------------------------------
# 7. abs_floor_rel: per-component absolute noise floor (real-trace regression)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_vggt_trace(path):
    per_step = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            per_step.setdefault(int(r["step"]), {"step": int(r["step"]), "records": []})
            per_step[int(r["step"])]["records"].append({
                "pair_id": r["pair_id"], "mask_id": int(r["mask_id"]),
                "components": r["components"], "total": r.get("total")})
    return [per_step[k] for k in sorted(per_step)]


def test_abs_floor_rel_electro_real_trace():
    """workspace/protocol_v2/vggt_eth3d/probe_trace/electro.jsonl: the couple
    record probe0/mask0 has step0=0.00083 while the component's typical scale
    is ~0.054. Pre-fix (abs_floor_rel=0, legacy tau rule) the +0.001 noise
    wiggle read as +108% rel_change and vetoed EVERY candidate -> whole-run
    false fallback. With the abs_floor_rel noise floor the wiggle is noise;
    under the redesigned selection rule (catastrophic safety net, default
    0.5) no candidate is vetoed at all and the best-improvement step wins.
    """
    path = os.path.join(_REPO_ROOT, "workspace/protocol_v2/vggt_eth3d/probe_trace/electro.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"real trace not available: {path}")
    trace = _load_vggt_trace(path)

    old = select(trace, ControllerConfig(abs_floor_rel=0.0, catastrophic_rel=None))
    assert old["fell_back_to_baseline"] is True           # the reported pathology
    assert old["selected_step"] == 0
    assert 10 in old["disqualified"] and "couple" in old["disqualified"][10]

    new = select(trace, ControllerConfig())
    assert new["disqualified"] == {}                      # safety net: no vetoes
    assert new["fell_back_to_baseline"] is False
    assert new["selected_step"] == 100
    assert new["improvement"] > 0


def test_abs_floor_rel_mechanics_synthetic():
    """couple records [0.05, 0.05, 8e-4, 8e-4] -> typical_k = 0.0254,
    floor = 0.2*0.0254 = 0.00508 (abs), denominator = floor/tau = 0.1016.
    A +0.001 wiggle passes; a +0.02 jump (2x floor, rel_change 0.197) passes
    the 0.5 safety net but is caught by the legacy tau rule; a +0.2 jump
    (rel_change ~2) is catastrophic."""
    base = _mk_step(0, {"feature": 1.0, "rkd": 1.0, "couple": 0.05, "rot_deg": 1.0})
    # probe1's couple baseline is near-zero while the component scale is 0.0254
    for rec in base["records"]:
        if rec["pair_id"] == "p1":
            rec["components"]["couple"] = 8e-4
    noise = _mk_step(5, {"feature": 1.0, "rkd": 1.0, "couple": 0.05, "rot_deg": 1.0})
    for rec in noise["records"]:
        if rec["pair_id"] == "p1":
            rec["components"]["couple"] = 8e-4 + 0.001   # noise-level wiggle
    old = select([base, noise], ControllerConfig(abs_floor_rel=0.0, catastrophic_rel=None))
    assert 5 in old["disqualified"] and "couple" in old["disqualified"][5]
    new = select([base, noise], ControllerConfig())
    assert 5 not in new["disqualified"]
    assert new["selected_step"] == 0                     # no improvement either

    jump = _mk_step(7, {"feature": 1.0, "rkd": 1.0, "couple": 0.05, "rot_deg": 1.0})
    for rec in jump["records"]:
        if rec["pair_id"] == "p1":
            rec["components"]["couple"] = 8e-4 + 0.02    # 2x floor: real but mild
    res = select([base, jump], ControllerConfig())
    assert 7 not in res["disqualified"]                  # below catastrophic 0.5
    res_legacy = select([base, jump], ControllerConfig(catastrophic_rel=None))
    assert 7 in res_legacy["disqualified"]               # legacy tau still vetoes

    cata = _mk_step(9, {"feature": 1.0, "rkd": 1.0, "couple": 0.05, "rot_deg": 1.0})
    for rec in cata["records"]:
        if rec["pair_id"] == "p1":
            rec["components"]["couple"] = 8e-4 + 0.2     # ~2x denominator: catastrophic
    res = select([base, cata], ControllerConfig())
    assert 9 in res["disqualified"] and "couple" in res["disqualified"][9]

    # the floor is relative to the component's OWN scale and the WIDER
    # envelope wins: a +4%-of-typical wiggle sits inside the absolute floor;
    # a +24% jump is still far below the 0.5 safety net.
    base2 = _mk_step(0, {"feature": 1.0, "rkd": 1.0, "couple": 0.05, "rot_deg": 1.0})
    mild = _mk_step(11, {"feature": 1.0, "rkd": 1.0, "couple": 0.052, "rot_deg": 1.0})
    res = select([base2, mild], ControllerConfig())
    assert 11 not in res["disqualified"]
    big = _mk_step(13, {"feature": 1.0, "rkd": 1.0, "couple": 0.062, "rot_deg": 1.0})
    res = select([base2, big], ControllerConfig())
    assert 13 not in res["disqualified"]


def test_catastrophic_safety_net_semantics():
    """The terrains motivation: a candidate with a mild (+0.25 rel_change)
    degradation on ONE (record, component) but the best mean improvement must
    WIN under the new rule, while the legacy rule vetoes it; a candidate with
    a genuine catastrophic jump (+0.75 rel_change) is vetoed by the safety
    net; fallback happens only when the best non-vetoed improvement <
    min_delta."""
    names = ("feature", "rkd", "couple", "rot_deg")
    base = _mk_step(0, {n: 1.0 for n in names})
    # candidate A: best mean improvement, one mild degradation (rc +0.25)
    a = _mk_step(20, {n: 0.9 for n in names})
    for rec in a["records"]:
        if rec["pair_id"] == "p0" and rec["mask_id"] == 0:
            rec["components"]["rot_deg"] = 2.0           # rc = 1.0/4.0 = +0.25
    # candidate B: tiny improvement below min_delta
    b = _mk_step(40, {n: 0.99 for n in names})
    # candidate C: catastrophic component (rc +0.75)
    c = _mk_step(60, {n: 0.9 for n in names})
    for rec in c["records"]:
        if rec["pair_id"] == "p0" and rec["mask_id"] == 0:
            rec["components"]["rot_deg"] = 4.0           # rc = 3.0/4.0 = +0.75

    new = select([base, a, b, c], ControllerConfig())
    assert new["selected_step"] == 20                    # A wins despite the mild hit
    assert 60 in new["disqualified"] and "rot_deg" in new["disqualified"][60]
    assert 40 not in new["disqualified"]                 # B merely loses on rank

    legacy = select([base, a, b, c], ControllerConfig(catastrophic_rel=None))
    assert 20 in legacy["disqualified"]                  # tau rule vetoes A
    assert legacy["selected_step"] == 0                  # B < min_delta -> fallback
    assert legacy["fell_back_to_baseline"] is True

    # fallback only when the best non-vetoed improvement is < min_delta:
    # B alone -> fallback under the new rule too
    both = select([base, b], ControllerConfig())
    assert both["selected_step"] == 0 and both["fell_back_to_baseline"] is True
    # ... and a clearly-improving candidate still wins
    strong = _mk_step(80, {n: 0.9 for n in names})
    res = select([base, strong], ControllerConfig())
    assert res["selected_step"] == 80 and res["improvement"] > 0.005

