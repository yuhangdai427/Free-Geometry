"""Phase-C unit tests: teacher-target reliability weights (tta_v2.reliability)
and the pure logic of the AB-context manifest builder."""
import math
import os
import random
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from free_geometry.tta_v2.reliability import (  # noqa: E402
    camera_task_weight, feature_reliability, rotation_reliability,
)
import build_ab_context_manifests as bab  # noqa: E402,E501  (pulls protocol_v1, ~5s)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rand_rot(gen):
    q = torch.randn(4, generator=gen, dtype=torch.float64)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _axis_angle(axis, theta_deg):
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    x, y, z = axis.tolist()
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64)
    th = math.radians(theta_deg)
    return I + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _rand_ext(S, gen):
    R = torch.stack([_rand_rot(gen) for _ in range(S)])
    t = torch.randn(S, 3, generator=gen, dtype=torch.float64)
    return torch.cat([R, t[..., None]], dim=-1)  # [S,3,4] w2c


# ---------------------------------------------------------------------------
# feature_reliability
# ---------------------------------------------------------------------------
def test_feature_reliability_identical_inputs_q_one():
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(2, 32, 8, generator=gen, dtype=torch.float64)
    q, stats = feature_reliability(z, z.clone())
    assert q.shape == (2, 32)
    torch.testing.assert_close(q, torch.ones_like(q))
    assert stats["tau"] == 0.0            # median of all-zero deviations
    assert stats["q_mean"] == 1.0
    # batch dim is squeezed
    q4, _ = feature_reliability(z[None], z[None].clone())
    torch.testing.assert_close(q4, q)


def test_feature_reliability_monotone_under_controlled_perturbation():
    gen = torch.Generator().manual_seed(1)
    zA = torch.randn(4, 64, 8, generator=gen, dtype=torch.float64)
    eps = torch.randn(4, 64, 8, generator=gen, dtype=torch.float64)
    # exact per-row orthogonality: u(sigma) = 1 - |z|/sqrt(|z|^2 + sigma^2|eps|^2)
    # is then strictly increasing in sigma, elementwise
    eps = eps - (eps * zA).sum(-1, keepdim=True) / (zA * zA).sum(-1, keepdim=True) * zA
    sigmas = [0.25, 0.5, 1.0, 2.0]
    fA = torch.cat([zA for _ in sigmas], dim=0)          # [16,64,8]: one shared tau
    fB = torch.cat([zA + s * eps for s in sigmas], dim=0)
    q, stats = feature_reliability(fA, fB)
    u = 1.0 - F.cosine_similarity(fA.float(), fB.float(), dim=-1)  # module upcasts to fp32
    assert stats["tau_used"] == pytest.approx(float(u.median()), rel=1e-5)
    qs = [q[i * 4:(i + 1) * 4] for i in range(len(sigmas))]
    for s_prev, s_next in zip(qs, qs[1:]):
        assert (s_next < s_prev).all()   # strictly decreasing, elementwise
    # sanity: at sigma=0.25 nothing is fully down-weighted, at sigma=2 much is
    assert float(qs[0].mean()) > 0.9
    assert float(qs[-1].mean()) < 0.5


def test_feature_reliability_tau_is_recorded_median():
    gen = torch.Generator().manual_seed(2)
    zA = torch.randn(3, 50, 6, generator=gen, dtype=torch.float64)
    zB = zA + 0.3 * torch.randn(3, 50, 6, generator=gen, dtype=torch.float64)
    q, stats = feature_reliability(zA, zB)
    u = 1.0 - F.cosine_similarity(zA.float(), zB.float(), dim=-1)  # module upcasts to fp32
    assert stats["tau"] == pytest.approx(float(u.median()), rel=1e-5)
    hand = 1.0 / (1.0 + (u / u.median()) ** 2)
    torch.testing.assert_close(q, hand, rtol=1e-5, atol=1e-8)


def test_feature_reliability_shape_mismatch_raises():
    gen = torch.Generator().manual_seed(3)
    a = torch.randn(2, 8, 4, generator=gen, dtype=torch.float64)
    b = torch.randn(3, 8, 4, generator=gen, dtype=torch.float64)
    with pytest.raises(ValueError):
        feature_reliability(a, b)


# ---------------------------------------------------------------------------
# rotation_reliability / camera_task_weight
# ---------------------------------------------------------------------------
def test_rotation_reliability_identical_ext_q_one():
    gen = torch.Generator().manual_seed(4)
    ext = _rand_ext(5, gen)
    q, stats = rotation_reliability(ext, ext.clone())
    E = 5 * 4 // 2
    assert q.shape == (E,)
    torch.testing.assert_close(q, torch.ones(E, dtype=torch.float64))
    assert stats["tau"] == 0.0
    assert stats["edge_index"].shape == (E, 2)


def _mixed_ext_with_one_90deg_view(S, gen):
    """ext_B = ext_A except the LAST view left-multiplied by a 90-degree
    rotation. Edges among views 0..S-2 agree exactly; the S-1 edges touching
    the last view differ by exactly 90 deg."""
    ext_A = _rand_ext(S, gen)
    R90 = _axis_angle([1.0, 0.3, -0.2], 90.0)
    R_B = ext_A[..., :3, :3].clone()
    R_B[-1] = R90 @ R_B[-1]
    ext_B = torch.cat([R_B, ext_A[..., :3, 3:]], dim=-1)
    return ext_A, ext_B


def test_rotation_reliability_90deg_outlier_edges_q_near_zero():
    gen = torch.Generator().manual_seed(5)
    S = 8  # 28 edges: 21 exactly consistent, 7 at 90 deg
    ext_A, ext_B = _mixed_ext_with_one_90deg_view(S, gen)
    q, stats = rotation_reliability(ext_A, ext_B)
    E = S * (S - 1) // 2
    outlier = (stats["edge_index"] == S - 1).any(dim=-1)
    assert int(outlier.sum()) == S - 1
    torch.testing.assert_close(stats["angle_deg"][~outlier],
                               torch.zeros(E - (S - 1), dtype=torch.float64))
    torch.testing.assert_close(stats["angle_deg"][outlier],
                               torch.full((S - 1,), 90.0, dtype=torch.float64),
                               rtol=1e-9, atol=1e-8)
    # exactly-consistent edges: u == 0 -> q == 1
    torch.testing.assert_close(q[~outlier], torch.ones(E - (S - 1), dtype=torch.float64))
    # median of [0 x 21, pi/2 x 7] is 0 -> mean fallback; 90 deg edges -> q = 1/(1+16)
    assert stats["tau"] == 0.0
    tau_used = (S - 1) * (math.pi / 2) / E
    assert stats["tau_used"] == pytest.approx(tau_used, rel=1e-12)
    hand = 1.0 / (1.0 + ((math.pi / 2) / tau_used) ** 2)
    torch.testing.assert_close(q[outlier], torch.full((S - 1,), hand, dtype=torch.float64))
    assert hand == pytest.approx(1.0 / 17.0)
    assert float(q[outlier].mean()) < 0.1   # extreme disagreement ~ 0
    # baseline stats: consistent edges agree in direction AND length
    torch.testing.assert_close(stats["tdir_inconsistency"][~outlier],
                               torch.zeros(E - (S - 1), dtype=torch.float64))
    torch.testing.assert_close(stats["baseline_log_ratio"][~outlier],
                               torch.zeros(E - (S - 1), dtype=torch.float64))
    assert bool((stats["tdir_inconsistency"][outlier] > 0).any())
    assert stats["baseline_t"].shape == (E,)
    # batch dim accepted
    q_b, stats_b = rotation_reliability(ext_A[None], ext_B[None])
    torch.testing.assert_close(q_b, q)


def test_camera_task_weight_is_mean_and_rejects_empty():
    q = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    assert camera_task_weight(q) == pytest.approx(0.5833333333)
    with pytest.raises(ValueError):
        camera_task_weight(torch.empty(0))


# ---------------------------------------------------------------------------
# manifest builder: file validation
# ---------------------------------------------------------------------------
def test_check_image_file_reasons(tmp_path):
    good = tmp_path / "a.png"
    good.write_bytes(b"\x89PNG\r\n\x1a\nnot-really-an-image-but-nonempty")
    assert bab.check_image_file(str(good)) is None
    empty = tmp_path / "b.png"
    empty.write_bytes(b"")
    assert bab.check_image_file(str(empty)) == "empty"
    note = tmp_path / "c.txt"
    note.write_bytes(b"hello")
    assert bab.check_image_file(str(note)) == "bad_extension:.txt"
    assert bab.check_image_file(str(tmp_path / "missing.jpg")) == "missing"


def _write_synthetic_scene(tmp_path, n=10, bad_index=5):
    import cv2
    import numpy as np
    rng = np.random.RandomState(7)
    files = []
    for i in range(n):
        p = str(tmp_path / f"img_{i:03d}.png")
        if i == bad_index:
            open(p, "wb").close()
        else:
            cv2.imwrite(p, rng.randint(0, 255, (48, 64, 3)).astype("uint8"))
        files.append(p)
    return files


def test_build_ab_scene_protocol_marks_degraded_and_excludes_bad_files(tmp_path):
    files = _write_synthetic_scene(tmp_path, n=10, bad_index=5)
    proto = bab.build_ab_scene_protocol(files, "syn", "synthds", n_train=10, n_probe=2)
    assert proto["protocol_version"] == "v2ab"
    assert proto["degraded"] is True                 # bad file + N_valid=9 < 12
    assert len(proto["bad_files"]) == 1
    assert proto["bad_files"][0]["index"] == 5
    assert "empty" in proto["bad_files"][0]["reason"]
    assert proto["N"] == 10 and proto["N_valid"] == 9
    assert len(proto["train_pairs"]) == 10 and len(proto["probe_pairs"]) == 2
    assert proto["ab_overlap_mean"] is not None
    assert max(i for p_ in proto["train_pairs"] for i in p_["teacher_frames"]) < 10
    for rec in proto["train_pairs"]:
        for key in ("teacher_frames", "teacher_frames_B", "student_frames",
                    "extras_A", "extras_B"):
            assert 5 not in rec[key]
            assert all(0 <= i < 10 for i in rec[key])
        assert rec["teacher_frames"][::2] == rec["student_frames"]      # A slots
        assert rec["teacher_frames_B"][::2] == rec["student_frames"]    # B slots
    assert proto["eval_frames"] == list(range(10))
    assert proto["image_files"] == files
    # every recorded reason mentions the bad file / short N
    assert any("bad file" in r for r in proto["degrade_reasons"])


# ---------------------------------------------------------------------------
# manifest builder: A/B selection logic (synthetic frac, no real SIFT)
# ---------------------------------------------------------------------------
def test_rank_candidates_prefers_in_range_near_midpoint():
    table = {10: 0.5, 11: 0.3, 12: 0.05, 13: 0.9}
    frac = lambda f, s: table[f]          # noqa: E731
    ranked = bab.rank_candidates([10, 11, 12, 13], [0, 1], frac)
    assert [f for f, _ in ranked] == [11, 10, 12, 13]   # 0.3 = midpoint, then in-range, then out


def test_split_ab_disjoint_then_overlap_fallback():
    scored = [(i, 0.3) for i in range(9)]
    A, B, sA, sB, ncand = bab.split_ab(scored, n_extras=4)
    assert ncand == 9 and len(A) == len(B) == 4
    assert set(A).isdisjoint(B)
    assert sA == [0.3] * 4 and sB == [0.3] * 4
    for n_cand in (6, 3):
        scored = [(100 + i, 0.3) for i in range(n_cand)]
        A, B, _, _, ncand = bab.split_ab(scored, n_extras=4)
        expect = min(4, max(0, 8 - n_cand), n_cand)  # shortage -> reuse A's best
        assert ncand == n_cand
        assert len(set(A) & set(B)) == expect


def test_build_ab_train_pair_keeps_slot_convention_and_disjointness():
    rng = random.Random(0)
    frac = lambda f, s: 0.3               # noqa: E731  constant in-range overlap
    rec = bab.build_ab_train_pair(60, dense=True, frac=frac, rng=rng)
    assert len(rec["teacher_frames"]) == 8 and len(rec["teacher_frames_B"]) == 8
    assert rec["teacher_frames"][::2] == rec["student_frames"]     # shared even slots
    assert rec["teacher_frames_B"][::2] == rec["student_frames"]
    assert rec["teacher_frames"][1::2] == sorted(rec["extras_A"])
    assert rec["teacher_frames_B"][1::2] == sorted(rec["extras_B"])
    assert rec["ab_overlap"] == 0 and rec["n_candidates"] == 36    # window(40) - 4
    # dense window is contiguous-ish: all frames inside the drawn window
    assert max(rec["teacher_frames"]) - min(rec["teacher_frames"]) < 60
    # random branch: 12-frame window -> 8 candidates -> still disjoint
    rec2 = bab.build_ab_train_pair(60, dense=False, frac=frac, rng=random.Random(1))
    assert rec2["ab_overlap"] == 0 and rec2["n_candidates"] == 8
    assert len(set(rec2["teacher_frames"]) | set(rec2["teacher_frames_B"])) == 12


def test_sample_ab_tasks_dedup_probe_disjoint_reproducible():
    frac = lambda f, s: 0.2               # noqa: E731
    train, probe, meta = bab.sample_ab_tasks(30, dense=False, frac=frac,
                                             dataset="synth", scene="s1",
                                             n_train=10, n_probe=2)
    assert len(train) == 10 and len(probe) == 2
    assert meta["train_dedup_short"] is False and meta["probe_dedup_short"] is False
    shared_keys = [tuple(p["student_frames"]) for p in train]
    union_keys = [tuple(sorted(set(p["teacher_frames"]) | set(p["teacher_frames_B"])))
                  for p in train]
    assert len(set(shared_keys)) == 10                     # shared-group dedup
    assert len(set(union_keys)) == 10
    probe_shared = {tuple(p["student_frames"]) for p in probe}
    probe_teacher = {tuple(p["teacher_frames"]) for p in probe}
    assert probe_shared.isdisjoint(shared_keys)            # probe keys disjoint
    assert probe_teacher.isdisjoint(union_keys)
    assert all(len(p["teacher_frames"]) == 8 for p in probe)
    assert max(i for p_ in train for i in p_["teacher_frames"]) < 30
    # deterministic per (dataset, scene)
    train2, probe2, _ = bab.sample_ab_tasks(30, dense=False, frac=frac,
                                            dataset="synth", scene="s1",
                                            n_train=10, n_probe=2)
    assert train == train2 and probe == probe2


def test_small_pool_random_branch_yields_distinct_shared_groups():
    """N <= shared + 2*extras: the window is the whole pool; a sorted stride
    pick would be deterministic. The shared group must still be drawn at
    random so dedup has something to discriminate."""
    frac = lambda f, s: 0.2               # noqa: E731
    for n, tN in ((13, 8), (26, 16)):     # hiroom-tiny and eth3d-office regimes
        train, probe, meta = bab.sample_ab_tasks(n, dense=False, frac=frac,
                                                 dataset="synth", scene=f"small{n}",
                                                 n_train=10, n_probe=2, teacher_N=tN)
        shared_keys = {tuple(p["student_frames"]) for p in train}
        assert len(shared_keys) > 2, f"N={n}: only {len(shared_keys)} distinct tasks"
        assert all(p["ab_overlap"] == max(0, 2 * (tN - 4) - (n - 4)) for p in train)
        assert all(len(p["teacher_frames"]) == tN == len(p["teacher_frames_B"])
                   for p in train)


def test_full_pool_union_dedup_relaxed_for_16frame_small_scenes():
    """N=26, teacher_N=16: every A-union-B spans all 26 frames, so union-key
    dedup is structurally impossible; shared-group dedup must still give
    10 distinct tasks and the scene must NOT be marked dedup-exhausted."""
    frac = lambda f, s: 0.2               # noqa: E731
    train, probe, meta = bab.sample_ab_tasks(26, dense=False, frac=frac,
                                             dataset="synth", scene="office26",
                                             n_train=10, n_probe=2, teacher_N=16)
    assert meta["train_dedup_short"] is False and meta["probe_dedup_short"] is False
    assert len({tuple(p["student_frames"]) for p in train}) == 10
    assert all(p["ab_overlap"] == 2 for p in train)        # 24 needed, 22 pool
    probe_teachers = {tuple(sorted(p["teacher_frames"])) for p in probe}
    train_ctx = {tuple(sorted(p["teacher_frames"])) for p in train} \
        | {tuple(sorted(p["teacher_frames_B"])) for p in train}
    assert probe_teachers.isdisjoint(train_ctx)            # probe vs A/B lists
