#!/usr/bin/env python3
"""Protocol-v2 phase-C AB-context manifest builder (pilot: eth3d/7scenes/scannetpp).

Each scene gets 10 train tasks; a task is ONE shared frame group S (4 frames,
student views at teacher slots [0,2,4,6]) seen under TWO different 8-frame
teacher contexts A and B — 4 distinct extras each, SIFT-ranked against S and
disjoint-preferred. The two frozen teacher caches of S on every task give a
per-patch / per-edge cross-check of the teacher target (consumed offline via
src/free_geometry/tta_v2/reliability.py). Probe pairs stay single-context
(protocol-v1 style, reusing protocol_v1.build_pair directly).

Frame/extras selection reuses protocol_v1 primitives (SiftCache/compute_tau/
assemble_teacher_list/stable_seed) so the SIFT semantics cannot diverge from
the running protocol; only the A/B split is new.

Output: <out>/<ds>/<scene>.json, schema-compatible with the existing protocol
JSONs (train pairs gain the optional teacher_frames_B / extras_A / extras_B /
ab_overlap / n_candidates keys; probe pairs are unchanged v1-style pairs at the
scene's teacher_N). Top level gains protocol_version="v2ab", degraded,
degrade_reasons, bad_files and AB statistics (ab_overlap_mean / ab_overlap_max
/ n_candidates_mean). teacher_N convention: --teacher_n 8 (default; DA3 v2ab,
8:4 everywhere) or auto16 (old VGGT manifest rule: 16:4 for N>=16, 8:4 below).
A/B extras are disjoint-preferred; candidate shortage (pool smaller than
shared + 2*extras) yields a recorded ab_overlap > 0 without degrading — the
VGGT 16:4 convention explicitly accepts this (e.g. N=26 -> 2 overlapping
frames); degradation tracks only bad files, N<12 pools and dedup exhaustion.

Data-loader validation: every image file is checked for existence, non-zero
size and a known image extension (header-level check; deliberately no full
cv2.imread decode — on large scans decode would dominate wall time). Files
failing the check are listed in bad_files, excluded from the sampling pool,
and mark the scene degraded.
"""
import argparse
import json
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))

import common as fg_common  # noqa: E402
from depth_anything_3.test_time_adaption import protocol_v1 as P  # noqa: E402

N_SHARED = 4
N_EXTRAS = 4  # extras per context when teacher_N = 8 (DA3 v2ab convention)
TEACHER_N = N_SHARED + N_EXTRAS  # 8-frame teacher contexts (4 shared + 4 extras)
OVERLAP_LO, OVERLAP_HI = P.OVERLAP_LO, P.OVERLAP_HI
MIN_SCENE_N = 8   # below this the protocol drops the scene (protocol_v1 parity)
DEGRADED_N = 12   # below this A/B extras cannot be fully disjoint -> degraded
WINDOW = 40       # dense-branch candidate-window cap (bounds the SIFT ranking cost)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}


def teacher_n_for(N: int, mode: str = "8") -> int:
    """Per-scene teacher_N convention. "8": fixed 8:4 everywhere (DA3 v2ab).
    "auto16": the old VGGT manifest rule — 16:4 for N>=16, 8:4 below (matching
    build_final_manifest's `teacher_N = 16 if N >= 16 else 8`)."""
    if mode == "auto16":
        return 16 if N >= 16 else 8
    return 8


# ---------------------------------------------------------------------------
# Data-loader validation (pure; unit-tested)
# ---------------------------------------------------------------------------
def check_image_file(path: str) -> Optional[str]:
    """Header-level data-loader check. Returns None if the file looks loadable
    (exists, non-empty, known image extension), else a short reason string.
    No pixel decode — file size + extension is the fast sanctioned check."""
    if not os.path.exists(path):
        return "missing"
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMG_EXTS:
        return f"bad_extension:{ext or '<none>'}"
    try:
        if os.path.getsize(path) <= 0:
            return "empty"
    except OSError as e:
        return f"stat_error:{e}"
    return None


def collect_valid_files(image_files: Sequence[str]) -> Tuple[List[int], List[Dict]]:
    """Split image_files into (valid indices, bad file records {index, path, reason})."""
    valid_idx, bad = [], []
    for i, p in enumerate(image_files):
        reason = check_image_file(p)
        if reason is None:
            valid_idx.append(i)
        else:
            bad.append({"index": i, "path": p, "reason": reason})
    return valid_idx, bad


# ---------------------------------------------------------------------------
# A/B extras selection (pure; unit-tested)
# ---------------------------------------------------------------------------
def rank_candidates(candidates: Sequence[int], shared: Sequence[int], frac: Callable[[int, int], float],
                    overlap_lo: float = OVERLAP_LO, overlap_hi: float = OVERLAP_HI
                    ) -> List[Tuple[int, float]]:
    """Candidates best-first by mean SIFT match fraction vs the shared frames.

    In-range scores [overlap_lo, overlap_hi] come first (closer to the range
    midpoint = better), then out-of-range scores by distance to the nearest
    bound. Deterministic tie-break on frame index.
    """
    mid = 0.5 * (overlap_lo + overlap_hi)
    scored = []
    for f in candidates:
        c = float(np.mean([frac(f, s) for s in shared]))
        if overlap_lo <= c <= overlap_hi:
            key = (0, abs(c - mid))
        else:
            key = (1, min(abs(c - overlap_lo), abs(c - overlap_hi)))
        scored.append((key, int(f), c))
    scored.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    return [(f, c) for _, f, c in scored]


def split_ab(scored: Sequence[Tuple[int, float]], n_extras: int = N_EXTRAS):
    """Split the ranked candidate list into extras A / extras B.

    A takes the best n_extras; B takes the next n_extras when enough candidates
    remain (disjoint by construction), otherwise B fills up disjoint-first and
    reuses A's best (candidate shortage -> overlap, recorded by the caller).
    Returns (A, B, scores_A, scores_B, n_candidates).
    """
    frame2score = {f: c for f, c in scored}
    frames = [f for f, _ in scored]
    A = frames[:n_extras]
    rest = frames[n_extras:]
    B = rest[:n_extras] if len(rest) >= n_extras else (rest + A)[:n_extras]
    return list(A), list(B), [frame2score[f] for f in A], [frame2score[f] for f in B], len(frames)


def build_ab_train_pair(N: int, dense: bool, frac: Callable[[int, int], float], rng: random.Random,
                        n_shared: int = N_SHARED, teacher_N: int = TEACHER_N,
                        overlap_lo: float = OVERLAP_LO, overlap_hi: float = OVERLAP_HI,
                        window: int = WINDOW) -> Dict:
    """One AB train task over a frame pool of size N (pool coordinates).

    Shared-group sampling keeps protocol_v1's tau dispatch: dense (tau>0.55)
    equidistant window (>= n_shared + 2*n_extras frames when the pool allows);
    random otherwise (window of n_shared + 2*n_extras random frames). Extras
    A/B (n_extras = teacher_N - n_shared per context) are always SIFT-ranked
    against the shared group (rank_candidates) and disjoint-preferred
    (split_ab); candidate shortage -> B reuses A's best, recorded in ab_overlap
    (NOT degraded: acceptable, per the 16:4 VGGT convention).

    Returns {teacher_frames (A assembly), teacher_frames_B, student_frames,
    extras_A, extras_B, ab_overlap, n_candidates, overlap_mean_A/B, ...} with
    the shared frames at even slots [0,2,4,6] of BOTH teacher lists
    (protocol_v1.assemble_teacher_list; extras fill the remaining slots).
    """
    n_extras = teacher_N - n_shared
    if N < n_shared + 1:
        raise ValueError(f"pool N={N} too small for n_shared={n_shared}")
    if dense:
        M = min(N, max(window, n_shared + 2 * n_extras))
        step = N / M
        off = rng.uniform(0, step)
        sup = sorted(set(int(off + i * step) for i in range(M)))[:M]
        stride = max(1, len(sup) // n_shared)
        shared = sorted(sup[::stride][:n_shared])
    else:
        need = n_shared + 2 * n_extras
        if need < N:
            sup = sorted(rng.sample(range(N), need))
            stride = max(1, len(sup) // n_shared)
            shared = sorted(sup[::stride][:n_shared])
        else:
            # pool smaller than one full A+B window: the window IS the whole
            # pool, so a sorted stride pick would be deterministic ([0, s, 2s,
            # ...] every task); draw the shared group at random instead so
            # tasks stay distinct.
            shared = sorted(rng.sample(range(N), n_shared))
            sup = list(range(N))
    candidates = [f for f in sup if f not in set(shared)]
    scored = rank_candidates(candidates, shared, frac, overlap_lo, overlap_hi)
    A, B, sA, sB, ncand = split_ab(scored, n_extras)
    teacher_A = P.assemble_teacher_list(shared, sorted(A))
    teacher_B = P.assemble_teacher_list(shared, sorted(B))
    return {"teacher_frames": [int(i) for i in teacher_A],
            "teacher_frames_B": [int(i) for i in teacher_B],
            "student_frames": list(shared),
            "teacher_N": len(teacher_A), "n_shared": n_shared, "kind": "fixed",
            "extras_A": sorted(A), "extras_B": sorted(B),
            "ab_overlap": len(set(A) & set(B)), "n_candidates": ncand,
            "overlap_mean_A": float(np.mean(sA)) if sA else float("nan"),
            "overlap_mean_B": float(np.mean(sB)) if sB else float("nan")}


class _FracAdapter:
    """Duck-typed SiftCache stand-in exposing .frac(i,j) over a pool-coordinate
    callable (protocol_v1.build_pair only calls sc.frac)."""

    def __init__(self, frac: Callable[[int, int], float]):
        self._frac = frac

    def frac(self, i: int, j: int) -> float:
        return self._frac(i, j)


def sample_ab_tasks(N: int, dense: bool, frac: Callable[[int, int], float], dataset: str, scene: str,
                    n_train: int = 10, n_probe: int = 2, window: int = WINDOW,
                    teacher_N: int = TEACHER_N
                    ) -> Tuple[List[Dict], List[Dict], Dict]:
    """10 AB train tasks (shared-group dedup) + 2 single-context probe pairs
    (protocol_v1.build_pair at the scene's teacher_N, key-disjoint from train).
    Pool coordinates; the caller maps to global image_files indices.
    Returns (train, probe, meta)."""
    r = random.Random(P.stable_seed("ab_train", dataset, scene))
    forb_s: set = set()
    forb_union: set = set()
    train: List[Dict] = []
    tries = 0
    while len(train) < n_train and tries < 10000:
        tries += 1
        rec = build_ab_train_pair(N, dense, frac, r, window=window, teacher_N=teacher_N)
        key_s = tuple(rec["student_frames"])
        key_u = tuple(sorted(set(rec["teacher_frames"]) | set(rec["teacher_frames_B"])))
        # union-key dedup is skipped when the union spans the whole pool: for
        # small-N 16:4 scenes (e.g. N=26 -> union == 26 frames every time) a
        # union collision is structural, not a duplicated task — the shared
        # group is then the meaningful discriminator.
        full_pool = len(key_u) >= N
        if key_s in forb_s or (not full_pool and key_u in forb_union):
            continue
        forb_s.add(key_s)
        forb_union.add(key_u)
        train.append(rec)
    train_short = len(train) < n_train
    while len(train) < n_train:  # tiny-scene fallback: fill anyway (protocol_v1 parity)
        train.append(build_ab_train_pair(N, dense, frac, r, window=window, teacher_N=teacher_N))

    adapter = _FracAdapter(frac)
    rp = random.Random(P.stable_seed("ab_probe", dataset, scene))
    # a single-context probe must not replicate ANY train context (A or B)
    probe_collide = forb_union \
        | {tuple(sorted(p["teacher_frames"])) for p in train} \
        | {tuple(sorted(p["teacher_frames_B"])) for p in train}
    probe: List[Dict] = []
    tries = 0
    while len(probe) < n_probe and tries < 10000:
        tries += 1
        teacher, shared = P.build_pair(N, teacher_N, dense, adapter, rp, n_shared=N_SHARED)
        key_s = tuple(sorted(shared))
        key_t = tuple(sorted(teacher))
        if key_s in forb_s or key_t in probe_collide:
            continue
        forb_s.add(key_s)
        probe_collide.add(key_t)
        probe.append({"teacher_frames": [int(i) for i in teacher],
                      "student_frames": [int(i) for i in shared],
                      "teacher_N": teacher_N, "n_shared": N_SHARED, "kind": "fixed"})
    probe_short = len(probe) < n_probe
    while len(probe) < n_probe:
        teacher, shared = P.build_pair(N, teacher_N, dense, adapter, rp, n_shared=N_SHARED)
        probe.append({"teacher_frames": [int(i) for i in teacher],
                      "student_frames": [int(i) for i in shared],
                      "teacher_N": teacher_N, "n_shared": N_SHARED, "kind": "fixed"})
    return train, probe, {"train_dedup_short": train_short, "probe_dedup_short": probe_short}


def build_ab_scene_protocol(image_files: Sequence[str], scene: str, dataset: str = "scannetpp",
                            n_train: int = 10, n_probe: int = 2, window: int = WINDOW,
                            teacher_n_mode: str = "8") -> Dict:
    """Full per-scene v2ab manifest. `frac` comes from a protocol_v1 SiftCache
    over the VALID files; all sampling runs in valid-pool coordinates and is
    mapped back to global image_files indices at the end. Raises if fewer than
    MIN_SCENE_N valid images (scene dropped, protocol_v1 parity). teacher_N
    follows teacher_n_for(Nv, teacher_n_mode): "8" fixed 8:4 (DA3), "auto16"
    the old VGGT rule 16 if N>=16 else 8. Candidate shortage for fully-disjoint
    A/B extras is RECORDED via ab_overlap but does NOT degrade the scene
    (acceptable per the 16:4 VGGT convention); degradation only tracks bad
    files, N<12 pools, and dedup-space exhaustion."""
    image_files = [os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in image_files]
    N = len(image_files)
    valid_idx, bad_files = collect_valid_files(image_files)
    Nv = len(valid_idx)
    if Nv < MIN_SCENE_N:
        raise ValueError(f"{scene}: only {Nv} valid images (< {MIN_SCENE_N}), scene dropped")
    teacher_N = teacher_n_for(Nv, teacher_n_mode)
    valid_files = [image_files[i] for i in valid_idx]
    tau = P.compute_tau(valid_files)
    dense = tau > P.TAU_THRESHOLD
    sc = P.SiftCache(valid_files)
    train, probe, samp_meta = sample_ab_tasks(Nv, dense, sc.frac, dataset, scene,
                                              n_train=n_train, n_probe=n_probe, window=window,
                                              teacher_N=teacher_N)
    # pool coordinates -> global image_files indices
    for rec in train:
        for k in ("teacher_frames", "teacher_frames_B", "student_frames", "extras_A", "extras_B"):
            rec[k] = [valid_idx[i] for i in rec[k]]
    for rec in probe:
        for k in ("teacher_frames", "student_frames"):
            rec[k] = [valid_idx[i] for i in rec[k]]

    degraded = bool(bad_files) or Nv < DEGRADED_N \
        or samp_meta["train_dedup_short"] or samp_meta["probe_dedup_short"]
    reasons = []
    if Nv < DEGRADED_N:
        reasons.append(f"N_valid={Nv}<{DEGRADED_N}: A/B extras cannot be fully disjoint")
    if bad_files:
        reasons.append(f"{len(bad_files)} bad file(s) excluded from the sampling pool")
    if samp_meta["train_dedup_short"]:
        reasons.append("train pair space exhausted before n_train; duplicates allowed")
    if samp_meta["probe_dedup_short"]:
        reasons.append("probe pair space exhausted before n_probe; duplicates allowed")

    ab = [rec["ab_overlap"] for rec in train]
    nc = [rec["n_candidates"] for rec in train]
    n_extras = teacher_N - N_SHARED
    full_identity = bool(ab) and max(ab) >= n_extras  # B fully reuses A: u == 0
    degraded = degraded or full_identity
    if full_identity:
        reasons.append(f"A/B extras fully overlap (ab_overlap_max={max(ab)} >= "
                       f"n_extras={n_extras}): contexts A and B are identical, "
                       f"the cross-check carries no signal")
    if N >= 100:
        r = random.Random(42)
        idx = list(range(N))
        r.shuffle(idx)
        eval_frames = sorted(idx[:100])
    else:
        eval_frames = list(range(N))
    return {"scene": scene, "dataset": dataset, "protocol_version": "v2ab",
            "N": N, "N_valid": Nv, "teacher_N": teacher_N,
            "teacher_n_mode": teacher_n_mode, "tau": tau,
            "strategy": "ab_dense_window_sift" if dense else "ab_random_window_sift",
            "degraded": degraded, "degrade_reasons": reasons, "bad_files": bad_files,
            "train_pairs": train, "probe_pairs": probe,
            "ab_overlap_mean": float(np.mean(ab)) if ab else None,
            "ab_overlap_max": int(max(ab)) if ab else None,
            "n_candidates_mean": float(np.mean(nc)) if nc else None,
            "n_shared": N_SHARED, "student_slots": list(range(0, 2 * N_SHARED, 2)),
            "eval_frames": eval_frames, "image_files": image_files}


# ---------------------------------------------------------------------------
# Dataset discovery + CLI
# ---------------------------------------------------------------------------
def discover_scenes(ds: str) -> List[str]:
    """Scene names from the canonical dataset-class sources (constants /
    scene-list files), falling back to benchmark_dataset directory listing.
    Handles hiroom's '/'-containing names and the hash-named scannetpp scenes;
    '/' is sanitized to '__' at output-path construction time."""
    if ds == "hiroom":
        from depth_anything_3.utils.constants import HIROOM_SCENE_LIST_PATH
        with open(os.path.join(ROOT, HIROOM_SCENE_LIST_PATH)) as f:
            return [l.strip() for l in f.read().splitlines() if l.strip()]
    if ds in ("dtu", "dtu64"):
        from depth_anything_3.utils.constants import DTU64_SCENES, DTU_SCENES
        return list(DTU64_SCENES if ds == "dtu64" else DTU_SCENES)
    if ds == "7scenes":
        base = os.path.join(ROOT, "workspace", "benchmark_dataset", "7scenes", "7Scenes")
    else:
        base = os.path.join(ROOT, "workspace", "benchmark_dataset", ds)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"dataset dir not found: {base}")
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)) and d != "meshes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["eth3d", "7scenes", "scannetpp"])
    ap.add_argument("--out", default=os.path.join(ROOT, "workspace", "protocol_v2", "ab_manifests"))
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_probe", type=int, default=2)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--teacher_n", choices=["8", "auto16"], default="8",
                    help="per-scene teacher_N convention: '8' = fixed 8:4 (DA3 v2ab); "
                         "'auto16' = old VGGT manifest rule, 16:4 for N>=16 else 8:4")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    summary: Dict = {"out": args.out, "teacher_n": args.teacher_n, "datasets": {}}
    for ds in args.datasets:
        try:
            scenes = discover_scenes(ds)
        except Exception as e:
            print(f"[{ds}] dataset discovery failed: {e!r}", flush=True)
            summary["datasets"][ds] = {"error": repr(e)}
            continue
        fg_common.set_dataset(ds)
        ds_stat: Dict = {"scenes_total": len(scenes), "ok": 0, "failed": 0,
                         "degraded": 0, "skipped_existing": 0, "ab_overlap_means": []}
        summary["datasets"][ds] = ds_stat
        for scene in scenes:
            path = os.path.join(args.out, ds, f"{scene.replace('/', '__')}.json")
            tag = f"{ds}/{scene}"
            if os.path.exists(path) and not args.overwrite:
                print(f"[{tag}] exists, skip (use --overwrite)", flush=True)
                ds_stat["skipped_existing"] += 1
                continue
            try:
                data = fg_common.get_scene_data(scene)
                proto = build_ab_scene_protocol(list(data.image_files), scene, ds,
                                                n_train=args.n_train, n_probe=args.n_probe,
                                                window=args.window,
                                                teacher_n_mode=args.teacher_n)
                proto["gt_intrinsics"] = np.asarray(data.intrinsics).astype(float).tolist()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(proto, f)
                ds_stat["ok"] += 1
                if proto["degraded"]:
                    ds_stat["degraded"] += 1
                ds_stat["ab_overlap_means"].append(proto["ab_overlap_mean"])
                print(f"[{tag}] N={proto['N']} Nv={proto['N_valid']} tN={proto['teacher_N']} "
                      f"tau={proto['tau']:.3f} "
                      f"dense={proto['strategy'].startswith('ab_dense')} "
                      f"ab_ov_mean={proto['ab_overlap_mean']:.2f} "
                      f"degraded={proto['degraded']} {proto['degrade_reasons']}", flush=True)
            except Exception as e:
                ds_stat["failed"] += 1
                print(f"[{tag}] FAILED: {e!r}", flush=True)
        ovs = [o for o in ds_stat["ab_overlap_means"] if o is not None]
        if ovs:
            print(f"[{ds}] done: ok={ds_stat['ok']} degraded={ds_stat['degraded']} "
                  f"failed={ds_stat['failed']} ab_overlap_mean "
                  f"min/med/max={min(ovs):.2f}/{float(np.median(ovs)):.2f}/{max(ovs):.2f}",
                  flush=True)
        else:
            print(f"[{ds}] done: ok={ds_stat['ok']} failed={ds_stat['failed']}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {os.path.join(args.out, '_build_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
