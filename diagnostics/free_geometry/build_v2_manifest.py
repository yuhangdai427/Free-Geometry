#!/usr/bin/env python3
"""Protocol-v2 manifests: N-correlated pair count + dense-long partition.

Rules (discussed 2026-09-13):
- n_pairs = clamp(round(N/50), 10, 20); probe pairs = 2.
- Dense-long branch (tau>0.55 and N>=192): split the sequence into n_pairs
  CONTIGUOUS equal segments; each segment is a small-scene problem.
    mode=tile: equidistant superset M=min(seg_len,24) with per-pair offset,
               shared=sup[::M//4][:4]; extras prefer SIFT match-fraction within
               the SELF-CALIBRATED band [q30,q90] of the pair's candidate
               distribution (soft filter, backfill out-of-band).
    mode=rand: random 16 within segment (sorted), shared=sorted[::4].
- teacher_N=16 (segment>=24) else 8; shared at teacher-list slots [0,2,4,6].
- eval frames: benchmark-100 (seed 42) if N>=100 else all.

Run from repo root: python diagnostics/free_geometry/build_v2_manifest.py --mode tile|rand [--datasets 7scenes ...]
"""

import argparse
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import build_final_manifest as BF  # noqa: E402
from common import save_manifest, stable_seed  # noqa: E402

OUT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_v2")


def n_pairs_for(N):
    return int(np.clip(round(N / 50), 10, 20))


def segments(N, k):
    """k contiguous equal segments over range(N). Returns list of index lists."""
    bounds = [int(i * N / k) for i in range(k + 1)]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(k)]


def build_pair_in_segment(seg, teacher_N, mode, sc, r):
    S = len(seg)
    if mode == "tile":
        M = min(S, teacher_N * 3 // 2)
        step = S / M
        off = r.uniform(0, step)
        sup = sorted(set(seg[int(off + i * step)] for i in range(M)))[:M]
        stride = max(1, M // 4)
        shared = sup[::stride][:4]
        rest = [f for f in sup if f not in set(shared)]
        fr = np.array([np.mean([sc.frac(f, s) for s in shared]) for f in rest])
        lo, hi = np.quantile(fr, 0.3), np.quantile(fr, 0.9)
        in_band = [f for f, c in zip(rest, fr) if lo <= c <= hi]
        out_band = [f for f, c in zip(rest, fr) if not (lo <= c <= hi)]
        extras = sorted((in_band + out_band)[:teacher_N - 4])
    else:
        t = sorted(r.sample(seg, teacher_N))
        shared = t[:: teacher_N // 4][:4]
        extras = [f for f in t if f not in set(shared)]
    shared = sorted(shared)
    return BF.assemble_teacher_list(shared, extras), shared


def build_scene(ds, scene, mode):
    BF.common.set_dataset(ds)
    data = BF.get_scene_data(scene)
    image_files = list(data.image_files)
    N = len(image_files)
    assert N >= 8, f"{scene}: N={N}<8"
    tau = BF.compute_tau(image_files)
    dense_long = tau > BF.TAU_THRESHOLD and N >= 192
    k = n_pairs_for(N)
    teacher_N = 16
    sc = BF.SiftCache(image_files) if (dense_long and mode == "tile") else None

    def sample(tag, n):
        r = random.Random(stable_seed("v2", mode, tag, ds, scene))
        segs = segments(N, k)
        order = segs.copy()
        r.shuffle(order)
        pairs = []
        for i in range(n):
            seg = order[i % len(order)]
            teacher, shared = build_pair_in_segment(seg, teacher_N, mode, sc, r)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        return pairs

    train_pairs = sample("train", k)
    probe_pairs = sample("probe", 2)

    if N >= 100:
        r = random.Random(42)
        idx = list(range(N))
        r.shuffle(idx)
        eval_frames = sorted(idx[:100])
    else:
        eval_frames = list(range(N))
    return {
        "num_frames_total": N,
        "num_frames_with_gt_depth": sum(1 for p in data.aux.gt_depth_files if os.path.exists(p)),
        "adaptation_pool_size": N, "teacher_N": teacher_N, "tau": tau,
        "n_train_pairs": k,
        "strategy": f"v2_{mode}" if dense_long else "random",
        "eval32_frames": eval_frames,
        "train_pairs": train_pairs, "probe_pairs": probe_pairs,
        "file_sha256": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tile", "rand"], required=True)
    ap.add_argument("--datasets", nargs="*", default=["7scenes"])
    args = ap.parse_args()
    ds_scenes = {"7scenes": list(BF.SEVEN_SCENES)}
    for ds in args.datasets:
        run_root = os.path.join(OUT, f"{ds}_{args.mode}")
        manifest = {"dataset": ds, "run_root": run_root, "seed_scene_select": 43,
                    "note": f"protocol v2 selection ablation: mode={args.mode}, "
                            f"n_pairs~N/50, contiguous segments, self-calibrated filter",
                    "scenes": {}}
        for scene in ds_scenes[ds]:
            manifest["scenes"][scene] = build_scene(ds, scene, args.mode)
            e = manifest["scenes"][scene]
            print(f"[{ds}_{args.mode}] {scene}: N={e['num_frames_total']} "
                  f"tau={e['tau']:.3f} pairs={e['n_train_pairs']} {e['strategy']}", flush=True)
        save_manifest(manifest, os.path.join(run_root, "scene_manifest.json"))
        print(f"wrote {run_root}")


if __name__ == "__main__":
    main()
