#!/usr/bin/env python3
"""final_protocol_t8 manifests: forced teacher_N=8 (8:4) variant of the frozen
final-protocol manifests. Mirrors build_final_manifest.py exactly (same tau
dispatch, same stable seeds, same eval-frames rule, same coverage-greedy
fallback); the ONLY change is teacher_N=8 for every scene (N>=8).

For scenes already at teacher_N=8 (N<16) the sampled pairs are IDENTICAL to the
frozen final_protocol manifests (same seed, same draws); for N>=16 scenes the
pairs differ by construction (8-frame teacher windows).

Run from repo root:  python diagnostics/free_geometry/build_t8_manifest.py
"""

import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import build_final_manifest as BF  # noqa: E402
import common  # noqa: E402
from common import get_scene_data, save_manifest, stable_seed  # noqa: E402

OUT_ROOT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_t8")
TEACHER_N = 8


def build_scene_entry_t8(ds, scene):
    common.set_dataset(ds)
    data = get_scene_data(scene)
    image_files = list(data.image_files)
    N = len(image_files)
    entry = {"num_frames_total": N}
    if N < 8:
        return None, {"scene": scene, "N": N, "dropped": "N<8"}
    tau = BF.compute_tau(image_files)
    dense = tau > BF.TAU_THRESHOLD
    sc = BF.SiftCache(image_files) if dense else None

    def sample(tag, n, forb_t, forb_s):
        r = random.Random(stable_seed("final", tag, ds, scene))
        pairs, tries = [], 0
        used_frames = set()
        while len(pairs) < n and tries < 10000:
            tries += 1
            teacher, shared = BF.build_pairs(N, TEACHER_N, dense, sc, r)
            key_t, key_s = tuple(sorted(teacher)), tuple(shared)
            if key_t in forb_t or key_s in forb_s:
                continue
            forb_t.add(key_t)
            forb_s.add(key_s)
            used_frames |= set(teacher)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        while len(pairs) < n:  # coverage-greedy fallback (tiny scenes)
            best, best_gain = None, -1
            for _ in range(64):
                teacher, shared = BF.build_pairs(N, TEACHER_N, dense, sc, r)
                gain = len(set(teacher) - used_frames)
                if gain > best_gain:
                    best, best_gain = (teacher, shared), gain
                if gain == TEACHER_N:
                    break
            teacher, shared = best
            used_frames |= set(teacher)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        return pairs

    train_pairs = sample("train", BF.N_TRAIN, set(), set())
    probe_pairs = sample("probe", BF.N_PROBE,
                         {tuple(sorted(p["teacher_frames"])) for p in train_pairs},
                         {tuple(p["student_frames"]) for p in train_pairs})

    if N >= 100:
        r = random.Random(42)
        idx = list(range(N))
        r.shuffle(idx)
        eval_frames = sorted(idx[:100])
    else:
        eval_frames = list(range(N))

    n_gt = sum(1 for p in data.aux.gt_depth_files if os.path.exists(p))
    entry.update({
        "num_frames_with_gt_depth": n_gt,
        "adaptation_pool_size": N,
        "teacher_N": TEACHER_N,
        "tau": tau,
        "strategy": "dense_equidistant_sift" if dense else "random",
        "eval32_frames": eval_frames,
        "train_pairs": train_pairs,
        "probe_pairs": probe_pairs,
    })
    info = {"scene": scene, "N": N, "tau": round(tau, 3), "teacher_N": TEACHER_N,
            "strategy": entry["strategy"], "eval_n": len(eval_frames),
            "gt_coverage": round(n_gt / N, 2)}
    return entry, info


def main():
    from depth_anything_3.utils.constants import ETH3D_SCENES
    datasets = {
        "scannetpp": list(BF.DATASETS["scannetpp"]),
        "7scenes": list(BF.DATASETS["7scenes"]),
        "hiroom": list(BF.DATASETS["hiroom"]),
        "eth3d": list(ETH3D_SCENES),
    }
    for ds, scenes in datasets.items():
        run_root = os.path.join(OUT_ROOT, ds)
        manifest = {"dataset": ds, "run_root": run_root, "seed_scene_select": 43,
                    "note": "final protocol t8 variant (2026-09-16): identical rules to "
                            "final_protocol v1 EXCEPT teacher_N=8 forced for all scenes "
                            "(8:4 ratio study); same stable seeds",
                    "scenes": {}}
        infos = []
        for scene in scenes:
            entry, info = build_scene_entry_t8(ds, scene)
            infos.append(info)
            if entry is not None:
                manifest["scenes"][scene] = entry
            print(f"[{ds}] {info}", flush=True)
        save_manifest(manifest, os.path.join(run_root, "scene_manifest.json"))
    print("DONE", OUT_ROOT)


if __name__ == "__main__":
    main()
