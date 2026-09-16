#!/usr/bin/env python3
"""Variant manifests: teacher view-count ablations on top of protocol v1.

- hiroom_r8   : 30 hiroom scenes, teacher_N=8 (random branch), eval=allv
- 7scenes_se24: 7 7scenes scenes, teacher_N=24 (dense branch w/ SIFT filter), eval=benchmark-100

Pair/mask seeds identical scheme to final_protocol ("final", tag, ds, scene) so
only teacher_N differs. Run from repo root."""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import build_final_manifest as BF  # noqa: E402
from common import save_manifest, stable_seed  # noqa: E402

VARIANTS = {
    "hiroom_r8": {"ds": "hiroom", "teacher_N": 8,
                  "scenes": [l.strip() for l in open(BF.HIROOM_LIST).read().splitlines() if l.strip()]},
    "7scenes_se24": {"ds": "7scenes", "teacher_N": 24, "scenes": list(BF.SEVEN_SCENES)},
}
OUT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_variants")


def build(ds, scene, teacher_N, n_train=None, n_probe=None):
    import numpy as np
    from common import get_scene_data
    BF.common.set_dataset(ds)
    data = get_scene_data(scene)
    image_files = list(data.image_files)
    N = len(image_files)
    assert N >= teacher_N, f"{scene}: N={N} < teacher_N={teacher_N}"
    tau = BF.compute_tau(image_files)
    dense = tau > BF.TAU_THRESHOLD and teacher_N > 8  # rand8: hiroom tau<=0.55 anyway
    sc = BF.SiftCache(image_files) if dense else None

    def sample(tag, n, forb_t, forb_s):
        r = random.Random(stable_seed("final", tag, ds, scene, f"t{teacher_N}"))
        pairs, tries = [], 0
        while len(pairs) < n and tries < 10000:
            tries += 1
            teacher, shared = BF.build_pairs(N, teacher_N, dense, sc, r)
            key_t, key_s = tuple(sorted(teacher)), tuple(shared)
            if key_t in forb_t or key_s in forb_s:
                continue
            forb_t.add(key_t)
            forb_s.add(key_s)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        while len(pairs) < n:  # small scene: pair space exhausted, reuse (masks differ per step)
            teacher, shared = BF.build_pairs(N, teacher_N, dense, sc, r)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
            print(f"  [relax] {scene}: pair space exhausted", flush=True)
        return pairs

    train_pairs = sample("train", n_train or BF.N_TRAIN, set(), set())
    probe_pairs = sample("probe", n_probe or BF.N_PROBE,
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
    return {
        "num_frames_total": N, "num_frames_with_gt_depth": n_gt,
        "adaptation_pool_size": N, "teacher_N": teacher_N, "tau": tau,
        "strategy": "dense_equidistant_sift" if dense else "random",
        "eval32_frames": eval_frames,
        "train_pairs": train_pairs, "probe_pairs": probe_pairs,
        "file_sha256": {},
    }


def main():
    for name, cfg in VARIANTS.items():
        ds, tN = cfg["ds"], cfg["teacher_N"]
        run_root = os.path.join(OUT, name)
        manifest = {"dataset": ds, "run_root": run_root, "seed_scene_select": 43,
                    "note": f"protocol v1 teacher-N ablation: teacher_N={tN}, "
                            f"same seeds as final_protocol", "scenes": {}}
        for scene in cfg["scenes"]:
            manifest["scenes"][scene] = build(ds, scene, tN)
            e = manifest["scenes"][scene]
            print(f"[{name}] {scene}: N={e['num_frames_total']} tau={e['tau']:.3f} "
                  f"strategy={e['strategy']}", flush=True)
        save_manifest(manifest, os.path.join(run_root, "scene_manifest.json"))
        print(f"[{name}] wrote {len(manifest['scenes'])} scenes -> {run_root}")


if __name__ == "__main__":
    main()
