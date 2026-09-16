#!/usr/bin/env python3
"""Single-scene controlled manifest: scannetpp 7bc286c1b6 with the RANDOM
selection branch forced (strategy=random_forced), everything else identical to
final_protocol v1.1.

- train_pairs/probe_pairs: rebuilt via build_final_manifest's non-dense branch
  (build_pairs(dense=False)) with the SAME stable_seed("final", tag, ds, scene)
  discipline -> exactly the pairs this scene would have drawn if tau had been
  classified <=0.55. Collision/uniqueness logic replicated from sample().
- eval32_frames: copied VERBATIM from the original final_protocol/scannetpp
  manifest entry (comparability hinge).
- Output run_root: artifacts/diagnostics/final_protocol/scannetpp_7bc_random/

New file; imports build_final_manifest but does not modify any existing module.
Run from repo root."""

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import common  # noqa: E402
from common import get_scene_data, stable_seed  # noqa: E402
from build_final_manifest import build_pairs  # noqa: E402

SCENE = "7bc286c1b6"
DS = "scannetpp"
SRC = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol", "scannetpp",
                   "scene_manifest.json")
OUT_ROOT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol",
                        "scannetpp_7bc_random")
N_TRAIN, N_PROBE = 10, 2


def sample_random(tag, n, N, teacher_N, forb_t, forb_s):
    """Verbatim replica of build_final_manifest.sample() with dense=False."""
    r = random.Random(stable_seed("final", tag, DS, SCENE))
    pairs, tries = [], 0
    used_frames = set()
    while len(pairs) < n and tries < 10000:
        tries += 1
        teacher, shared = build_pairs(N, teacher_N, False, None, r)
        key_t, key_s = tuple(sorted(teacher)), tuple(shared)
        if key_t in forb_t or key_s in forb_s:
            continue
        forb_t.add(key_t)
        forb_s.add(key_s)
        used_frames |= set(teacher)
        pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
    if len(pairs) < n:
        while len(pairs) < n:
            best, best_gain = None, -1
            for _ in range(64):
                teacher, shared = build_pairs(N, teacher_N, False, None, r)
                gain = len(set(teacher) - used_frames)
                if gain > best_gain:
                    best, best_gain = (teacher, shared), gain
                if gain == teacher_N:
                    break
            teacher, shared = best
            used_frames |= set(teacher)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        print(f"  [relax] {SCENE}: pair space exhausted", flush=True)
    return pairs


def main():
    src = json.load(open(SRC))
    orig = src["scenes"][SCENE]
    common.set_dataset(DS)
    data = get_scene_data(SCENE)
    N = len(data.image_files)
    assert N == orig["num_frames_total"], (N, orig["num_frames_total"])
    teacher_N = orig["teacher_N"]

    train_pairs = sample_random("train", N_TRAIN, N, teacher_N, set(), set())
    probe_pairs = sample_random(
        "probe", N_PROBE, N, teacher_N,
        {tuple(sorted(p["teacher_frames"])) for p in train_pairs},
        {tuple(p["student_frames"]) for p in train_pairs})

    entry = {
        "num_frames_total": orig["num_frames_total"],
        "file_sha256": orig.get("file_sha256", {}),
        "num_frames_with_gt_depth": orig["num_frames_with_gt_depth"],
        "adaptation_pool_size": orig["adaptation_pool_size"],
        "teacher_N": teacher_N,
        "tau": orig["tau"],
        "strategy": "random_forced",
        "eval32_frames": list(orig["eval32_frames"]),  # verbatim: comparability
        "train_pairs": train_pairs,
        "probe_pairs": probe_pairs,
    }
    # train_arms.py layout assertion: student == teacher[STUDENT_INDICES]
    for p in train_pairs + probe_pairs:
        assert p["student_frames"] == [p["teacher_frames"][i] for i in (0, 2, 4, 6)], p

    manifest = {
        "dataset": DS,
        "run_root": OUT_ROOT,
        "seed_scene_select": src.get("seed_scene_select", 43),
        "note": ("controlled single-scene ablation 2026-09-14: 7bc286c1b6 forced to "
                 "the random branch (tau=0.5523 was the only scannetpp scene "
                 "dispatched to dense_equidistant_sift). eval32_frames identical to "
                 "final_protocol/scannetpp; pairs rebuilt with stable_seed('final',...)"
                 " random-branch logic; strategy=random_forced."),
        "scenes": {SCENE: entry},
    }
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "scene_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"N={N} teacher_N={teacher_N} tau={entry['tau']:.4f} strategy=random_forced")
    print("eval32 identical:", entry["eval32_frames"] == orig["eval32_frames"])
    print("train0 teacher:", train_pairs[0]["teacher_frames"])
    print("train0 student:", train_pairs[0]["student_frames"])
    print("orig(dense) train0 teacher:", orig["train_pairs"][0]["teacher_frames"])
    print("Wrote", os.path.join(OUT_ROOT, "scene_manifest.json"))


if __name__ == "__main__":
    main()
