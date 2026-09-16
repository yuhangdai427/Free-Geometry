#!/usr/bin/env python3
"""Dynamic student-count manifests (idea-2 family): teacher:student ratios.

- 7scenes_s8t32   : 32:8 equidistant spread (no SIFT), 10 pairs
- scannetpp_s8t24 : 24:8 random (tau<=0.55 branch), 10 pairs
- hiroom_s2t8     : N<16 scenes, 8:2 random, 10 pairs
- hiroom_s2t4     : N<16 scenes, 4:2 random, 10 pairs

Student = equidistant n_stu quarter of the SORTED teacher set; shared frames sit
at teacher-list slots [0,2,...,2*(n_stu-1)]. Eval frames copied from
final_protocol manifests (same frames => paired vs v1 numbers).
Run from repo root.
"""

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import build_final_manifest as BF  # noqa: E402
from common import load_manifest, save_manifest, stable_seed  # noqa: E402

OUT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol_s8")


def assemble(teacher_set_sorted, shared, n_stu):
    slots = list(range(0, 2 * n_stu, 2))
    teacher = [None] * len(teacher_set_sorted)
    for slot, s in zip(slots, shared):
        teacher[slot] = s
    rest = [i for i in range(len(teacher)) if i not in slots]
    extras = [f for f in teacher_set_sorted if f not in set(shared)]
    for slot, e in zip(rest, extras):
        teacher[slot] = e
    return teacher


def build(ds, scene, teacher_N, n_stu, mode, n_train, eval_frames):
    BF.common.set_dataset(ds)
    data = BF.get_scene_data(scene)
    N = len(data.image_files)
    assert N >= teacher_N, f"{scene}: N={N}<{teacher_N}"

    def one_pair(r):
        if mode == "equidistant":
            step = N / teacher_N
            off = r.uniform(0, step)
            t = sorted(set(int(off + i * step) for i in range(teacher_N)))[:teacher_N]
        else:
            t = sorted(r.sample(range(N), teacher_N))
        stride = max(1, teacher_N // n_stu)
        shared = sorted(t[::stride][:n_stu])
        return assemble(t, shared, n_stu), shared

    def sample(tag, n):
        r = random.Random(stable_seed("dyn", tag, ds, scene, teacher_N, n_stu))
        out = []
        for _ in range(n):
            t, s = one_pair(r)
            out.append({"teacher_frames": t, "student_frames": list(s)})
        return out

    return {"num_frames_total": N, "num_frames_with_gt_depth": N,
            "adaptation_pool_size": N, "teacher_N": teacher_N, "n_student": n_stu,
            "strategy": f"{mode}_t{teacher_N}_s{n_stu}", "eval32_frames": eval_frames,
            "train_pairs": sample("train", n_train), "probe_pairs": sample("probe", 2),
            "file_sha256": {}}


CONFIGS = {
    "7scenes_s8t32": ("7scenes", list(BF.SEVEN_SCENES), 32, 8, "equidistant", 10),
    "scannetpp_s8t24": ("scannetpp", list(BF.SCANNETPP_SCENE_LIST), 24, 8, "random", 10),
    "hiroom_s2t8": ("hiroom", None, 8, 2, "random", 10),
    "hiroom_s2t4": ("hiroom", None, 4, 2, "random", 10),
    "7scenes_s8t16": ("7scenes", list(BF.SEVEN_SCENES), 16, 8, "equidistant", 10),
}


def main():
    hiroom_all = [l.strip() for l in open(BF.HIROOM_LIST).read().splitlines() if l.strip()]
    main_mani = load_manifest(os.path.join(
        _REPO, "artifacts", "diagnostics", "final_protocol", "hiroom", "scene_manifest.json"))
    hiroom_small = [s for s in hiroom_all
                    if main_mani["scenes"][s]["num_frames_total"] < 16]
    for name, (ds, scenes, tN, nS, mode, ntr) in CONFIGS.items():
        if scenes is None:
            scenes = hiroom_small
        ref_mani = load_manifest(os.path.join(
            _REPO, "artifacts", "diagnostics", "final_protocol", ds, "scene_manifest.json"))
        run_root = os.path.join(OUT, name)
        mani = {"dataset": ds, "run_root": run_root, "seed_scene_select": 43,
                "note": f"dynamic-student ablation {tN}:{nS} ({mode})",
                "scenes": {}}
        for scene in scenes:
            ev = ref_mani["scenes"][scene]["eval32_frames"]
            mani["scenes"][scene] = build(ds, scene, tN, nS, mode, ntr, ev)
            print(f"[{name}] {scene}: N={mani['scenes'][scene]['num_frames_total']}",
                  flush=True)
        save_manifest(mani, os.path.join(run_root, "scene_manifest.json"))
        print(f"wrote {run_root} ({len(mani['scenes'])} scenes)")


if __name__ == "__main__":
    main()
