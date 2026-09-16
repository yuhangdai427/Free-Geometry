#!/usr/bin/env python3
"""Phase-4 control manifests: student clustered-block selection ablation.

Same scenes, same teacher SETS as artifacts/diagnostics/final_protocol/<ds>,
but the 4 shared frames are a CONTIGUOUS block of the sorted teacher set
(control) instead of the equidistant quarter (protocol v1). The teacher list
is re-laid-out so the block sits at slots [0,2,4,6] (STUDENT_INDICES).
Only the 8 sub8 scenes. Eval frames copied verbatim.

Run from repo root: python diagnostics/free_geometry/build_sub8_studentctl.py
"""

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from common import load_manifest, save_manifest, stable_seed  # noqa: E402

SUB8 = {
    "scannetpp": ["7831862f02", "bde1e479ad"],
    "7scenes": ["chess", "office"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12"],
    "eth3d": ["courtyard", "office"],
}
SLOTS = [0, 2, 4, 6]


def convert_pair(pair, seed_tag):
    t_sorted = sorted(pair["teacher_frames"])
    n = len(t_sorted)
    r = random.Random(stable_seed("studentctl", *seed_tag))
    k = r.randint(0, n - 4)
    shared = t_sorted[k:k + 4]
    extras = [f for f in t_sorted if f not in set(shared)]
    teacher = [None] * n
    for slot, s in zip(SLOTS, shared):
        teacher[slot] = s
    rest = [i for i in range(n) if i not in SLOTS]
    for slot, e in zip(rest, extras):
        teacher[slot] = e
    return {"teacher_frames": teacher, "student_frames": list(shared)}


def main():
    for ds, scenes in SUB8.items():
        src = load_manifest(os.path.join(
            _REPO, "artifacts", "diagnostics", "final_protocol", ds, "scene_manifest.json"))
        out = {"dataset": ds, "seed_scene_select": 43,
               "note": "Phase-4 control: clustered-block student (vs equidistant), "
                       "teacher sets identical to final_protocol",
               "run_root": os.path.join("artifacts", "diagnostics",
                                        "final_protocol_studentctl", ds),
               "scenes": {}}
        for scene in scenes:
            sc = dict(src["scenes"][scene])
            sc["train_pairs"] = [convert_pair(p, (ds, scene, "tr", i))
                                 for i, p in enumerate(sc["train_pairs"])]
            sc["probe_pairs"] = [convert_pair(p, (ds, scene, "pr", i))
                                 for i, p in enumerate(sc["probe_pairs"])]
            out["scenes"][scene] = sc
        dst = os.path.join(_REPO, "artifacts", "diagnostics",
                           "final_protocol_studentctl", ds)
        save_manifest(out, os.path.join(dst, "scene_manifest.json"))
        print(f"[{ds}] wrote {len(out['scenes'])} scenes -> {dst}")


if __name__ == "__main__":
    main()
