#!/usr/bin/env python3
"""Build the scannetpp SELF-EVO style manifest (2026-09-23).

Same scene list and protocol wrapper as build_scannetpp_manifest_fg.py, but
pairs are SelfEvo-faithful: student = teacher window's FIRST + LAST frame plus
L-2 random middle frames, L ~ U{2, min(6, teacher_N//2)} drawn PER PAIR
(endpoint-anchored, variable-length students), kind="se".
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg

OUT = "workspace/ndispatch/scannetpp_selfevo"

SCENES = ("09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 "
          "40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 "
          "acd95847c5 bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 "
          "f3d64c30f8 fb5a96b1a2").split()


def main():
    fg.set_dataset("scannetpp")
    out = {"dataset": "scannetpp",
           "note": "scannetpp SELF-EVO manifest (2026-09-23): endpoint-anchored "
                   "variable-length students (L ~ U{2, min(6, tN//2)} per pair), "
                   "protocol_v1 build_scene_protocol(selfevo=True)",
           "run_root": OUT, "scenes": {}}
    from collections import Counter
    for scene in SCENES:
        try:
            sd = fg.get_scene_data(scene)
            files = list(sd.image_files)
        except Exception as exc:
            print(f"scannetpp/{scene}: SKIP ({exc})")
            continue
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset="scannetpp",
                                       n_train=10, n_shared=4, teacher_N=None,
                                       selfevo=True)
        pairs = proto["train_pairs"]
        ls = Counter(len(p["student_frames"]) for p in pairs)
        # 结构验证：student ⊆ teacher、集合去重
        assert all(set(p["student_frames"]) <= set(p["teacher_frames"]) for p in pairs)
        tsets = {tuple(sorted(p["teacher_frames"])) for p in pairs}
        out["scenes"][scene] = {
            "num_frames_total": N,
            "num_frames_with_gt_depth": N,
            "eval32_frames": list(range(N)),
            "adaptation_pool_size": 0,
            "train_pairs": pairs,
            "probe_pairs": proto["probe_pairs"],
            "teacher_N": proto["teacher_N"],
            "tau": proto.get("tau"),
            "strategy": proto.get("strategy"),
        }
        print(f"scannetpp/{scene}: N={N} tN={proto['teacher_N']} "
              f"kind={pairs[0].get('kind')} L分布={dict(sorted(ls.items()))} "
              f"teacher去重={len(tsets)}/10")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "scene_manifest.json"), "w"), indent=1)
    print(f"-> {OUT}/scene_manifest.json ({len(out['scenes'])} scenes)")


if __name__ == "__main__":
    main()
