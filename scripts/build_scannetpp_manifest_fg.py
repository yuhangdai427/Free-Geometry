#!/usr/bin/env python3
"""Build the scannetpp scene manifest (fresh) with the SAME frame-selection
strategy as the campaign (protocol_v1.build_scene_protocol, N-dispatch rule).
Scene list = the 20 scannetpp scenes used by the ndispatch campaign.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg

OUT = "workspace/ndispatch/scannetpp"

SCENES = ("09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 "
          "40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 "
          "acd95847c5 bcd246daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 "
          "f3d64c30f8 fb5a96b1a2").split()


def main():
    fg.set_dataset("scannetpp")
    out = {"dataset": "scannetpp",
           "note": "scannetpp manifest (2026-09-23) for the longer-is-better "
                   "probe: fresh build, protocol_v1 auto rule, eval = all frames",
           "run_root": OUT, "scenes": {}}
    for scene in SCENES:
        try:
            sd = fg.get_scene_data(scene)
            files = list(sd.image_files)
        except Exception as exc:
            print(f"scannetpp/{scene}: SKIP ({exc})")
            continue
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset="scannetpp",
                                       n_train=10, n_shared=4, teacher_N=None)
        ev = list(range(N))
        out["scenes"][scene] = {
            "num_frames_total": N,
            "num_frames_with_gt_depth": N,
            "eval32_frames": ev,
            "adaptation_pool_size": N - len(set(ev)),
            "train_pairs": proto["train_pairs"],
            "probe_pairs": proto["probe_pairs"],
            "teacher_N": proto["teacher_N"],
            "tau": proto.get("tau"),
            "strategy": proto.get("strategy"),
        }
        print(f"scannetpp/{scene}: N={N} tN={proto['teacher_N']} tau={proto.get('tau')} "
              f"pairs={len(proto['train_pairs'])}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scene_manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT}/scene_manifest.json ({len(out['scenes'])} scenes)")


if __name__ == "__main__":
    main()
