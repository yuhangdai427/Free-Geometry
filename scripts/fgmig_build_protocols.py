#!/usr/bin/env python3
"""Build per-scene protocol JSONs for the 3-model migration (runs in da3 env).

Output: workspace/fgmig/protocols/<ds>/<scene>.json with image_files (abs),
train/probe pairs (8:4, GT-free tau-dispatched), eval frames (100-or-all).
"""
import argparse
import json
import os
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))
sys.path.insert(0, os.path.join(ROOT, "src", "free_geometry"))

import common as fg_common  # noqa: E402
from free_geometry import sampling  # noqa: E402

SCENES = {
    "7scenes": ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
    "eth3d": ["courtyard", "delivery_area", "electro", "facade", "kicker", "office",
              "pipes", "playground", "relief", "relief_2", "terrains"],
    "hiroom": [l.strip() for l in open(os.path.join(
        ROOT, "workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt"))
        .read().splitlines() if l.strip()],
    "scannetpp": ["09c1414f1b", "1ada7a0617", "21d970d8de", "286b55a2bf", "38d58a7a31",
                  "3e8bba0176", "40aec5fffa", "578511c8a9", "5f99900f09", "7831862f02",
                  "7bc286c1b6", "9071e139d9", "acd95847c5", "bcd2436daf", "bde1e479ad",
                  "c4c04e6d6c", "c5439f4607", "cc5237fd77", "f3d64c30f8", "fb5a96b1a2"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["7scenes", "eth3d", "hiroom", "scannetpp"])
    ap.add_argument("--out_root", default=os.path.join(ROOT, "workspace/fgmig/protocols"))
    args = ap.parse_args()

    for ds in args.datasets:
        fg_common.set_dataset(ds)
        out_dir = os.path.join(args.out_root, ds)
        os.makedirs(out_dir, exist_ok=True)
        for scene in SCENES[ds]:
            path = os.path.join(out_dir, f"{scene.replace('/', '__')}.json")
            if os.path.exists(path):
                print(f"[{ds}/{scene}] exists, skip", flush=True)
                continue
            data = fg_common.get_scene_data(scene)
            image_files = [os.path.join(ROOT, p) if not os.path.isabs(p) else p
                           for p in data.image_files]
            proto = sampling.build_protocol(image_files, scene, ds)
            proto["image_files"] = image_files
            proto["gt_intrinsics"] = __import__("numpy") \
                .asarray(data.intrinsics).astype(float).tolist()
            with open(path, "w") as f:
                json.dump(proto, f)
            print(f"[{ds}/{scene}] N={proto['N']} tau={proto['tau']:.3f} "
                  f"strategy={proto['strategy']}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
