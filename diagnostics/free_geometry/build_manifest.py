#!/usr/bin/env python3
"""Build the frozen 6-scene ScanNet++ manifest for the diagnostic bake-off."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import RUN_ROOT_DEFAULT, build_scene_manifest, save_manifest, verify_disjoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default=RUN_ROOT_DEFAULT)
    ap.add_argument("--n_scenes", type=int, default=6)
    args = ap.parse_args()

    manifest = build_scene_manifest(args.run_root, n_scenes=args.n_scenes)
    verify_disjoint(manifest)

    out = os.path.join(args.run_root, "scene_manifest.json")
    save_manifest(manifest, out)

    print(f"Wrote {out}")
    for scene, sc in manifest["scenes"].items():
        print(
            f"  {scene}: frames={sc['num_frames_with_gt_depth']}/{sc['num_frames_total']} "
            f"eval32={sc['eval32_frames'][:4]}... pool={sc['adaptation_pool_size']} "
            f"train_pairs={len(sc['train_pairs'])} probe_pairs={len(sc['probe_pairs'])}"
        )


if __name__ == "__main__":
    main()
