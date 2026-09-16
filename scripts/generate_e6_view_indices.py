#!/usr/bin/env python3
"""Generate auditable, shared nested training/evaluation frame pools for E6."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))

from vggt.vggt.test_time_adaption.dataset import DATASET_REGISTRY
from vggt.vggt.test_time_adaption.nested_view_sampling import make_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic E6 frame-index JSONL files")
    parser.add_argument("--dataset", default="eth3d", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--samples_per_scene", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=16, choices=[16, 32])
    parser.add_argument(
        "--skip_insufficient",
        action="store_true",
        help="Skip scenes with fewer frames than --pool_size and record them in the manifest.",
    )
    parser.add_argument(
        "--use_all_if_insufficient",
        action="store_true",
        help="For evaluation pools, retain every available frame when a scene has fewer than --pool_size.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="train")
    args = parser.parse_args()

    dataset = DATASET_REGISTRY[args.dataset]()
    frame_counts = {}
    for scene_id in dataset.SCENES:
        data = dataset.get_data(scene_id)
        image_files = data["image_files"] if isinstance(data, dict) else data.image_files
        frame_counts[scene_id] = len(image_files)

    short_scenes = {scene_id: count for scene_id, count in frame_counts.items() if count < args.pool_size}
    if args.skip_insufficient and args.use_all_if_insufficient:
        raise ValueError("Choose only one of --skip_insufficient or --use_all_if_insufficient")
    if short_scenes and not args.skip_insufficient and not args.use_all_if_insufficient:
        details = ", ".join(f"{scene}={count}" for scene, count in sorted(short_scenes.items()))
        raise ValueError(f"Cannot make unique P{args.pool_size} pools: {details}")
    skipped = short_scenes if args.skip_insufficient else {}
    eligible = {
        scene_id: count
        for scene_id, count in frame_counts.items()
        if scene_id not in skipped
    }
    if skipped:
        print(f"Skipping {len(skipped)} scene(s) below P{args.pool_size}: {skipped}")
    elif short_scenes:
        print(f"Using all frames for {len(short_scenes)} short scene(s): {short_scenes}")

    manifest = {
        "dataset": args.dataset,
        "pool_size": args.pool_size,
        "samples_per_scene": args.samples_per_scene,
        "eligible_scenes": sorted(eligible),
        "skipped_scenes": skipped,
        "short_scenes_using_all_frames": short_scenes if args.use_all_if_insufficient else {},
        "seeds": args.seeds,
    }
    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_p{args.pool_size}_manifest.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Wrote manifest: {manifest_path}")

    for seed in args.seeds:
        records = make_records(
            eligible,
            seed=seed,
            samples_per_scene=args.samples_per_scene,
            pool_size=args.pool_size,
            allow_short=args.use_all_if_insufficient,
        )
        path = os.path.join(args.output_dir, f"{args.prefix}_seed{seed}_p{args.pool_size}.jsonl")
        write_jsonl(records, path)
        print(f"Wrote {len(records)} records: {path}")


if __name__ == "__main__":
    main()
