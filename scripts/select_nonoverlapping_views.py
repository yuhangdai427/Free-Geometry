#!/usr/bin/env python3
"""Select per-scene non-overlapping view subsets from co-visibility artifacts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from depth_anything_3.bench.covisibility_subset import select_non_overlapping_indices  # noqa: E402


DEFAULT_THRESHOLDS = {"7scenes": 0.10, "scannetpp": 0.25, "eth3d": 0.025, "hiroom": 0.10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=REPO_ROOT / "artifacts/covisibility")
    parser.add_argument("--output", type=Path, required=True, help="JSONL manifest of selected original frame indices.")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_THRESHOLDS, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--threshold", type=float, help="Override all per-dataset thresholds.")
    parser.add_argument("--require-reference", action="store_true", help="Force candidate index zero into every scene subset.")
    args = parser.parse_args()
    records = []
    for dataset in args.datasets:
        threshold = DEFAULT_THRESHOLDS[dataset] if args.threshold is None else args.threshold
        for path in sorted((args.matrix_root / dataset).glob("*.npz")):
            with np.load(path) as data:
                local = select_non_overlapping_indices(data["max_directional_overlap"], threshold, required_index=0 if args.require_reference else None)
                records.append({
                    "dataset": dataset,
                    "scene": str(data["scene"].item()),
                    "threshold": threshold,
                    "candidate_indices": [int(value) for value in data["candidate_indices"]],
                    "selected_candidate_positions": local,
                    "selected_frame_indices": [int(data["candidate_indices"][index]) for index in local],
                    "selected_frame_ids": [str(data["frame_ids"][index]) for index in local],
                    "matrix_path": str(path),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Wrote {len(records)} scene selections to {args.output}")


if __name__ == "__main__":
    main()
