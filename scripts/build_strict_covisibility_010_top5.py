#!/usr/bin/env python3
"""Select up to five real 8V non-overlapping scenes per dataset.

This is intentionally an offline manifest builder.  It consumes the existing
GT depth/pose co-visibility matrices but neither changes a benchmark loader nor
provides GT to the model or its test-time adaptation loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resolve_matrix(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.resolve().parents[2] / path


def audit_record(record: dict, manifest: Path, threshold: float) -> dict:
    matrix_path = resolve_matrix(manifest, record["matrix_path"])
    with np.load(matrix_path, allow_pickle=False) as matrix:
        overlap = np.asarray(matrix["max_directional_overlap"], dtype=np.float32)
        frame_ids = [str(value) for value in matrix["frame_ids"]]
    positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    selected = [str(value) for value in record["selected_frame_ids"]]
    indices = [positions[frame_id] for frame_id in selected]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{record['dataset']}/{record['scene']}: duplicate selected frame")
    if len(indices) < 8:
        raise ValueError(f"{record['dataset']}/{record['scene']}: fewer than eight candidates")
    pairwise = overlap[np.ix_(indices, indices)]
    max_pair = float(pairwise[np.triu_indices(len(indices), k=1)].max())
    if max_pair > threshold + 1e-7:
        raise ValueError(
            f"{record['dataset']}/{record['scene']}: selected pool has overlap {max_pair:.6f} > {threshold}"
        )
    result = dict(record)
    result.update({
        "protocol": "strict_pairwise_nonoverlap_8v",
        "threshold": threshold,
        "candidate_pool_size": len(indices),
        "candidate_pool_max_pair_overlap": max_pair,
        "selection_note": "Eligible because the whole listed candidate pool contains at least 8 unique frames and every pair has max directional overlap <= threshold.",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/covisibility_all_datasets_0025/nonoverlap_010_counts.jsonl")
    parser.add_argument("--output", default="artifacts/covisibility_010_strict_top5/selected_scenes.jsonl")
    parser.add_argument("--report", default="artifacts/covisibility_010_strict_top5/selection_report.json")
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "scannetpp"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-scenes", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.input)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen, report = [], {"threshold": args.threshold, "datasets": {}}
    for dataset in args.datasets:
        eligible = [record for record in records if record["dataset"] == dataset and int(record["selected_count"]) >= 8]
        eligible.sort(key=lambda record: (-int(record["selected_count"]), str(record["scene"])))
        selected = [audit_record(record, source, args.threshold) for record in eligible[:args.max_scenes]]
        chosen.extend(selected)
        report["datasets"][dataset] = {
            "eligible_scene_count": len(eligible),
            "selected_scene_count": len(selected),
            "eligible_scenes": [{"scene": item["scene"], "candidate_pool_size": item["selected_count"]} for item in eligible],
            "selected_scenes": [{"scene": item["scene"], "candidate_pool_size": item["candidate_pool_size"], "max_pair_overlap": item["candidate_pool_max_pair_overlap"]} for item in selected],
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in chosen:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
