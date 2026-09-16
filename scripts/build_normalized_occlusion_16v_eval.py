#!/usr/bin/env python3
"""Build held-out, real normalized-occlusion 16V evaluation windows.

The 16V windows are scored from GT depth and poses under the same normalized
occlusion rule as the 8V subsets.  They are a new sequence type: no exact 16V
ordered sequence can appear in the 8V training manifest.  Frame reuse across
splits is unavoidable for small ETH3D scenes and is deliberately not falsely
called a disjoint-frame holdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "vggt")]
from scripts.build_hard_view_subsets import (  # noqa: E402
    MATRIX_ROOTS, load_scenes, manifest_record, normalized_occlusion_scores,
    stable_seed,
)


def random_16(scene, count: int, seed: int):
    rng = np.random.default_rng(stable_seed(scene.dataset, scene.name, "normalized-occlusion-16v-eval", seed=seed))
    available = list(range(len(scene.ids)))
    rows = []
    for _ in range(count):
        if len(available) >= 16:
            rows.append(rng.choice(available, 16, replace=False).tolist())
        else:
            # A short scene has no valid distinct 16V context. Evaluate all of
            # its real frames once instead of inventing duplicate observations.
            rows.append(available.copy())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(MATRIX_ROOTS))
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/hard_view_subsets")
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--samples-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.artifact_root / args.dataset / "occlusion"
    report = json.loads((root / "selection_report.json").read_text())
    threshold = float(report["q4_threshold"])
    output = []
    for scene_name in report["top5_scenes"]:
        scene = load_scenes(args.dataset, need_geometry=True, selected=[scene_name])[0]
        candidates = random_16(scene, args.candidate_count, args.seed)
        scores = np.asarray(normalized_occlusion_scores(scene, candidates, args.device))
        # The entire 16V context is evaluated; require at least one Q4 view and
        # rank by mean 16-view normalized occlusion.
        eligible = [(float(row.mean()), sequence, row) for sequence, row in zip(candidates, scores)
                    if np.any(row >= threshold)]
        if len(eligible) < args.samples_per_scene:
            raise RuntimeError(f"{args.dataset}/{scene_name}: only {len(eligible)} held-out Q4 16V windows")
        eligible.sort(key=lambda item: (-item[0], tuple(item[1])))
        for sample_idx, (score, sequence, row) in enumerate(eligible[:args.samples_per_scene]):
            hard_positions = np.flatnonzero(row >= threshold).astype(int).tolist()
            record = manifest_record(scene, sequence, "eval_16v", sample_idx, None,
                                     "normalized_occlusion_16v", hard_positions, score, False)
            # Keep explicit keys: benchmark code must not accidentally treat this
            # window as an eight-view sequence.
            record["sixteen_local_indices"] = record.pop("eight_local_indices")
            record["sixteen_frame_indices"] = record.pop("eight_frame_indices")
            record["sixteen_frame_ids"] = record.pop("eight_frame_ids")
            record["sixteen_image_files"] = record.pop("eight_image_files")
            record["normalized_occlusion_scores"] = [float(value) for value in row]
            output.append(record)
        del scene.depths, scene.valids
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    destination = root / "eval_16v.jsonl"
    destination.write_text("".join(json.dumps(record) + "\n" for record in output))
    metadata = {
        "dataset": args.dataset, "view_count": 16, "samples_per_scene": args.samples_per_scene,
        "candidate_count": args.candidate_count, "q4_threshold_from_8v": threshold,
        "held_out_rule": "16V sequences are a separate ordered sequence type and are absent from the 8V training manifest; frame reuse is permitted.",
        "occlusion_definition": report["frame_denominator"],
    }
    (root / "eval_16v_report.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[done] {destination}: {len(output)} held-out normalized-occlusion 16V windows")


if __name__ == "__main__":
    main()
