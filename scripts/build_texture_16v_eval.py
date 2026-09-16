#!/usr/bin/env python3
"""Build fixed textureless 16V (or all-real-frame) evaluation windows."""
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
    MATRIX_ROOTS, load_scenes, manifest_record, stable_seed, texture_raw_scores,
)


def candidate_windows(scene, count, seed):
    rng = np.random.default_rng(stable_seed(scene.dataset, scene.name, "texture-16v-eval", seed=seed))
    all_indices = list(range(len(scene.ids)))
    if len(all_indices) < 16:
        return [all_indices.copy() for _ in range(count)]
    return [rng.choice(all_indices, 16, replace=False).tolist() for _ in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(MATRIX_ROOTS))
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/hard_view_subsets")
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--samples-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.artifact_root / args.dataset / "texture"
    report = json.loads((root / "selection_report.json").read_text())
    patch_q30 = float(report["patch_gradient_q30"])
    frame_q4 = float(report["q4_threshold"])
    output = []
    for scene_name in report["top5_scenes"]:
        scene = load_scenes(args.dataset, need_geometry=False, selected=[scene_name])[0]
        patches, _ = texture_raw_scores(scene.images, args.device)
        scores = np.asarray([(row < patch_q30).mean() for row in patches], dtype=np.float32)
        windows = candidate_windows(scene, args.candidate_count, args.seed)
        eligible = [(float(scores[sequence].mean()), sequence) for sequence in windows
                    if np.any(scores[sequence] >= frame_q4)]
        if len(eligible) < args.samples_per_scene:
            raise RuntimeError(f"{args.dataset}/{scene_name}: only {len(eligible)} Q4 texture 16V windows")
        eligible.sort(key=lambda item: (-item[0], tuple(item[1])))
        for sample_idx, (score, sequence) in enumerate(eligible[:args.samples_per_scene]):
            hard_positions = [i for i, index in enumerate(sequence) if scores[index] >= frame_q4]
            record = manifest_record(scene, sequence, "eval_16v", sample_idx, None,
                                     "textureless_16v", hard_positions, score, False)
            for old, new in (("eight_local_indices", "sixteen_local_indices"),
                             ("eight_frame_indices", "sixteen_frame_indices"),
                             ("eight_frame_ids", "sixteen_frame_ids"),
                             ("eight_image_files", "sixteen_image_files")):
                record[new] = record.pop(old)
            record["texture_scores"] = [float(scores[index]) for index in sequence]
            output.append(record)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    destination = root / "eval_16v.jsonl"
    destination.write_text("".join(json.dumps(record) + "\n" for record in output))
    (root / "eval_16v_report.json").write_text(json.dumps({
        "dataset": args.dataset, "criterion": "textureless", "view_count": 16,
        "samples_per_scene": args.samples_per_scene, "candidate_count": args.candidate_count,
        "patch_gradient_q30": patch_q30, "frame_q4_threshold": frame_q4,
        "selection": "Each window contains at least one dataset-Q4 low-texture frame.",
    }, indent=2, sort_keys=True))
    print(f"[done] {destination}: {len(output)} textureless 16V windows")


if __name__ == "__main__":
    main()
