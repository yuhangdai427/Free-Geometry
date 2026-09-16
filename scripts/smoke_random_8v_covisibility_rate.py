#!/usr/bin/env python3
"""Estimate how often ordinary random 8V draws are strictly non-overlapping.

Uses the real precomputed MapAnything-style matrices.  A smoke draw chooses a
scene uniformly, then chooses eight distinct candidate frames uniformly within
that scene.  It is accepted only when every pair has max directional overlap at
most the supplied threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = z * np.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def load_matrices(directory: Path) -> list[tuple[str, np.ndarray]]:
    matrices = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            overlap = np.asarray(data["max_directional_overlap"], dtype=np.float32)
        if overlap.shape[0] >= 8:
            matrices.append((path.stem, overlap))
    if not matrices:
        raise ValueError(f"No usable matrices in {directory}")
    return matrices


def sample_dataset(matrices: list[tuple[str, np.ndarray]], draws: int, threshold: float, rng: np.random.Generator) -> dict:
    successes, maxima, examples = 0, [], []
    per_scene = {scene: {"draws": 0, "accepted": 0} for scene, _ in matrices}
    for draw_id in range(draws):
        scene, overlap = matrices[int(rng.integers(len(matrices)))]
        indices = rng.choice(overlap.shape[0], size=8, replace=False)
        max_pair = float(overlap[np.ix_(indices, indices)][np.triu_indices(8, k=1)].max())
        accepted = max_pair <= threshold
        successes += int(accepted)
        maxima.append(max_pair)
        per_scene[scene]["draws"] += 1
        per_scene[scene]["accepted"] += int(accepted)
        if accepted and len(examples) < 5:
            examples.append({"draw": draw_id, "scene": scene, "candidate_indices": [int(x) for x in indices], "max_pair_overlap": max_pair})
    return {
        "draws": draws,
        "strict_nonoverlap_count": successes,
        "strict_nonoverlap_rate": successes / draws,
        "wilson_95ci": wilson(successes, draws),
        "max_pair_overlap_quantiles": {str(q): float(np.quantile(maxima, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)},
        "accepted_examples": examples,
        "per_scene": per_scene,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth3d-dir", default="artifacts/covisibility_eth3d_all/eth3d")
    parser.add_argument("--scannetpp-dir", default="artifacts/covisibility_scannetpp_all_frames/scannetpp")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", default="artifacts/covisibility_010_strict_top5/random_8v_smoke_1000.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {
        "protocol": "uniform-scene then uniform-without-replacement eight-frame draw",
        "pairwise_rule": "max_directional_overlap <= threshold for every one of the 28 unordered pairs",
        "threshold": args.threshold,
        "seed": args.seed,
        "datasets": {
            "eth3d": sample_dataset(load_matrices(Path(args.eth3d_dir)), args.draws, args.threshold, rng),
            "scannetpp": sample_dataset(load_matrices(Path(args.scannetpp_dir)), args.draws, args.threshold, rng),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
