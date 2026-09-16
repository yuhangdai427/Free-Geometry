#!/usr/bin/env python3
"""Paired scene-macro and window-level bootstrap confidence intervals."""
import argparse
import json
from pathlib import Path

import numpy as np


def interval(samples):
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-metrics", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", default="base_4v")
    parser.add_argument("--comparisons", nargs="+", default=["base_8v", "base_8v_extract_4v", "lora_4v", "lora_8v"])
    parser.add_argument("--metric", default="auc03")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    data = json.loads(args.window_metrics.read_text())[args.dataset]
    scenes = sorted(data[args.baseline])
    rng = np.random.default_rng(args.seed)
    out = {"dataset": args.dataset, "metric": args.metric, "baseline": args.baseline, "scene_macro": {}, "window_paired": {}}
    for arm in args.comparisons:
        scene_diff = np.asarray([
            np.mean([x[args.metric] for x in data[arm][scene]]) -
            np.mean([x[args.metric] for x in data[args.baseline][scene]]) for scene in scenes
        ])
        scene_boot = [scene_diff[rng.integers(0, len(scene_diff), len(scene_diff))].mean() for _ in range(args.iterations)]
        all_diff = np.asarray([x[args.metric] - y[args.metric]
                               for scene in scenes for x, y in zip(data[arm][scene], data[args.baseline][scene])])
        window_boot = [all_diff[rng.integers(0, len(all_diff), len(all_diff))].mean() for _ in range(args.iterations)]
        out["scene_macro"][arm] = {"mean_delta": float(scene_diff.mean()), "ci95": interval(scene_boot)}
        out["window_paired"][arm] = {"mean_delta": float(all_diff.mean()), "ci95": interval(window_boot)}
    output = args.window_metrics.with_name("bootstrap_ci.json")
    output.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
