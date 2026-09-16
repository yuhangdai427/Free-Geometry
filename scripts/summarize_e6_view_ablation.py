#!/usr/bin/env python3
"""Summarize E6 metrics and telemetry across datasets, train seeds, eval seeds, and views."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional


CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
METRIC_HINTS = {
    "pose_auc": ("auc03", "auc_03", "auc3", "auc_3"),
    "chamfer_distance": ("chamfer", "chamfer_distance", "cd", "overall"),
    "f1": ("f1", "fscore", "f_score", "f-score"),
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_metric(entry: dict, hints: Iterable[str]) -> Optional[float]:
    lower = {str(key).lower(): value for key, value in entry.items()}
    for hint in hints:
        if hint in lower and isinstance(lower[hint], (int, float)):
            return float(lower[hint])
    return None


def scene_metrics(metrics_dir: Path, dataset: str) -> Dict[str, Dict[str, Optional[float]]]:
    pose = load_json(metrics_dir / f"{dataset}_pose.json")
    recon = load_json(metrics_dir / f"{dataset}_recon_unposed.json")
    rows = {}
    for scene in sorted((set(pose) | set(recon)) - {"mean"}):
        rows[scene] = {
            "pose_auc": find_metric(pose.get(scene, {}), METRIC_HINTS["pose_auc"]),
            "chamfer_distance": find_metric(recon.get(scene, {}), METRIC_HINTS["chamfer_distance"]),
            "f1": find_metric(recon.get(scene, {}), METRIC_HINTS["f1"]),
        }
    return rows


def stat(values: List[float]) -> Optional[Dict[str, float]]:
    return None if not values else {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed E6 runs")
    parser.add_argument("--checkpoints_root", default="checkpoints/e6_nested_views")
    parser.add_argument("--results_root", default="results/e6_nested_views")
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "7scenes"])
    parser.add_argument("--views", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    checkpoints = Path(args.checkpoints_root)
    results = Path(args.results_root)
    output = Path(args.output_dir) if args.output_dir else results / "summary"
    output.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    summary = {"metrics": {}, "telemetry": {}}

    for dataset in args.datasets:
        summary["metrics"][dataset] = {}
        summary["telemetry"][dataset] = {}
        for config in CONFIGS:
            telemetry = [load_json(path) for path in sorted((checkpoints / dataset / config).glob("seed*/telemetry.json"))]
            summary["telemetry"][dataset][config] = {
                "teacher_forward_peak_gib": stat([item.get("teacher_forward_peak_bytes", 0) / 2**30 for item in telemetry]),
                "training_peak_gib": stat([item.get("training_peak_bytes", 0) / 2**30 for item in telemetry]),
                "total_wall_clock_minutes": stat([item.get("total_wall_clock_seconds", 0) / 60 for item in telemetry]),
                "epoch_minutes": stat([epoch.get("wall_clock_seconds", 0) / 60 for item in telemetry for epoch in item.get("epochs", [])]),
            }
            summary["metrics"][dataset][config] = {}
            for views in args.views:
                values = {name: [] for name in METRIC_HINTS}
                scene_keys = set()
                pattern = f"train_seed*/eval_seed*/{views}v/{views}v/metric_results"
                for metric_dir in sorted((results / dataset / config).glob(pattern)):
                    parts = metric_dir.parts
                    train_seed = next(part.removeprefix("train_seed") for part in parts if part.startswith("train_seed"))
                    eval_seed = next(part.removeprefix("eval_seed") for part in parts if part.startswith("eval_seed"))
                    for scene, row in scene_metrics(metric_dir, dataset).items():
                        scene_keys.add((train_seed, eval_seed, scene))
                        paired_rows.append({"dataset": dataset, "config": config, "train_seed": int(train_seed), "eval_seed": int(eval_seed), "views": views, "scene": scene, **row})
                        for name, value in row.items():
                            if value is not None:
                                values[name].append(value)
                summary["metrics"][dataset][config][f"{views}v"] = {
                    "paired_scene_count": len(scene_keys),
                    **{name: stat(metric_values) for name, metric_values in values.items()},
                }

    (output / "e6_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "e6_paired_scenes.json").write_text(json.dumps(paired_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output / 'e6_summary.json'}")


if __name__ == "__main__":
    main()
