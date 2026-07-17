#!/usr/bin/env python3
"""Plot DA3 leave-one-dataset-out benchmark metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _metric_values(metrics: dict[str, Any], key: str) -> dict[str, dict[str, float]]:
    data = metrics.get(key, {})
    if not isinstance(data, dict):
        return {}
    return {
        scene: values
        for scene, values in data.items()
        if scene != "mean" and isinstance(values, dict)
    }


def _mean_values(metrics: dict[str, Any], key: str) -> dict[str, float]:
    values = metrics.get(key, {}).get("mean", {})
    return values if isinstance(values, dict) else {}


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError(
            "matplotlib is required to render figures. Install it or run training/benchmark only."
        ) from exc
    return plt


def plot_pose_auc(metrics: dict[str, Any], dataset: str, out_dir: Path) -> Path | None:
    scenes = _metric_values(metrics, f"{dataset}_pose")
    if not scenes:
        return None

    plt = _safe_import_matplotlib()
    auc_names = ["auc30", "auc15", "auc05", "auc03"]
    scene_names = list(scenes)
    x = range(len(scene_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#1f6f8b", "#3f8f7d", "#e3a32f", "#b44b3a"]
    for offset, (metric, color) in enumerate(zip(auc_names, colors)):
        positions = [i + (offset - 1.5) * width for i in x]
        ax.bar(
            positions,
            [scenes[scene].get(metric, float("nan")) for scene in scene_names],
            width,
            label=metric,
            color=color,
        )

    mean_vals = _mean_values(metrics, f"{dataset}_pose")
    title_bits = [f"{metric}={mean_vals[metric]:.3f}" for metric in auc_names if metric in mean_vals]
    ax.set_title(f"DA3 LODO Cross-Domain Pose on {dataset}" + (f" | mean: {', '.join(title_bits)}" if title_bits else ""))
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(scene_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()

    out_path = out_dir / f"{dataset}_pose_auc.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_recon_metrics(metrics: dict[str, Any], dataset: str, out_dir: Path) -> Path | None:
    scenes = _metric_values(metrics, f"{dataset}_recon_unposed")
    if not scenes:
        return None

    plt = _safe_import_matplotlib()
    scene_names = list(scenes)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = range(len(scene_names))

    error_metrics = ["acc", "comp", "overall"]
    score_metrics = ["precision", "recall", "fscore"]
    error_colors = ["#7a3e1d", "#bb6b2c", "#d9a441"]
    score_colors = ["#1f6f8b", "#3f8f7d", "#8fb339"]

    for metric, color in zip(error_metrics, error_colors):
        axes[0].plot(
            list(x),
            [scenes[scene].get(metric, float("nan")) for scene in scene_names],
            marker="o",
            linewidth=2,
            label=metric,
            color=color,
        )
    axes[0].set_title(f"DA3 LODO Cross-Domain Reconstruction on {dataset} | lower error is better")
    axes[0].set_ylabel("Distance")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, frameon=False)

    for metric, color in zip(score_metrics, score_colors):
        axes[1].plot(
            list(x),
            [scenes[scene].get(metric, float("nan")) for scene in scene_names],
            marker="o",
            linewidth=2,
            label=metric,
            color=color,
        )
    axes[1].set_title("Higher score is better")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(scene_names, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncol=3, frameon=False)

    mean_vals = _mean_values(metrics, f"{dataset}_recon_unposed")
    if mean_vals:
        mean_text = " | ".join(f"{k}={v:.3f}" for k, v in mean_vals.items())
        fig.text(0.01, 0.01, f"Mean: {mean_text}", fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = out_dir / f"{dataset}_recon_unposed.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_summary(metrics: dict[str, Any], dataset: str, out_dir: Path) -> Path:
    summary_path = out_dir / f"{dataset}_summary.md"
    lines = [f"# DA3 LODO 7Scenes Summary", ""]
    for key in (f"{dataset}_pose", f"{dataset}_recon_unposed"):
        mean_vals = _mean_values(metrics, key)
        if not mean_vals:
            continue
        lines.extend([f"## {key}", ""])
        for metric, value in mean_vals.items():
            lines.append(f"- {metric}: {value:.6f}")
        lines.append("")
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path, help="Path to benchmark metrics.json")
    parser.add_argument("--dataset", default="7scenes", help="Dataset name in metrics keys")
    parser.add_argument("--out_dir", required=True, type=Path, help="Directory for generated figures")
    args = parser.parse_args()

    metrics = _load_json(args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_pose_auc(metrics, args.dataset, args.out_dir),
        plot_recon_metrics(metrics, args.dataset, args.out_dir),
        write_summary(metrics, args.dataset, args.out_dir),
    ]

    print("Generated outputs:")
    for path in outputs:
        if path is not None:
            print(f"  {os.fspath(path)}")


if __name__ == "__main__":
    main()
