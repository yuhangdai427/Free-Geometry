#!/usr/bin/env python3
"""Compute scene-level weighted Free-Geo improvements with bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = [
    ("ETH3D", "eth3d"),
    ("ScanNet++", "scannetpp"),
    ("7-Scenes", "7scenes"),
    ("HiRoom", "hiroom"),
]

METRICS = [
    ("A03", "pose", "auc03", True),
    ("A30", "pose", "auc30", True),
    ("F1", "recon_unposed", "fscore", True),
    ("CD", "recon_unposed", "overall", False),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_path(metric_dir: Path, dataset: str, kind: str) -> Path:
    return metric_dir / f"{dataset}_{kind}.json"


def get_metric(value_by_kind: dict[str, dict], scene: str, kind: str, field: str) -> float:
    return float(value_by_kind[kind][scene][field])


def collect_matched_rows(workspace: Path, views: list[int]) -> list[dict]:
    configs = {
        "VGGT": {
            "base_root": workspace / "all_vggt",
            "free_root": workspace / "vggt_final_benchmark",
            "base_dir": lambda root, view, ds: root / f"base_{view}v" / ds,
            "free_dir": lambda root, view, ds: root / ds / f"{view}v",
        },
        "DA3": {
            "base_root": workspace / "all_da3",
            "free_root": workspace / "da3_final_benchmark",
            "base_dir": lambda root, view, ds: root / "baseline" / f"frames_{view}" / ds,
            "free_dir": lambda root, view, ds: root / ds / f"frames_{view}",
        },
    }

    rows: list[dict] = []
    for backbone, cfg in configs.items():
        for view in views:
            for dataset_label, dataset_dir_name in DATASETS:
                base_dir = cfg["base_dir"](cfg["base_root"], view, dataset_dir_name)
                free_dir = cfg["free_dir"](cfg["free_root"], view, dataset_dir_name)
                if not base_dir.exists() or not free_dir.exists():
                    continue

                seed_names = sorted(
                    {p.name for p in base_dir.glob("seed*") if p.is_dir()}
                    & {p.name for p in free_dir.glob("seed*") if p.is_dir()}
                )
                for seed_name in seed_names:
                    base_metric_dir = base_dir / seed_name / "metric_results"
                    free_metric_dir = free_dir / seed_name / "metric_results"
                    if not base_metric_dir.exists() or not free_metric_dir.exists():
                        continue

                    base_data = {
                        "pose": load_json(metric_path(base_metric_dir, dataset_dir_name, "pose")),
                        "recon_unposed": load_json(
                            metric_path(base_metric_dir, dataset_dir_name, "recon_unposed")
                        ),
                    }
                    free_data = {
                        "pose": load_json(metric_path(free_metric_dir, dataset_dir_name, "pose")),
                        "recon_unposed": load_json(
                            metric_path(free_metric_dir, dataset_dir_name, "recon_unposed")
                        ),
                    }

                    scenes = set(base_data["pose"])
                    scenes &= set(base_data["recon_unposed"])
                    scenes &= set(free_data["pose"])
                    scenes &= set(free_data["recon_unposed"])
                    scenes.discard("mean")

                    for scene in sorted(scenes):
                        row = {
                            "backbone": backbone,
                            "view": view,
                            "dataset": dataset_label,
                            "scene": scene,
                            "seed": seed_name,
                        }
                        for metric, kind, field, _higher_better in METRICS:
                            row[f"base_{metric}"] = get_metric(base_data, scene, kind, field)
                            row[f"free_{metric}"] = get_metric(free_data, scene, kind, field)
                        rows.append(row)
    return rows


def add_deltas(rows: list[dict], min_relative_baseline: float) -> None:
    for row in rows:
        for metric, _kind, _field, higher_better in METRICS:
            base = row[f"base_{metric}"]
            free = row[f"free_{metric}"]
            abs_delta = free - base if higher_better else base - free
            row[f"abs_delta_{metric}"] = abs_delta
            if base < min_relative_baseline:
                row[f"rel_delta_{metric}"] = None
                row[f"rel_valid_{metric}"] = False
            else:
                row[f"rel_delta_{metric}"] = abs_delta / base * 100.0
                row[f"rel_valid_{metric}"] = True


def percentile_ci(values_by_dataset: dict[str, list[float]], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    datasets = [d for d, vals in values_by_dataset.items() if vals]
    boot_means = np.empty(n_boot, dtype=np.float64)

    for boot_idx in range(n_boot):
        sampled_parts = []
        for dataset in datasets:
            vals = np.asarray(values_by_dataset[dataset], dtype=np.float64)
            sampled_parts.append(rng.choice(vals, size=len(vals), replace=True))
        boot_means[boot_idx] = np.concatenate(sampled_parts).mean()

    return (
        float(np.std(boot_means, ddof=1)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def ratio_of_means_ci(
    scenes_by_dataset: dict[str, list[dict]],
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    datasets = [d for d, scenes in scenes_by_dataset.items() if scenes]
    boot_values = np.empty(n_boot, dtype=np.float64)

    for boot_idx in range(n_boot):
        sampled_base = []
        sampled_delta = []
        for dataset in datasets:
            scenes = scenes_by_dataset[dataset]
            indices = rng.choice(len(scenes), size=len(scenes), replace=True)
            for idx in indices:
                sampled_base.append(scenes[idx][f"base_mean_{metric}"])
                sampled_delta.append(scenes[idx][f"abs_delta_{metric}"])
        boot_values[boot_idx] = np.mean(sampled_delta) / np.mean(sampled_base) * 100.0

    all_base = []
    all_delta = []
    for scenes in scenes_by_dataset.values():
        for scene in scenes:
            all_base.append(scene[f"base_mean_{metric}"])
            all_delta.append(scene[f"abs_delta_{metric}"])
    mean_improvement = float(np.mean(all_delta) / np.mean(all_base) * 100.0)
    return (
        mean_improvement,
        float(np.std(boot_values, ddof=1)),
        float(np.percentile(boot_values, 2.5)),
        float(np.percentile(boot_values, 97.5)),
    )


def summarize(
    rows: list[dict],
    n_boot: int,
    seed: int,
    summary_metric: str,
    agree_tolerance: float,
) -> tuple[list[dict], list[dict]]:
    scene_delta_rows: list[dict] = []
    summary_rows: list[dict] = []

    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["backbone"], row["view"], row["dataset"], row["scene"])].append(row)

    for (backbone, view, dataset, scene), group_rows in grouped.items():
        scene_row = {
            "backbone": backbone,
            "view": view,
            "dataset": dataset,
            "scene": scene,
        }
        for metric, _kind, _field, _higher_better in METRICS:
            valid_seed_rows = [r for r in group_rows if r[f"rel_valid_{metric}"]]
            rel_values = [r[f"rel_delta_{metric}"] for r in valid_seed_rows]
            base_values = [r[f"base_{metric}"] for r in group_rows]
            free_values = [r[f"free_{metric}"] for r in group_rows]
            abs_values = [r[f"abs_delta_{metric}"] for r in group_rows]
            scene_base_mean = float(np.mean(base_values))
            scene_free_mean = float(np.mean(free_values))
            scene_row[f"rel_delta_{metric}"] = (
                None
                if scene_base_mean < 0.01
                else (
                    (scene_free_mean - scene_base_mean) / scene_base_mean * 100.0
                    if metric != "CD"
                    else (scene_base_mean - scene_free_mean) / scene_base_mean * 100.0
                )
            )
            scene_row[f"abs_delta_{metric}"] = float(np.mean(abs_values))
            scene_row[f"base_mean_{metric}"] = scene_base_mean
            scene_row[f"free_mean_{metric}"] = scene_free_mean
            scene_row[f"n_valid_rel_seeds_{metric}"] = len(valid_seed_rows)
            scene_row[f"n_total_seeds_{metric}"] = len(group_rows)
            scene_row[f"n_zero_base_dropped_{metric}"] = len(group_rows) - len(valid_seed_rows)
        scene_delta_rows.append(scene_row)

    for backbone in sorted({r["backbone"] for r in rows}):
        for view in sorted({r["view"] for r in rows if r["backbone"] == backbone}):
            view_rows = [r for r in rows if r["backbone"] == backbone and r["view"] == view]
            view_scene_rows = [
                r for r in scene_delta_rows if r["backbone"] == backbone and r["view"] == view
            ]
            for metric, _kind, _field, _higher_better in METRICS:
                all_scene_abs_vals_by_dataset: dict[str, list[float]] = defaultdict(list)
                scenes_by_dataset: dict[str, list[dict]] = defaultdict(list)
                for scene_row in view_scene_rows:
                    abs_value = scene_row[f"abs_delta_{metric}"]
                    all_scene_abs_vals_by_dataset[scene_row["dataset"]].append(abs_value)
                    scenes_by_dataset[scene_row["dataset"]].append(scene_row)

                pooled_scene_abs_vals = [
                    value
                    for vals in all_scene_abs_vals_by_dataset.values()
                    for value in vals
                ]
                if not pooled_scene_abs_vals:
                    continue

                all_sample_vals = [row[f"abs_delta_{metric}"] for row in view_rows]
                dataset_means = {
                    dataset: float(np.mean(vals))
                    for dataset, vals in all_scene_abs_vals_by_dataset.items()
                    if vals
                }
                weighted_mean, boot_stderr, ci_low, ci_high = ratio_of_means_ci(
                    scenes_by_dataset, metric=metric, n_boot=n_boot, seed=seed
                )

                n_positive_scenes = int(
                    (np.asarray([v for vals in all_scene_abs_vals_by_dataset.values() for v in vals]) > 0).sum()
                )
                n_positive_sample_pairs = int((np.asarray(all_sample_vals) > 0).sum())
                n_agree_sample_pairs = sum(
                    1 for row in view_rows if row[f"abs_delta_{metric}"] >= -agree_tolerance
                )
                summary_rows.append(
                    {
                        "backbone": backbone,
                        "view": view,
                        "metric": metric,
                        "n_datasets": len(dataset_means),
                        "n_scenes": len([v for vals in all_scene_abs_vals_by_dataset.values() for v in vals]),
                        "n_rel_scenes": len([v for vals in all_scene_abs_vals_by_dataset.values() for v in vals]),
                        "n_sample_pairs": len(all_sample_vals),
                        "weighted_mean_improvement": weighted_mean,
                        "bootstrap_stderr": boot_stderr,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n_positive_scenes": n_positive_scenes,
                        "scene_AR_%": float(
                            n_positive_scenes
                            / len([v for vals in all_scene_abs_vals_by_dataset.values() for v in vals])
                            * 100.0
                        ),
                        "n_positive_sample_pairs": n_positive_sample_pairs,
                        "sample_AR_%": float(n_positive_sample_pairs / len(all_sample_vals) * 100.0),
                        "n_agree_sample_pairs": n_agree_sample_pairs,
                        "sample_agree_AR_%": float(n_agree_sample_pairs / len(view_rows) * 100.0),
                        "positive_datasets": f"{sum(v > 0 for v in dataset_means.values())}/{len(dataset_means)}",
                        "significant": "yes" if ci_low > 0 else "no",
                    }
                )

    return summary_rows, scene_delta_rows


def fmt_signed_pct(value: float) -> str:
    return f"{value:+.1f}%"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_abs(value: float) -> str:
    return f"{value:+.3f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Weighted Scene-Level Relative Improvement with 95% CI",
        "",
        "Improvement uses matched baseline/Free-Geo seed-scene pairs. Seeds are averaged within each scene, then all valid scenes are pooled across datasets. Datasets with more valid scenes therefore receive more weight. CI is a stratified bootstrap over scene deltas with dataset scene counts preserved.",
        "",
        "Relative improvement is computed as the ratio of pooled scene means: first average seeds inside each scene, then average those scene means across datasets with scene-count weighting, then take `(free_mean - base_mean) / base_mean`. All matched scenes contribute to this statistic.",
        "",
        "`Sample AR` is strict improvement on matched `(seed, scene)` samples. `Sample Agree AR` counts a matched sample as agreement if the absolute directional delta is at least `-agree_tolerance`, so tiny degradations are accepted.",
        "",
        "| Backbone | View | Metric | N datasets | N scenes | N rel scenes | N sample pairs | Improvement | Bootstrap stderr | 95% CI | Scene AR | Sample AR | Sample Agree AR | Positive datasets | Significant |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ci = f"[{fmt_signed_pct(row['ci95_low'])}, {fmt_signed_pct(row['ci95_high'])}]"
        lines.append(
            "| {backbone} | {view} | {metric} | {n_datasets} | {n_scenes} | "
            "{n_rel_scenes} | {n_sample_pairs} | {improvement} | {stderr} | "
            "{ci} | {scene_ar} | {seed_ar} | {sample_agree_ar} | {positive_datasets} | {significant} |".format(
                backbone=row["backbone"],
                view=row["view"],
                metric=row["metric"],
                n_datasets=row["n_datasets"],
                n_scenes=row["n_scenes"],
                n_rel_scenes=row["n_rel_scenes"],
                n_sample_pairs=row["n_sample_pairs"],
                improvement=fmt_signed_pct(row["weighted_mean_improvement"]),
                stderr=fmt_pct(row["bootstrap_stderr"]),
                ci=ci,
                scene_ar=fmt_pct(row["scene_AR_%"]),
                seed_ar=fmt_pct(row["sample_AR_%"]),
                sample_agree_ar=fmt_pct(row["sample_agree_AR_%"]),
                positive_datasets=row["positive_datasets"],
                significant=row["significant"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/root/autodl-tmp/da3/workspace"))
    parser.add_argument("--views", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_relative_baseline", type=float, default=0.01)
    parser.add_argument("--agree_tolerance", type=float, default=0.001)
    parser.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("/root/autodl-tmp/da3/workspace/weighted_scene_level_rel_improvement_ci_4v_32v"),
    )
    args = parser.parse_args()

    matched_rows = collect_matched_rows(args.workspace, args.views)
    add_deltas(matched_rows, args.min_relative_baseline)
    summary_rows, scene_delta_rows = summarize(
        matched_rows,
        args.n_boot,
        args.seed,
        summary_metric="relative",
        agree_tolerance=args.agree_tolerance,
    )

    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_matched_seed_scene.csv"), matched_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_scene_deltas.csv"), scene_delta_rows)
    write_csv(args.out_prefix.with_suffix(".csv"), summary_rows)
    write_markdown(args.out_prefix.with_suffix(".md"), summary_rows)

    print(f"Wrote {args.out_prefix.with_suffix('.md')}")
    print(f"Wrote {args.out_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.out_prefix.with_name(args.out_prefix.name + '_matched_seed_scene.csv')}")
    print(f"Wrote {args.out_prefix.with_name(args.out_prefix.name + '_scene_deltas.csv')}")


if __name__ == "__main__":
    main()
