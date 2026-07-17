#!/usr/bin/env python3
"""Compare baseline VGGT with VGGT-Test3R on identical benchmark samples."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(LOCAL_HF_HOME / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_vggt import BaseVGGT
from scripts.run_vggt_test3r import AdaptedVGGTEvaluator, VGGTTest3RAPI, parse_int_or_none
from vggt.bench.evaluator import VGGTEvaluator


def run_baseline(args, seed: int, n_views: int, device: torch.device):
    work_dir = os.path.join(args.work_dir, "baseline", f"seed{seed}_{n_views}v")
    evaluator = VGGTEvaluator(
        work_dir=work_dir,
        datas=args.datasets,
        modes=args.modes,
        scenes=args.scenes,
        max_frames=n_views,
        image_size=args.image_size,
        seed=seed,
        debug=args.debug,
        num_fusion_workers=args.num_fusion_workers,
    )
    if not args.eval_only:
        api = BaseVGGT(
            model_name=args.base_model,
            image_size=args.image_size,
        ).to(device)
        evaluator.infer(api)
        del api
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)
    return metrics


def run_test3r(args, seed: int, n_views: int, device: torch.device):
    work_dir = os.path.join(args.work_dir, "test3r", f"seed{seed}_{n_views}v")
    evaluator = AdaptedVGGTEvaluator(
        work_dir=work_dir,
        datas=args.datasets,
        modes=args.modes,
        scenes=args.scenes,
        max_frames=n_views,
        image_size=args.image_size,
        seed=seed,
        debug=args.debug,
        num_fusion_workers=args.num_fusion_workers,
    )
    if not args.eval_only:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        api = VGGTTest3RAPI(
            model_name=args.base_model,
            prompt_size=args.prompt_size,
            device=device,
            image_size=args.image_size,
            epochs=args.epochs,
            lr=args.lr,
            accum_iter=args.accum_iter,
            max_triplets=args.max_triplets,
            remove_degenerate_triplets=args.remove_degenerate_triplets,
            seed=seed,
            prompt_dir=os.path.join(work_dir, "prompts"),
            use_amp=not args.no_amp,
        )
        evaluator.infer(api)
        del api
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)
    return metrics


def mean_metrics(metrics: dict) -> dict:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and isinstance(value.get("mean"), dict):
            flat[key] = value["mean"]
    return flat


def delta_metrics(test3r: dict, baseline: dict) -> dict:
    out = {}
    for group, test_group in test3r.items():
        base_group = baseline.get(group, {})
        group_delta = {}
        for metric, test_value in test_group.items():
            if isinstance(test_value, (int, float)) and isinstance(base_group.get(metric), (int, float)):
                group_delta[metric] = float(test_value) - float(base_group[metric])
        if group_delta:
            out[group] = group_delta
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline VGGT vs VGGT-Test3R comparison")
    parser.add_argument("--base_model", default="facebook/vggt-1b")
    parser.add_argument("--work_dir", default="./workspace/vggt_test3r_comparison")
    parser.add_argument("--datasets", nargs="+", default=["7scenes"])
    parser.add_argument("--modes", nargs="+", default=["pose", "recon_unposed"])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "test3r"], default=["baseline", "test3r"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--view_counts", nargs="+", type=int, default=[4])
    parser.add_argument("--image_size", type=int, default=504)
    parser.add_argument("--prompt_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum_iter", type=int, default=2)
    parser.add_argument("--max_triplets", type=parse_int_or_none, default=None)
    parser.add_argument("--remove_degenerate_triplets", action="store_true")
    parser.add_argument("--num_fusion_workers", type=int, default=1)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)

    summary = {
        "settings": {
            "base_model": args.base_model,
            "datasets": args.datasets,
            "modes": args.modes,
            "scenes": args.scenes,
            "seeds": args.seeds,
            "view_counts": args.view_counts,
            "image_size": args.image_size,
            "test3r": {
                "prompt_size": args.prompt_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "accum_iter": args.accum_iter,
                "max_triplets": args.max_triplets,
                "remove_degenerate_triplets": args.remove_degenerate_triplets,
            },
        },
        "runs": {},
    }

    for seed in args.seeds:
        for n_views in args.view_counts:
            run_key = f"seed{seed}_{n_views}v"
            print(f"\n[COMPARE] {run_key}")
            run_result = {}
            if "baseline" in args.methods:
                print("\n[COMPARE] Running baseline VGGT")
                baseline = run_baseline(args, seed, n_views, device)
                run_result["baseline"] = baseline
            if "test3r" in args.methods:
                print("\n[COMPARE] Running VGGT-Test3R")
                test3r = run_test3r(args, seed, n_views, device)
                run_result["test3r"] = test3r

            if "baseline" in run_result and "test3r" in run_result:
                run_result["mean"] = {
                    "baseline": mean_metrics(run_result["baseline"]),
                    "test3r": mean_metrics(run_result["test3r"]),
                }
                run_result["delta_test3r_minus_baseline"] = delta_metrics(
                    run_result["mean"]["test3r"],
                    run_result["mean"]["baseline"],
                )
            summary["runs"][run_key] = run_result

    summary_path = os.path.join(args.work_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison summary to: {summary_path}")


if __name__ == "__main__":
    main()
