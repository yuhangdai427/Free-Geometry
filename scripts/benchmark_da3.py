#!/usr/bin/env python3
"""
Benchmark script for a Free-Geometry adapted DepthAnything3 model.

Usage (legacy-format checkpoints; new runs use fg evaluate):
    python scripts/benchmark_da3.py \
        --lora_path checkpoints/tta/best_lora.pt \
        --datasets scannetpp \
        --modes pose recon_unposed \
        --max_frames 4
"""

import argparse
import json
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.test_time_adaption.models import StudentModel, DA3StudentFinetune


class LoRADepthAnything3:
    """
    Wrapper to make a Free-Geometry StudentModel compatible with the Evaluator API.

    The Evaluator expects an object with an `inference` method matching
    the DepthAnything3 API signature.
    """

    def __init__(
        self,
        base_model: str = "depth-anything/DA3-GIANT",
        lora_path: str = None,
        lora_rank: int = None,
        lora_alpha: float = None,
        device: str = "cuda",
    ):
        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        # Auto-detect rank/alpha from saved PEFT adapter_config.json
        peft_dir = lora_path.replace('.pt', '_peft') if lora_path else None
        if peft_dir and os.path.isdir(peft_dir):
            config_path = os.path.join(peft_dir, 'adapter_config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    peft_cfg = json.load(f)
                saved_rank = peft_cfg.get('r')
                saved_alpha = peft_cfg.get('lora_alpha')
                if lora_rank is not None and lora_rank != saved_rank:
                    print(f"WARNING: --lora_rank={lora_rank} but checkpoint has r={saved_rank}. Using checkpoint value.")
                if lora_alpha is not None and lora_alpha != saved_alpha:
                    print(f"WARNING: --lora_alpha={lora_alpha} but checkpoint has alpha={saved_alpha}. Using checkpoint value.")
                lora_rank = saved_rank
                lora_alpha = saved_alpha
                print(f"Auto-detected from checkpoint: rank={lora_rank}, alpha={lora_alpha}")

        if lora_rank is None:
            lora_rank = 16
        if lora_alpha is None:
            lora_alpha = float(lora_rank)

        # Create student model with LoRA
        self.student = StudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            # For inference we merge LoRA into base weights; keep FFN fully fused for speed.
            patch_swiglu_mlp_for_lora=False,
        )

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        # Merge LoRA into base weights so fused SwiGLU reads the merged weights directly
        peft_model = self.student.da3.model.backbone.pretrained
        self.student.da3.model.backbone.pretrained = peft_model.merge_and_unload()
        print("Merged LoRA weights into base model for inference")

        self.student.eval()
        self.device = device

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.student = self.student.to(device)
        return self

    def inference(
        self,
        image,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        export_dir=None,
        export_format="mini_npz",
        ref_view_strategy="first",
        **kwargs,
    ):
        """
        Run inference using the LoRA-adapted model.

        This wraps the underlying DA3 API's inference method.
        """
        # Use the underlying DA3 model's inference
        return self.student.da3.inference(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            export_dir=export_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            **kwargs,
        )


class FinetuneDepthAnything3:
    """
    Wrapper to make a Free-Geometry DA3StudentFinetune model compatible with the Evaluator API.
    """

    def __init__(
        self,
        base_model: str = "depth-anything/DA3-GIANT",
        weights_path: str = None,
        device: str = "cuda",
    ):
        print(f"Loading base model (finetune): {base_model}")
        print(f"Loading finetune weights: {weights_path}")

        self.student = DA3StudentFinetune(
            model_name=base_model,
        )

        if weights_path and os.path.exists(weights_path):
            self.student.load_finetune_weights(weights_path)
        else:
            raise FileNotFoundError(f"Finetune weights not found: {weights_path}")

        self.student.eval()
        self.device = device

    def to(self, device):
        self.device = device
        self.student = self.student.to(device)
        return self

    def inference(
        self,
        image,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        export_dir=None,
        export_format="mini_npz",
        ref_view_strategy="first",
        **kwargs,
    ):
        return self.student.da3.inference(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            export_dir=export_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA-finetuned DA3 model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/tta/best_lora.pt",
        help="Path to Free-Geometry adapter weights",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="depth-anything/DA3-GIANT",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Adapter rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="Adapter alpha (must match training)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Load finetune weights instead of LoRA",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./workspace/evaluation_lora",
        help="Output directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eth3d"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["pose", "recon_unposed"],
        help="Evaluation modes",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Max frames per scene",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to evaluate",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (skip inference)",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print saved metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling (used if --seeds not set)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multiple seeds for frame sampling. Runs once per seed, aggregates results.",
    )
    args = parser.parse_args()

    # Determine seed list
    seeds = args.seeds if args.seeds else [args.seed]

    # Load model once (shared across seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = None

    if not args.print_only and not args.eval_only:
        if args.finetune:
            api = FinetuneDepthAnything3(
                base_model=args.base_model,
                weights_path=args.lora_path,
            ).to(device)
        else:
            api = LoRADepthAnything3(
                base_model=args.base_model,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
            ).to(device)

    if len(seeds) == 1:
        # Single seed: original behavior
        evaluator = Evaluator(
            work_dir=args.work_dir,
            datas=args.datasets,
            modes=args.modes,
            max_frames=args.max_frames,
            scenes=args.scenes,
            seed=seeds[0],
        )

        if args.print_only:
            evaluator.print_metrics()
            return
        if args.eval_only:
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
            # Save metrics for multi-seed aggregation
            _save_metrics(metrics, args.work_dir)
            return

        evaluator.infer(api)
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
        # Save metrics for multi-seed aggregation
        _save_metrics(metrics, args.work_dir)
    else:
        # Multi-seed: run inference sequentially, then eval in parallel
        if not args.eval_only and not args.print_only:
            _run_multi_seed_inference(args, seeds, api)
            # Free model before launching parallel eval subprocesses
            del api
            api = None
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            print("Model freed from GPU/CPU memory")

        _run_multi_seed_eval(args, seeds)


def _save_metrics(metrics, work_dir):
    """Save metrics to JSON for multi-seed aggregation."""
    metrics_file = os.path.join(work_dir, "metrics.json")
    os.makedirs(work_dir, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")


def _run_multi_seed_inference(args, seeds, api):
    """Run inference sequentially across all seeds (GPU-bound)."""
    for seed in seeds:
        seed_work_dir = os.path.join(args.work_dir, f"seed{seed}")
        print(f"\n{'='*60}")
        print(f"[Inference] seed {seed}")
        print(f"{'='*60}")

        evaluator = Evaluator(
            work_dir=seed_work_dir,
            datas=args.datasets,
            modes=args.modes,
            max_frames=args.max_frames,
            scenes=args.scenes,
            seed=seed,
        )
        evaluator.infer(api)


def _run_multi_seed_eval(args, seeds):
    """Run eval in sequential subprocesses across all seeds.

    Each eval subprocess uses ~25-30GB RAM. Running them sequentially ensures
    we stay within the cgroup memory limit (110GB) and avoid OOM kills.
    """
    import numpy as np
    import subprocess

    all_seed_metrics = {}

    print(f"\n{'='*60}")
    print(f"Running eval for {len(seeds)} seeds sequentially (subprocess isolation)")
    print(f"{'='*60}")

    for seed in seeds:
        seed_work_dir = os.path.join(args.work_dir, f"seed{seed}")

        cmd = [
            sys.executable,
            __file__,
            "--lora_path", args.lora_path,
            "--base_model", args.base_model,
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--work_dir", seed_work_dir,
            "--datasets", *args.datasets,
            "--modes", *args.modes,
            "--max_frames", str(args.max_frames),
            "--seed", str(seed),
            "--eval_only",
        ]

        if args.finetune:
            cmd.append("--finetune")
        if args.scenes:
            cmd.extend(["--scenes", *args.scenes])

        log_file = os.path.join(args.work_dir, f"seed{seed}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        print(f"\n  [Eval] seed {seed} (log: {log_file})")
        with open(log_file, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            proc.wait()

        rc = proc.returncode
        if rc != 0:
            print(f"  WARNING: seed {seed} eval exited with code {rc}. See {log_file}")
        else:
            print(f"  Seed {seed} eval complete")

        seed_work_dir = os.path.join(args.work_dir, f"seed{seed}")
        metrics_file = os.path.join(seed_work_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                all_seed_metrics[seed] = json.load(f)
        else:
            print(f"  WARNING: No metrics file found for seed {seed}")
            all_seed_metrics[seed] = {}

    print(f"\nAll {len(seeds)} seeds completed. Aggregating results...")

    # Print aggregated summary
    _print_multi_seed_summary(all_seed_metrics, seeds, args)


def _print_multi_seed_summary(all_seed_metrics, seeds, args):
    """Print combined multi-seed metrics summary."""
    import numpy as np

    all_keys = set()
    for seed_metrics in all_seed_metrics.values():
        all_keys.update(seed_metrics.keys())

    for metric_key in sorted(all_keys):
        print(f"\n{'='*60}")
        print(f"  {metric_key} — Multi-seed summary")
        print(f"{'='*60}")

        all_scenes = set()
        for seed in seeds:
            if metric_key in all_seed_metrics[seed]:
                for scene in all_seed_metrics[seed][metric_key]:
                    if scene != "mean":
                        all_scenes.add(scene)
        all_scenes = sorted(all_scenes)

        if not all_scenes:
            continue

        metric_names = None
        for seed in seeds:
            if metric_key in all_seed_metrics[seed] and all_scenes[0] in all_seed_metrics[seed][metric_key]:
                metric_names = list(all_seed_metrics[seed][metric_key][all_scenes[0]].keys())
                break
        if not metric_names:
            continue

        header = f"{'scene':<30s}"
        for name in metric_names:
            header += f"  {name:>10s}"
        print(header)
        print("-" * len(header))

        seed_means = {seed: {name: [] for name in metric_names} for seed in seeds}

        for scene in all_scenes:
            for seed in seeds:
                if metric_key not in all_seed_metrics[seed]:
                    continue
                scene_data = all_seed_metrics[seed][metric_key].get(scene)
                if scene_data is None:
                    continue

                row_label = f"{scene}/seed{seed}"
                row = f"{row_label:<30s}"
                for name in metric_names:
                    val = scene_data.get(name, float('nan'))
                    row += f"  {val:>10.4f}"
                    seed_means[seed][name].append(val)
                print(row)

        print("-" * len(header))
        for seed in seeds:
            row = f"{'mean/seed' + str(seed):<30s}"
            for name in metric_names:
                vals = seed_means[seed][name]
                mean_val = np.mean(vals) if vals else float('nan')
                row += f"  {mean_val:>10.4f}"
            print(row)

        print("-" * len(header))
        row = f"{'mean/overall':<30s}"
        for name in metric_names:
            all_vals = []
            for seed in seeds:
                all_vals.extend(seed_means[seed][name])
            mean_val = np.mean(all_vals) if all_vals else float('nan')
            row += f"  {mean_val:>10.4f}"
        print(row)

    # Save aggregated JSON
    agg_path = os.path.join(args.work_dir, "multi_seed_summary.json")
    agg = {}
    for metric_key in sorted(all_keys):
        agg[metric_key] = {}
        for seed in seeds:
            if metric_key in all_seed_metrics[seed]:
                for scene, vals in all_seed_metrics[seed][metric_key].items():
                    agg[metric_key][f"{scene}/seed{seed}"] = vals
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved multi-seed summary to: {agg_path}")


if __name__ == "__main__":
    main()
