#!/usr/bin/env python3
"""
Benchmark script for a Free-Geometry LoRA-tuned VGGT model.

Usage:
    python scripts/benchmark_vggt.py \
        --lora_path checkpoints/vggt_tta/epoch_1_lora.pt \
        --datasets scannetpp \
        --modes pose recon_unposed \
        --max_frames 8
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add vggt submodule to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vggt'))

from vggt.bench.evaluator import VGGTEvaluator
from vggt.test_time_adaption.models import VGGTStudentModel
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class LoRAVGGT:
    """
    Wrapper to make a Free-Geometry VGGTStudentModel compatible with the VGGTEvaluator API.

    The Evaluator expects an object with an `inference` method that takes
    image files and returns predictions.
    """

    def __init__(
        self,
        base_model: str = "facebook/vggt-1b",
        lora_path: str = None,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_layers_start: int = 12,
        device: str = "cuda",
        image_size: int = 504,
    ):
        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        self.image_size = image_size
        self.device = device

        # Create student model with LoRA
        lora_layers = list(range(lora_layers_start, 24))
        self.student = VGGTStudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
        )

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        elif lora_path:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        self.student.eval()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.student = self.student.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images using Free-Geometry preprocessing.

        - Resize longest side to image_size (default 504)
        - Preserve aspect ratio
        - Round dimensions to nearest multiple of PATCH_SIZE (14)
        """
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]

            # Resize longest side to image_size (Free-Geometry upper_bound_resize)
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Round to nearest multiple of PATCH_SIZE
            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            # Use appropriate interpolation
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        # All images in a scene have the same original size, so stack directly
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)  # Add batch dim: [1, S, 3, H, W]

    def inference(
        self,
        image_files,
        extrinsics=None,
        intrinsics=None,
        **kwargs,
    ):
        """
        Run inference using the LoRA-adapted VGGT model.

        Args:
            image_files: List of image file paths
            extrinsics: Optional GT extrinsics (for posed mode)
            intrinsics: Optional GT intrinsics (for posed mode)

        Returns:
            Dict with predictions (pose_enc, depth, world_points, etc.)
        """
        # Load and preprocess images
        images = self._load_images(image_files).to(self.device)

        # Access the underlying VGGT model through PEFT wrapper
        if hasattr(self.student.vggt, 'base_model'):
            vggt_model = self.student.vggt.base_model.model
        else:
            vggt_model = self.student.vggt

        # Run VGGT forward pass
        with torch.no_grad():
            predictions = vggt_model(images)

        # Convert pose_enc to extrinsics for evaluation
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            # Get actual image dimensions
            _, _, _, H, W = images.shape
            ext, intr = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),  # Use actual processed size
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)  # [S, 3, 4]
            predictions["intrinsics"] = intr.squeeze(0)  # [S, 3, 3]

        return predictions


class BaseVGGT:
    """
    Wrapper for base VGGT model (without LoRA) for comparison.
    """

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        device: str = "cuda",
        image_size: int = 504,
    ):
        print(f"Loading base VGGT model: {model_name}")

        self.image_size = image_size
        self.device = device

        # Load base VGGT model
        self.vggt = VGGT.from_pretrained(model_name)
        self.vggt.eval()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.vggt = self.vggt.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images using Free-Geometry preprocessing.

        - Resize longest side to image_size (default 504)
        - Preserve aspect ratio
        - Round dimensions to nearest multiple of PATCH_SIZE (14)
        """
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]

            # Resize longest side to image_size (Free-Geometry upper_bound_resize)
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Round to nearest multiple of PATCH_SIZE
            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            # Use appropriate interpolation
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        # All images in a scene have the same original size, so stack directly
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)

    def inference(
        self,
        image_files,
        extrinsics=None,
        intrinsics=None,
        **kwargs,
    ):
        """Run inference using the base VGGT model."""
        images = self._load_images(image_files).to(self.device)

        with torch.no_grad():
            predictions = self.vggt(images)

        # Convert pose_enc to extrinsics
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            # Get actual image dimensions
            _, _, _, H, W = images.shape
            ext, intr = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),  # Use actual processed size
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA-finetuned VGGT model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (if None, uses base model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/vggt-1b",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha (must match training)",
    )
    parser.add_argument(
        "--lora_layers_start",
        type=int,
        default=12,
        help="First layer with LoRA (must match training)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./workspace/vggt_evaluation",
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
        default=["pose"],
        help="Evaluation modes (pose, recon_unposed, recon_posed)",
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
        "--image_size",
        type=int,
        default=504,
        help="VGGT input image size (longest side, Free-Geometry style)",
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
        help="Multiple seeds for frame sampling. Runs once per seed, aggregates results with seed-tagged scene keys, per-seed means, and overall mean.",
    )
    parser.add_argument(
        "--eval_frames",
        type=int,
        default=None,
        help="If set, use even-indexed frames from max_frames sample",
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
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--subset_sampling",
        action="store_true",
        help="Use consecutive window sampling from subset (like training)",
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.1,
        help="Ratio of frames to include in subset (default: 0.1 = 10%%)",
    )
    args = parser.parse_args()

    # Determine seed list
    seeds = args.seeds if args.seeds else [args.seed]

    # Load model once (shared across seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = None

    if not args.print_only and not args.eval_only:
        if args.lora_path:
            api = LoRAVGGT(
                base_model=args.base_model,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_layers_start=args.lora_layers_start,
                image_size=args.image_size,
            ).to(device)
        else:
            api = BaseVGGT(
                model_name=args.base_model,
                image_size=args.image_size,
            ).to(device)

    if len(seeds) == 1:
        # Single seed: original behavior
        evaluator = VGGTEvaluator(
            work_dir=args.work_dir,
            datas=args.datasets,
            modes=args.modes,
            max_frames=args.max_frames,
            scenes=args.scenes,
            image_size=args.image_size,
            debug=args.debug,
            seed=seeds[0],
            eval_frames=args.eval_frames,
            subset_sampling=args.subset_sampling,
            subset_ratio=args.subset_ratio,
        )

        if args.print_only:
            evaluator.print_metrics()
            return
        if args.eval_only:
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
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
    import json
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

        evaluator = VGGTEvaluator(
            work_dir=seed_work_dir,
            datas=args.datasets,
            modes=args.modes,
            max_frames=args.max_frames,
            scenes=args.scenes,
            image_size=args.image_size,
            debug=args.debug,
            seed=seed,
            eval_frames=args.eval_frames,
            subset_sampling=args.subset_sampling,
            subset_ratio=args.subset_ratio,
        )
        evaluator.infer(api)


def _run_multi_seed_eval(args, seeds):
    """Run eval in sequential subprocesses across all seeds.

    Each eval subprocess uses ~25-30GB RAM. Running them sequentially ensures
    we stay within the cgroup memory limit (110GB) and avoid OOM kills.
    """
    import subprocess
    import json

    all_seed_metrics = {}

    print(f"\n{'='*60}")
    print(f"Running eval for {len(seeds)} seeds sequentially (subprocess isolation)")
    print(f"{'='*60}")

    for seed in seeds:
        seed_work_dir = os.path.join(args.work_dir, f"seed{seed}")

        cmd = [
            sys.executable,
            __file__,
            "--base_model", args.base_model,
            "--work_dir", seed_work_dir,
            "--datasets", *args.datasets,
            "--modes", *args.modes,
            "--max_frames", str(args.max_frames),
            "--image_size", str(args.image_size),
            "--seed", str(seed),
            "--eval_only",
        ]

        if args.lora_path:
            cmd.extend(["--lora_path", args.lora_path])
            cmd.extend(["--lora_rank", str(args.lora_rank)])
            cmd.extend(["--lora_alpha", str(args.lora_alpha)])
            cmd.extend(["--lora_layers_start", str(args.lora_layers_start)])
        if args.scenes:
            cmd.extend(["--scenes", *args.scenes])
        if args.eval_frames:
            cmd.extend(["--eval_frames", str(args.eval_frames)])
        if args.subset_sampling:
            cmd.append("--subset_sampling")
            cmd.extend(["--subset_ratio", str(args.subset_ratio)])
        if args.debug:
            cmd.append("--debug")

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

        metrics_file = os.path.join(seed_work_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                all_seed_metrics[seed] = json.load(f)
        else:
            print(f"  WARNING: No metrics file found for seed {seed}")
            all_seed_metrics[seed] = {}

    print(f"\nAll {len(seeds)} seeds completed. Aggregating results...")

    # Aggregate: build combined metrics with seed-tagged keys
    _print_multi_seed_summary(all_seed_metrics, seeds, args)


def _print_multi_seed_summary(all_seed_metrics, seeds, args):
    """Print combined multi-seed metrics summary."""
    # Collect all metric keys (e.g. "eth3d_pose", "eth3d_recon_unposed")
    all_keys = set()
    for seed_metrics in all_seed_metrics.values():
        all_keys.update(seed_metrics.keys())

    for metric_key in sorted(all_keys):
        print(f"\n{'='*60}")
        print(f"  {metric_key} — Multi-seed summary")
        print(f"{'='*60}")

        # Collect all scenes across seeds (excluding "mean")
        all_scenes = set()
        for seed in seeds:
            if metric_key in all_seed_metrics[seed]:
                for scene in all_seed_metrics[seed][metric_key]:
                    if scene != "mean":
                        all_scenes.add(scene)
        all_scenes = sorted(all_scenes)

        if not all_scenes:
            continue

        # Get metric names from first available result
        metric_names = None
        for seed in seeds:
            if metric_key in all_seed_metrics[seed] and all_scenes[0] in all_seed_metrics[seed][metric_key]:
                metric_names = list(all_seed_metrics[seed][metric_key][all_scenes[0]].keys())
                break
        if not metric_names:
            continue

        # Print header
        header = f"{'scene':<30s}"
        for name in metric_names:
            header += f"  {name:>10s}"
        print(header)
        print("-" * len(header))

        # Per-scene per-seed rows
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

        # Per-seed means
        print("-" * len(header))
        for seed in seeds:
            row = f"{'mean/seed' + str(seed):<30s}"
            for name in metric_names:
                vals = seed_means[seed][name]
                mean_val = np.mean(vals) if vals else float('nan')
                row += f"  {mean_val:>10.4f}"
            print(row)

        # Overall mean (across all seeds)
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
