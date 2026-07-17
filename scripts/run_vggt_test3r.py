#!/usr/bin/env python3
"""Run VGGT-test3r scene-wise test-time adaptation and VGGT evaluation.

Feed-forward and eval delegate to scripts/benchmark_vggt.BaseVGGT so the
baseline path is bit-identical to the reference benchmark script.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Keep Hugging Face downloads inside this repo and use the mirror endpoint.
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(LOCAL_HF_HOME / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))
sys.path.insert(0, os.path.dirname(__file__))  # so we can import benchmark_vggt

from benchmark_vggt import BaseVGGT  # authoritative feed-forward + preprocessing
from vggt.models.vggt_test3r import load_vggt_test3r
from vggt.bench.evaluator import VGGTEvaluator
from vggt.test_time_adaption.test3r_utils import (
    adapt_prompt_one_scene,
    save_prompt_checkpoint,
)


class VGGTTest3RAPI:
    """Adapter that applies Test3R-style prompt TTA then delegates the final
    feed-forward to benchmark_vggt.BaseVGGT so eval matches the reference."""

    def __init__(
        self,
        model_name: str,
        prompt_size: int,
        device: torch.device,
        image_size: int,
        epochs: int,
        lr: float,
        accum_iter: int,
        max_triplets: Optional[int],
        remove_degenerate_triplets: bool,
        seed: int,
        prompt_dir: str,
        save_prompts: bool = True,
        use_amp: bool = True,
        no_ttt: bool = False,
        triplet_batch_size: int = 1,
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.epochs = epochs
        self.lr = lr
        self.accum_iter = accum_iter
        self.max_triplets = max_triplets
        self.remove_degenerate_triplets = remove_degenerate_triplets
        self.seed = seed
        self.prompt_dir = Path(prompt_dir)
        self.save_prompts = save_prompts
        self.use_amp = use_amp
        self.no_ttt = no_ttt
        self.triplet_batch_size = int(triplet_batch_size)
        self._call_idx = 0

        # Reference baseline wrapper: preprocessing, model load, forward, and
        # pose_enc post-processing come from benchmark_vggt.BaseVGGT. In the
        # adapted path we swap its .vggt with the prompt-augmented module
        # before the final forward so the rest of the pipeline is unchanged.
        self.base = BaseVGGT(model_name=model_name, image_size=image_size).to(self.device)

        if no_ttt:
            self.model = None  # baseline: just use self.base.vggt
        else:
            self.model = load_vggt_test3r(
                model_name=model_name,
                prompt_size=prompt_size,
                device=self.device,
            )

    def to(self, device):
        self.device = torch.device(device)
        self.base.to(self.device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def inference(self, image_files, extrinsics=None, intrinsics=None, scene_name: str = None, **kwargs):
        # Always use benchmark_vggt preprocessing — shared by baseline and adapted.
        if self.no_ttt:
            print(
                "[VGGT-test3r][TTA] "
                f"scene={scene_name or self._call_idx} "
                f"views={len(image_files)} baseline (no adaptation)",
                flush=True,
            )
            stats = None
            self._call_idx += 1
            # Delegate the full forward path (preprocessing + VGGT + pose_enc
            # post-processing) to BaseVGGT so metrics match benchmark_vggt.py.
            predictions = self.base.inference(image_files, extrinsics=extrinsics, intrinsics=intrinsics, **kwargs)
            predictions["test3r_adaptation"] = {"no_ttt": True}
            return predictions

        # Adapted path: load images via the reference preprocessor, run TTA,
        # then run the final feed-forward through BaseVGGT with the prompt-
        # augmented aggregator swapped in.
        images_bsch = self.base._load_images(image_files).to(self.device)   # [1, S, 3, H, W]
        images = images_bsch.squeeze(0)                                      # [S, 3, H, W]

        stats = adapt_prompt_one_scene(
            self.model,
            images,
            seed=self.seed,
            epochs=self.epochs,
            lr=self.lr,
            max_triplets=self.max_triplets,
            accum_iter=self.accum_iter,
            use_amp=self.use_amp,
            remove_degenerate_triplets=self.remove_degenerate_triplets,
            triplet_batch_size=self.triplet_batch_size,
        )
        print(
            "[VGGT-test3r][TTA] "
            f"scene={scene_name or self._call_idx} "
            f"views={len(image_files)} "
            f"triplets={stats.num_triplets} "
            f"steps={stats.optimizer_steps} "
            f"time={stats.adaptation_time_sec:.2f}s "
            f"time_per_triplet={stats.time_per_triplet_sec:.4f}s "
            f"peak_alloc={stats.cuda_peak_allocated_mb if stats.cuda_peak_allocated_mb is not None else 'NA'}MB "
            f"mean_alloc={stats.cuda_mean_allocated_mb if stats.cuda_mean_allocated_mb is not None else 'NA'}MB "
            f"loss={stats.loss_last if stats.loss_last is not None else 'NA'}",
            flush=True,
        )

        if self.save_prompts:
            prompt_name = scene_name or f"call_{self._call_idx:05d}"
            prompt_name = prompt_name.replace("/", "_")
            save_prompt_checkpoint(
                self.model,
                self.prompt_dir / f"{prompt_name}_seed{self.seed}_{len(image_files)}v.pt",
                {
                    "scene_name": scene_name,
                    "seed": self.seed,
                    "num_views": len(image_files),
                    "image_files": list(map(str, image_files)),
                    "adaptation": stats.__dict__,
                },
            )

        self._call_idx += 1

        # Final feed-forward through the BaseVGGT code path, with our adapted
        # model swapped in so the prompt is active. This guarantees the eval
        # pipeline (forward under no_grad, pose_enc → extrinsics/intrinsics,
        # returned keys) is identical to benchmark_vggt.BaseVGGT.inference.
        original_vggt = self.base.vggt
        try:
            self.base.vggt = self.model
            self.base.vggt.eval()
            predictions = self.base.inference(image_files, extrinsics=extrinsics, intrinsics=intrinsics, **kwargs)
        finally:
            self.base.vggt = original_vggt

        predictions["test3r_adaptation"] = stats.__dict__
        return predictions


class AdaptedVGGTEvaluator(VGGTEvaluator):
    def _sample_frames(self, scene_data, scene: str):
        if self.max_frames == 0:
            print(f"  [Sampling] {scene}: using all {len(scene_data.image_files)} frames")
            return scene_data
        return super()._sample_frames(scene_data, scene)

    def _run_inference(self, api, scene_data, export_dir: str, use_gt_poses: bool = False) -> None:
        scene_name = self._scene_name_from_export_dir(export_dir)
        if use_gt_poses:
            predictions = api.inference(
                scene_data.image_files,
                extrinsics=scene_data.extrinsics,
                intrinsics=scene_data.intrinsics,
                scene_name=scene_name,
            )
        else:
            predictions = api.inference(scene_data.image_files, scene_name=scene_name)

        results_dir = os.path.join(export_dir, "exports", "mini_npz")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.npz")
        save_dict = {}

        if "extrinsics" in predictions:
            save_dict["extrinsics"] = predictions["extrinsics"].detach().cpu().numpy()
        if "intrinsics" in predictions:
            save_dict["intrinsics"] = predictions["intrinsics"].detach().cpu().numpy()
        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 5:
                depth = depth.squeeze(0).squeeze(-1)
            elif depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            save_dict["depth"] = depth.detach().cpu().numpy()
        if "depth_conf" in predictions:
            depth_conf = predictions["depth_conf"]
            if depth_conf.dim() == 5:
                depth_conf = depth_conf.squeeze(0).squeeze(1)
            elif depth_conf.dim() == 4:
                depth_conf = depth_conf.squeeze(0)
            save_dict["depth_conf"] = depth_conf.detach().cpu().numpy()
        if "world_points" in predictions:
            world_points = predictions["world_points"]
            if world_points.dim() == 5:
                world_points = world_points.squeeze(0)
            save_dict["world_points"] = world_points.detach().cpu().numpy()
        if "world_points_conf" in predictions:
            world_points_conf = predictions["world_points_conf"]
            if world_points_conf.dim() == 4:
                world_points_conf = world_points_conf.squeeze(0)
            save_dict["world_points_conf"] = world_points_conf.detach().cpu().numpy()

        np.savez_compressed(results_path, **save_dict)
        with open(os.path.join(results_dir, "test3r_adaptation.json"), "w") as f:
            json.dump(predictions.get("test3r_adaptation", {}), f, indent=2)

    @staticmethod
    def _scene_name_from_export_dir(export_dir: str) -> str:
        parts = Path(export_dir).parts
        if "model_results" in parts:
            i = parts.index("model_results")
            # parts: .../model_results/<dataset>/<scene...>/<suffix>
            middle = parts[i + 2 : -1]
            if middle:
                return "_".join(middle)
        if len(parts) >= 2:
            return parts[-2]
        return Path(export_dir).name


def parse_int_or_none(value: str) -> Optional[int]:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="VGGT-test3r test-time adaptation pipeline")
    parser.add_argument("--base_model", default="facebook/vggt-1b")
    parser.add_argument("--work_dir", default="./workspace/vggt_test3r")
    parser.add_argument("--datasets", nargs="+", default=["7scenes"])
    parser.add_argument("--modes", nargs="+", default=["pose"])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--view_counts", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--image_size", type=int, default=504)
    parser.add_argument("--prompt_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum_iter", type=int, default=2)
    parser.add_argument("--max_triplets", type=parse_int_or_none, default=None)
    parser.add_argument(
        "--triplet_batch_size",
        type=int,
        default=1,
        help="Number of triplets packed into a single VGGT forward during TTA. "
        "1 = original serial behavior; larger values speed up TTA on big GPUs.",
    )
    parser.add_argument(
        "--remove_degenerate_triplets",
        action="store_true",
        help="Debug ablation: skip triplets with repeated frame indices. Original Test3R keeps them.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--print_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--no_ttt",
        action="store_true",
        help="Baseline: skip prompt adaptation and run plain VGGT inference.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = {}

    for seed in args.seeds:
        for n_views in args.view_counts:
            run_name = f"seed{seed}_{n_views}v"
            run_work_dir = os.path.join(args.work_dir, run_name)
            print(f"\n[VGGT-test3r] {run_name}")
            evaluator = AdaptedVGGTEvaluator(
                work_dir=run_work_dir,
                datas=args.datasets,
                modes=args.modes,
                scenes=args.scenes,
                max_frames=n_views,
                image_size=args.image_size,
                seed=seed,
                debug=args.debug,
            )

            if args.print_only:
                evaluator.print_metrics()
                continue

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
                    prompt_dir=os.path.join(run_work_dir, "prompts"),
                    use_amp=not args.no_amp,
                    no_ttt=args.no_ttt,
                    triplet_batch_size=args.triplet_batch_size,
                )
                evaluator.infer(api)
                del api
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
            all_metrics[run_name] = metrics

    summary_path = os.path.join(args.work_dir, "vggt_test3r_summary.json")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
