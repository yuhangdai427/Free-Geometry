#!/usr/bin/env python3
"""Run Test3R-style Free-Geometry LoRA adaptation for VGGT."""

from __future__ import annotations

import argparse
import cv2
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Optional, Sequence, Union

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

if importlib.util.find_spec("plyfile") is None:
    plyfile_stub = types.ModuleType("plyfile")

    class _MissingPlyData:
        @staticmethod
        def read(*args, **kwargs):
            raise ImportError("DTU evaluation requires the optional 'plyfile' package")

    plyfile_stub.PlyData = _MissingPlyData
    sys.modules["plyfile"] = plyfile_stub

from vggt.bench.evaluator import VGGTEvaluator
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

FREE_GEO_MODEL_PATH = REPO_ROOT / "src" / "vggt" / "vggt" / "test_time_adaption" / "test3r_free_geo" / "model.py"
free_geo_spec = importlib.util.spec_from_file_location("vggt_test3r_free_geo_model", FREE_GEO_MODEL_PATH)
if free_geo_spec is None or free_geo_spec.loader is None:
    raise ImportError(f"Failed to load Free-Geo model module from {FREE_GEO_MODEL_PATH}")
free_geo_model = importlib.util.module_from_spec(free_geo_spec)
sys.modules[free_geo_spec.name] = free_geo_model
free_geo_spec.loader.exec_module(free_geo_model)
VGGTTest3RFreeGeoLoRAModel = free_geo_model.VGGTTest3RFreeGeoLoRAModel
adapt_lora_one_scene = free_geo_model.adapt_lora_one_scene


def resize_image_to_vggt(
    image_path: Union[str, Path],
    image_size: int = 504,
    patch_size: int = 14,
) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_h = max(patch_size, round(int(round(h * scale)) / patch_size) * patch_size)
    new_w = max(patch_size, round(int(round(w * scale)) / patch_size) * patch_size)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_images_for_vggt(
    image_files: Sequence[Union[str, Path]],
    image_size: int = 504,
    patch_size: int = 14,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    images = [resize_image_to_vggt(path, image_size=image_size, patch_size=patch_size) for path in image_files]
    return torch.stack(images, dim=0).to(device)


class VGGTTest3RFreeGeoAPI:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        image_size: int,
        epochs: int,
        lr: float,
        accum_iter: int,
        max_triplets: Optional[int],
        remove_degenerate_triplets: bool,
        seed: int,
        lora_dir: str,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_layers_start: int,
        lora_target: str,
        save_lora: bool = True,
        use_amp: bool = True,
    ) -> None:
        lora_layers = list(range(lora_layers_start, 24))
        self.model = VGGTTest3RFreeGeoLoRAModel(
            model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_layers=lora_layers,
            lora_target=lora_target,
        ).to(device)
        self.device = device
        self.image_size = image_size
        self.epochs = epochs
        self.lr = lr
        self.accum_iter = accum_iter
        self.max_triplets = max_triplets
        self.remove_degenerate_triplets = remove_degenerate_triplets
        self.seed = seed
        self.lora_dir = Path(lora_dir)
        self.save_lora = save_lora
        self.use_amp = use_amp
        self._call_idx = 0

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def inference(self, image_files, extrinsics=None, intrinsics=None, scene_name: str = None, **kwargs):
        images = load_images_for_vggt(
            image_files,
            image_size=self.image_size,
            device=self.device,
        )
        stats = adapt_lora_one_scene(
            self.model,
            images,
            seed=self.seed,
            epochs=self.epochs,
            lr=self.lr,
            max_triplets=self.max_triplets,
            accum_iter=self.accum_iter,
            use_amp=self.use_amp,
            remove_degenerate_triplets=self.remove_degenerate_triplets,
        )
        print(
            "[VGGT-test3r-free-geo][LoRA-TTA] "
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

        prompt_name = scene_name or f"call_{self._call_idx:05d}"
        lora_name = prompt_name.replace("/", "_")
        if self.save_lora:
            self.model.save_lora_checkpoint(
                self.lora_dir / f"{lora_name}_seed{self.seed}_{len(image_files)}v.pt",
                {
                    "scene_name": scene_name,
                    "seed": self.seed,
                    "num_views": len(image_files),
                    "image_files": list(map(str, image_files)),
                    "adaptation": stats.__dict__,
                },
            )

        self._call_idx += 1
        with torch.no_grad():
            predictions = self.model(images[None])

        if "pose_enc" in predictions:
            _, _, _, h, w = images[None].shape
            ext, intr = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                image_size_hw=(h, w),
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        predictions["test3r_free_geo_adaptation"] = stats.__dict__
        return predictions


class FreeGeoAdaptedVGGTEvaluator(VGGTEvaluator):
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
        with open(os.path.join(results_dir, "test3r_free_geo_adaptation.json"), "w") as f:
            json.dump(predictions.get("test3r_free_geo_adaptation", {}), f, indent=2)

    @staticmethod
    def _scene_name_from_export_dir(export_dir: str) -> str:
        parts = Path(export_dir).parts
        if len(parts) >= 2:
            return parts[-2]
        return Path(export_dir).name


def parse_int_or_none(value: str) -> Optional[int]:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT Test3R-style Free-Geometry LoRA TTA")
    parser.add_argument("--base_model", default="facebook/vggt-1b")
    parser.add_argument("--work_dir", default="./workspace/vggt_test3r_free_geo")
    parser.add_argument("--datasets", nargs="+", default=["7scenes"])
    parser.add_argument("--modes", nargs="+", default=["pose"])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--view_counts", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--image_size", type=int, default=504)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum_iter", type=int, default=2)
    parser.add_argument("--max_triplets", type=parse_int_or_none, default=None)
    parser.add_argument("--remove_degenerate_triplets", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers_start", type=int, default=0)
    parser.add_argument(
        "--lora_target",
        choices=["attention", "attention_mlp"],
        default="attention_mlp",
        help="attention_mlp mirrors the existing Free-Geometry VGGT LoRA target set.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--infer_only", action="store_true")
    parser.add_argument("--print_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_save_lora", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = {}

    for seed in args.seeds:
        for n_views in args.view_counts:
            run_name = f"seed{seed}_{n_views}v"
            run_work_dir = os.path.join(args.work_dir, run_name)
            print(f"\n[VGGT-test3r-free-geo] {run_name}")
            evaluator = FreeGeoAdaptedVGGTEvaluator(
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
                api = VGGTTest3RFreeGeoAPI(
                    model_name=args.base_model,
                    device=device,
                    image_size=args.image_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    accum_iter=args.accum_iter,
                    max_triplets=args.max_triplets,
                    remove_degenerate_triplets=args.remove_degenerate_triplets,
                    seed=seed,
                    lora_dir=os.path.join(run_work_dir, "lora"),
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_layers_start=args.lora_layers_start,
                    lora_target=args.lora_target,
                    save_lora=not args.no_save_lora,
                    use_amp=not args.no_amp,
                )
                evaluator.infer(api)
                del api
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not args.infer_only:
                metrics = evaluator.eval()
                evaluator.print_metrics(metrics)
                all_metrics[run_name] = metrics

    summary_path = os.path.join(args.work_dir, "vggt_test3r_free_geo_summary.json")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
