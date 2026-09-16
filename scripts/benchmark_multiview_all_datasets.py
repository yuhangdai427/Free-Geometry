#!/usr/bin/env python3
"""
Benchmark DA3 or VGGT Free-Geometry results on multiple datasets at multiple input-view counts.

This replaces the old fixed teacher/student 8v->4v workflow with direct
multi-view benchmarking. Each requested view count is saved as its own
experiment folder (`8v`, `16v`, `32v`, ...), including:
  - prediction NPZs under `model_results/`
  - raw GLB + depth visualizations under `visualizations/`
  - evaluator outputs under `metric_results/`

Usage:
    python scripts/benchmark_multiview_all_datasets.py --model_family da3 --results_root results
    python scripts/benchmark_multiview_all_datasets.py --model_family vggt --results_root results
    python scripts/benchmark_multiview_all_datasets.py --model_family da3 --lora_path checkpoints/.../epoch_3_lora.pt --work_dir results/multiview_da3_hiroom_scannetpp_lora_fixed
    python scripts/benchmark_multiview_all_datasets.py --model_family vggt --no_lora --work_dir results/multiview_vggt_hiroom_scannetpp_base_fixed --datasets hiroom scannetpp --view_counts 8 16 32
"""

import argparse
import hashlib
import json
import os
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))


DEFAULT_DATASETS = ["scannetpp", "hiroom", "7scenes", "eth3d"]
DEFAULT_SCENE = "20241230/828805/cam_sampled_06"
DEFAULT_VIEW_COUNTS = [8, 16, 32]
EVAL_MODES = ["pose", "recon_unposed"]
DEFAULT_DA3_LORA_ROOT = "checkpoints/da3_lora_final"
DEFAULT_VGGT_LORA_ROOT = "checkpoints/vggt_lora_final"
DEFAULT_RESULTS_ROOT = "results"


def load_view_index_file(path: str, expected_pool_size: int) -> Dict[Tuple[str, int], List[int]]:
    """Load an auditable per-scene view file generated for an E6 evaluation seed."""
    pools: Dict[Tuple[str, int], List[int]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                key = (str(record["scene_id"]), int(record["sample_idx"]))
                indices = [int(value) for value in record["frame_indices"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid view-index entry at {path}:{line_number}") from exc
            if not indices or len(indices) > expected_pool_size or len(indices) != len(set(indices)) or indices[0] != 0:
                raise ValueError(f"Expected unique pool of at most P{expected_pool_size} beginning at ref 0: {path}:{line_number}")
            if key in pools:
                raise ValueError(f"Duplicate view-index entry for {key} in {path}")
            pools[key] = indices
    return pools


class LoRADepthAnything3:
    """Minimal wrapper to make a Free-Geometry DA3 LoRA model match the benchmark API."""

    def __init__(self, base_model="depth-anything/DA3-GIANT-1.1", lora_path=None, device="cuda"):
        from depth_anything_3.test_time_adaption.models import StudentModel

        print(f"Loading DA3 base model: {base_model}")
        print(f"Loading DA3 LoRA weights: {lora_path}")

        lora_rank = None
        lora_alpha = None
        peft_dir = lora_path.replace(".pt", "_peft") if lora_path else None
        if peft_dir and os.path.isdir(peft_dir):
            config_path = os.path.join(peft_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    peft_cfg = json.load(f)
                lora_rank = peft_cfg.get("r")
                lora_alpha = peft_cfg.get("lora_alpha")
                print(f"Auto-detected DA3 LoRA config: rank={lora_rank}, alpha={lora_alpha}")

        if lora_rank is None:
            lora_rank = 16
        if lora_alpha is None:
            lora_alpha = float(lora_rank)

        self.student = StudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            patch_swiglu_mlp_for_lora=False,
        )

        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        peft_model = self.student.da3.model.backbone.pretrained
        self.student.da3.model.backbone.pretrained = peft_model.merge_and_unload()
        print("Merged DA3 LoRA weights into the base model for inference")

        self.student.eval()
        self.device = device

    def to(self, device):
        self.device = device
        self.student = self.student.to(device)
        return self

    def inference(self, image, export_dir=None, export_format="mini_npz", ref_view_strategy="first", **kwargs):
        return self.student.da3.inference(
            image=image,
            export_dir=export_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            **kwargs,
        )


class LoRAVGGT:
    """Minimal wrapper to make a Free-Geometry VGGT LoRA model match the benchmark API."""

    def __init__(
        self,
        base_model="facebook/vggt-1b",
        lora_path=None,
        lora_rank=32,
        lora_alpha=32.0,
        lora_layers_start=0,
        device="cuda",
        image_size=504,
    ):
        from vggt.vggt.test_time_adaption.models import VGGTStudentModel
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        print(f"Loading VGGT base model: {base_model}")
        print(f"Loading VGGT LoRA weights: {lora_path}")

        self.image_size = image_size
        self.device = device
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        detected = self._detect_lora_config(
            lora_path,
            default_rank=lora_rank,
            default_alpha=lora_alpha,
            default_layers_start=lora_layers_start,
        )
        lora_rank = detected["rank"]
        lora_alpha = detected["alpha"]
        lora_layers_start = detected["layers_start"]
        print(
            f"Using VGGT LoRA config: rank={lora_rank}, "
            f"alpha={lora_alpha}, layers_start={lora_layers_start}"
        )

        lora_layers = list(range(lora_layers_start, 24))
        self.student = VGGTStudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
        )

        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        elif lora_path:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        self.student.eval()

    @staticmethod
    def _detect_lora_config(
        lora_path: Optional[str],
        *,
        default_rank: int,
        default_alpha: float,
        default_layers_start: int,
    ) -> Dict[str, float]:
        rank = int(default_rank)
        alpha = float(default_alpha)
        layers_start = int(default_layers_start)

        peft_dir = lora_path.replace(".pt", "_peft") if lora_path else None
        cfg_path = os.path.join(peft_dir, "adapter_config.json") if peft_dir else None
        if not cfg_path or not os.path.exists(cfg_path):
            return {"rank": rank, "alpha": alpha, "layers_start": layers_start}

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return {"rank": rank, "alpha": alpha, "layers_start": layers_start}

        try:
            if cfg.get("r") is not None:
                rank = int(cfg["r"])
            if cfg.get("lora_alpha") is not None:
                alpha = float(cfg["lora_alpha"])

            target_modules = cfg.get("target_modules") or []
            layer_ids = []
            for module_name in target_modules:
                if not isinstance(module_name, str):
                    continue
                for part in module_name.split("."):
                    if part.isdigit():
                        layer_ids.append(int(part))
                        break
            if layer_ids:
                layers_start = min(layer_ids)
        except Exception:
            pass

        return {"rank": rank, "alpha": alpha, "layers_start": layers_start}

    def to(self, device):
        self.device = device
        self.student = self.student.to(device)
        return self

    def _load_images(self, image_files):
        import cv2

        patch_size = 14
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            scale = self.image_size / max(h, w)
            new_h = max(patch_size, int(round(h * scale / patch_size) * patch_size))
            new_w = max(patch_size, int(round(w * scale / patch_size) * patch_size))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            images.append(img.astype(np.float32) / 255.0)

        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)

    def inference(self, image_files, **kwargs):
        images = self._load_images(image_files).to(self.device)

        if hasattr(self.student.vggt, "base_model"):
            model = self.student.vggt.base_model.model
        else:
            model = self.student.vggt

        with torch.no_grad():
            predictions = model(images)

        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)
            _, _, _, h, w = images.shape
            ext, intr = self.pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(h, w),
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


class BaseVGGT:
    """Base VGGT wrapper with the same inference interface as LoRAVGGT."""

    def __init__(self, model_name="facebook/vggt-1b", device="cuda", image_size=504):
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        print(f"Loading VGGT model: {model_name}")
        self.image_size = image_size
        self.device = device
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self.vggt = VGGT.from_pretrained(model_name)
        self.vggt.eval()

    def to(self, device):
        self.device = device
        self.vggt = self.vggt.to(device)
        return self

    def _load_images(self, image_files):
        import cv2

        patch_size = 14
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            scale = self.image_size / max(h, w)
            new_h = max(patch_size, int(round(h * scale / patch_size) * patch_size))
            new_w = max(patch_size, int(round(w * scale / patch_size) * patch_size))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            images.append(img.astype(np.float32) / 255.0)

        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)

    def inference(self, image_files, **kwargs):
        images = self._load_images(image_files).to(self.device)

        with torch.no_grad():
            predictions = self.vggt(images)

        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)
            _, _, _, h, w = images.shape
            ext, intr = self.pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(h, w),
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


def scene_slug(scene: str) -> str:
    return "-".join(scene.split("/")[-3:])


def exp_key(view_count: int) -> str:
    return f"{int(view_count)}v"


def _results_npz_path(work_dir: str, experiment: str, dataset_name: str, scene: str) -> str:
    return os.path.join(
        work_dir,
        experiment,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "mini_npz",
        "results.npz",
    )


def _gt_meta_path(work_dir: str, experiment: str, dataset_name: str, scene: str) -> str:
    return os.path.join(
        work_dir,
        experiment,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "gt_meta.npz",
    )


def _visualization_dir(work_dir: str, experiment: str, dataset_name: str, scene: str) -> str:
    return os.path.join(work_dir, experiment, "visualizations", dataset_name, scene_slug(scene))


def save_results_npz(export_dir, depth, extrinsics, intrinsics, conf=None):
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dict = {"depth": np.round(depth, 8), "extrinsics": extrinsics, "intrinsics": intrinsics}
    if conf is not None:
        save_dict["conf"] = np.round(conf, 2)
    np.savez_compressed(output_file, **save_dict)


def save_gt_meta(
    export_dir,
    extrinsics,
    intrinsics,
    image_files,
    *,
    requested_view_count: int,
    actual_view_count: int,
    sampled_indices: Sequence[int],
):
    meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    np.savez_compressed(
        meta_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array(image_files, dtype=object),
        requested_view_count=np.array([requested_view_count], dtype=np.int32),
        actual_view_count=np.array([actual_view_count], dtype=np.int32),
        sampled_indices=np.array(list(sampled_indices), dtype=np.int32),
    )


def _canonical_scene_views(scene_data) -> Tuple[List[str], np.ndarray, np.ndarray]:
    image_files = [str(p) for p in scene_data.image_files]
    order = sorted(range(len(image_files)), key=lambda idx: image_files[idx])
    ordered_image_files = [image_files[idx] for idx in order]
    ordered_extrinsics = np.asarray(scene_data.extrinsics)[order]
    ordered_intrinsics = np.asarray(scene_data.intrinsics)[order]
    return ordered_image_files, ordered_extrinsics, ordered_intrinsics


def _stable_scene_seed(dataset_name: str, scene: str, seed: int, view_count: int) -> int:
    key = f"{dataset_name}::{scene}::{int(seed)}::{int(view_count)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_view_indices(
    *,
    num_frames: int,
    target_views: int,
    dataset_name: str,
    scene: str,
    seed: int = 43,
) -> List[int]:
    if num_frames <= target_views:
        return list(range(num_frames))
    rng = random.Random(_stable_scene_seed(dataset_name, scene, seed, target_views))
    indices = list(range(num_frames))
    rng.shuffle(indices)
    return sorted(indices[:target_views])


def _load_metric_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_scene_metrics(work_dir: str, experiment: str, dataset_name: str, mode: str) -> Dict[str, dict]:
    path = os.path.join(work_dir, experiment, "metric_results", f"{dataset_name}_{mode}.json")
    data = _load_metric_json(path)
    return {k: v for k, v in data.items() if k != "mean" and isinstance(v, dict)}


def _mean_metrics(work_dir: str, experiment: str, dataset_name: str, mode: str) -> Dict[str, float]:
    path = os.path.join(work_dir, experiment, "metric_results", f"{dataset_name}_{mode}.json")
    data = _load_metric_json(path)
    mean = data.get("mean")
    return mean if isinstance(mean, dict) else {}


def _choose_metric_key(
    scene_maps: List[Dict[str, dict]],
    preferred_keys: List[str],
    fallback_keys: List[str],
) -> Tuple[Optional[str], bool]:
    all_entries = []
    for scene_map in scene_maps:
        all_entries.extend(scene_map.values())

    for key in preferred_keys:
        if any(key in entry for entry in all_entries):
            return key, True
    for key in fallback_keys:
        if any(key in entry for entry in all_entries):
            return key, False
    return None, True


def _resolve_scenes(dataset_name: str, dataset, args) -> List[str]:
    available = list(getattr(dataset, "SCENES", []))
    if not available:
        print(f"[WARN] Dataset {dataset_name} has no registered scenes")
        return []

    available_set = set(available)

    if args.all_scenes:
        return available

    if args.scenes:
        matched = [s for s in args.scenes if s in available_set]
        missing = [s for s in args.scenes if s not in available_set]
        if missing:
            print(f"[WARN] {dataset_name}: skipping unknown scenes: {missing}")
        return matched

    if args.scene:
        if args.scene in available_set:
            return [args.scene]
        print(f"[WARN] {dataset_name}: scene not found: {args.scene}")
        return []

    if dataset_name == "hiroom" and DEFAULT_SCENE in available_set:
        return [DEFAULT_SCENE] + [s for s in available if s != DEFAULT_SCENE]
    return available


def _scenes_from_existing_metrics(work_dir: str, dataset_name: str, experiments: Sequence[str]) -> List[str]:
    for experiment in experiments:
        path = os.path.join(work_dir, experiment, "metric_results", f"{dataset_name}_pose.json")
        data = _load_metric_json(path)
        if data:
            scenes = [k for k, v in data.items() if k != "mean" and isinstance(v, dict)]
            if scenes:
                return sorted(scenes)
    return []


def _export_vggt_visualizations(depth, extrinsics, intrinsics, image_files, vis_dir):
    from depth_anything_3.specs import Prediction
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    images = []
    for img_path in image_files:
        img = imageio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        images.append(img)

    images = np.stack(images, axis=0)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    if images.shape[1:3] != depth.shape[1:3]:
        import cv2

        resized_images = []
        for img in images:
            resized_images.append(
                cv2.resize(img, (depth.shape[2], depth.shape[1]), interpolation=cv2.INTER_LINEAR)
            )
        images = np.stack(resized_images, axis=0)

    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype)
        ext_4x4[:, :3, :] = extrinsics
        ext_4x4[:, 3, 3] = 1.0
        extrinsics = ext_4x4

    prediction = Prediction(
        depth=depth,
        is_metric=1,
        conf=np.ones_like(depth) * 1.5,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_images=images.astype(np.uint8),
    )

    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)


def run_da3_inference(model, image_files, scene_data, selected_indices, work_dir, experiment, dataset_name, scene):
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    ext = scene_data.extrinsics[selected_indices]
    intr = scene_data.intrinsics[selected_indices]

    prediction = model.inference(image_files, ref_view_strategy="first")

    export_dir = os.path.join(work_dir, experiment, "model_results", dataset_name, scene, "unposed")
    save_results_npz(
        export_dir,
        prediction.depth,
        prediction.extrinsics,
        prediction.intrinsics,
        getattr(prediction, "conf", None),
    )
    save_gt_meta(
        export_dir,
        ext,
        intr,
        image_files,
        requested_view_count=int(experiment[:-1]),
        actual_view_count=len(image_files),
        sampled_indices=selected_indices,
    )

    vis_dir = _visualization_dir(work_dir, experiment, dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)


def run_vggt_inference(model, image_files, scene_data, selected_indices, work_dir, experiment, dataset_name, scene):
    ext = scene_data.extrinsics[selected_indices]
    intr = scene_data.intrinsics[selected_indices]

    predictions = model.inference(image_files)

    depth = predictions["depth"].squeeze(0).cpu().numpy()
    pred_ext = predictions["extrinsics"].squeeze(0).cpu().numpy()
    pred_intr = predictions["intrinsics"].squeeze(0).cpu().numpy()

    export_dir = os.path.join(work_dir, experiment, "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, depth, pred_ext, pred_intr, conf=None)
    save_gt_meta(
        export_dir,
        ext,
        intr,
        image_files,
        requested_view_count=int(experiment[:-1]),
        actual_view_count=len(image_files),
        sampled_indices=selected_indices,
    )

    vis_dir = _visualization_dir(work_dir, experiment, dataset_name, scene)
    _export_vggt_visualizations(depth, pred_ext, pred_intr, image_files, vis_dir)


def run_evaluation(
    model_family: str,
    work_dir: str,
    scenes_by_dataset: Dict[str, List[str]],
    experiments: Sequence[str],
    num_fusion_workers: int,
) -> None:
    if model_family == "da3":
        from depth_anything_3.bench.evaluator import Evaluator as EvalCls
    else:
        from vggt.vggt.bench.evaluator import VGGTEvaluator as EvalCls

    for experiment in experiments:
        exp_work_dir = os.path.join(work_dir, experiment)
        if not os.path.exists(os.path.join(exp_work_dir, "model_results")):
            print(f"  Skipping {experiment} (no model_results found)")
            continue

        print(f"\n{'=' * 64}")
        print(f"Evaluating: {experiment}")
        print(f"{'=' * 64}")

        for dataset_name, scenes in scenes_by_dataset.items():
            if not scenes:
                continue

            scenes_to_eval = []
            for scene in scenes:
                if os.path.exists(_results_npz_path(work_dir, experiment, dataset_name, scene)):
                    scenes_to_eval.append(scene)

            if not scenes_to_eval:
                print(f"  Skipping {experiment}/{dataset_name} (no results.npz found)")
                continue

            print(f"\n  Dataset: {dataset_name} ({len(scenes_to_eval)} scenes)")
            evaluator = EvalCls(
                work_dir=exp_work_dir,
                datas=[dataset_name],
                modes=EVAL_MODES,
                max_frames=-1,
                scenes=scenes_to_eval,
                num_fusion_workers=num_fusion_workers,
            )
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)


def write_manifest(
    path: str,
    *,
    model_family: str,
    model_name: str,
    lora_path: Optional[str],
    datasets: Sequence[str],
    view_counts: Sequence[int],
    scenes_by_dataset: Dict[str, List[str]],
) -> None:
    manifest = {
        "layout": "multiview_benchmark_v1",
        "model_family": model_family,
        "model_name": model_name,
        "lora_path": lora_path,
        "datasets": list(datasets),
        "view_counts": [int(v) for v in view_counts],
        "experiments": [
            {"key": exp_key(v), "label": f"{int(v)} views", "view_count": int(v)}
            for v in view_counts
        ],
        "scenes_by_dataset": scenes_by_dataset,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_summary_report(work_dir: str, scenes_by_dataset: Dict[str, List[str]], experiments: Sequence[str]) -> None:
    summary = {
        "datasets": {},
        "experiments": list(experiments),
    }

    print(f"\n{'=' * 72}")
    print("Multi-View Summary Report")
    print(f"{'=' * 72}")

    for dataset_name, scenes in scenes_by_dataset.items():
        dataset_out = {"mean_pose": {}, "mean_recon_unposed": {}, "best_by_scene": {}}
        summary["datasets"][dataset_name] = dataset_out

        scene_pose_maps = [_load_scene_metrics(work_dir, exp, dataset_name, "pose") for exp in experiments]
        scene_recon_maps = [_load_scene_metrics(work_dir, exp, dataset_name, "recon_unposed") for exp in experiments]

        pose_key, pose_higher_is_better = _choose_metric_key(
            scene_pose_maps,
            preferred_keys=["auc03", "auc_03", "auc3", "auc_3"],
            fallback_keys=[],
        )
        recon_key, recon_higher_is_better = _choose_metric_key(
            scene_recon_maps,
            preferred_keys=["fscore", "f_score", "f-score", "precision", "recall"],
            fallback_keys=["overall", "acc", "comp"],
        )

        print(f"\n[{dataset_name}]")
        if pose_key is not None:
            print(f"  Pose mean ({pose_key}):")
            for experiment in experiments:
                mean_pose = _mean_metrics(work_dir, experiment, dataset_name, "pose")
                value = mean_pose.get(pose_key)
                dataset_out["mean_pose"][experiment] = value
                if value is None:
                    print(f"    {experiment:>4}: —")
                else:
                    print(f"    {experiment:>4}: {float(value):.4f}")

        if recon_key is not None:
            print(f"  Recon mean ({recon_key}):")
            for experiment in experiments:
                mean_recon = _mean_metrics(work_dir, experiment, dataset_name, "recon_unposed")
                value = mean_recon.get(recon_key)
                dataset_out["mean_recon_unposed"][experiment] = value
                if value is None:
                    print(f"    {experiment:>4}: —")
                else:
                    print(f"    {experiment:>4}: {float(value):.4f}")

        for scene in scenes:
            pose_best = None
            recon_best = None

            if pose_key is not None:
                pose_values = []
                for idx, experiment in enumerate(experiments):
                    value = scene_pose_maps[idx].get(scene, {}).get(pose_key)
                    if value is not None:
                        pose_values.append((float(value), experiment))
                if pose_values:
                    pose_best = (max if pose_higher_is_better else min)(pose_values)[1]

            if recon_key is not None:
                recon_values = []
                for idx, experiment in enumerate(experiments):
                    value = scene_recon_maps[idx].get(scene, {}).get(recon_key)
                    if value is not None:
                        recon_values.append((float(value), experiment))
                if recon_values:
                    recon_best = (max if recon_higher_is_better else min)(recon_values)[1]

            dataset_out["best_by_scene"][scene] = {
                "pose": pose_best,
                "recon_unposed": recon_best,
            }

    out_path = os.path.join(work_dir, "multiview_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\nSaved summary: {out_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DA3 or VGGT Free-Geometry results at 8v/16v/32v across multiple datasets"
    )
    parser.add_argument("--model_family", choices=["da3", "vggt"], required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--view_counts", nargs="+", type=int, default=DEFAULT_VIEW_COUNTS)
    parser.add_argument("--all_scenes", action="store_true", help="Run all scenes for each dataset")
    parser.add_argument("--scene", type=str, default=None, help="Single scene to run")
    parser.add_argument(
        "--scenes",
        action="append",
        default=None,
        help="Repeatable scene filter. Example: --scenes scene_a --scenes scene_b",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--view_indices_file",
        type=str,
        default=None,
        help="Optional JSONL with fixed per-scene evaluation indices. Requires one entry per scene at sample_idx=0.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Shared results root. Use the same directory passed to scripts/visualize_free_geometry.py",
    )
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_root", type=str, default=None)
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable Free-Geometry adaptation and benchmark the base model instead",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_layers_start", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=504, help="VGGT input image size")
    parser.add_argument(
        "--num_fusion_workers",
        type=int,
        default=4,
        help="CPU workers for per-scene TSDF reconstruction fusion (default: 4)",
    )
    parser.add_argument("--eval_only", action="store_true", help="Skip inference, only evaluate")
    parser.add_argument("--report_only", action="store_true", help="Only print the saved summary report")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip inference if results.npz already exists for that scene/experiment",
    )
    return parser.parse_args()


def _default_lora_root(model_family: str) -> str:
    return DEFAULT_DA3_LORA_ROOT if model_family == "da3" else DEFAULT_VGGT_LORA_ROOT


def _resolve_lora_path(args, dataset_name: str) -> Optional[str]:
    if args.no_lora:
        return None
    if args.lora_path:
        return args.lora_path

    lora_root = args.lora_root or _default_lora_root(args.model_family)
    candidate = os.path.join(lora_root, dataset_name, "lora.pt")
    return candidate if os.path.exists(candidate) else None


def _load_model_for_dataset(args, dataset_name: str, device):
    lora_path = _resolve_lora_path(args, dataset_name)

    if args.model_family == "da3":
        if lora_path:
            print(f"\nLoading DA3 LoRA model for {dataset_name}: {lora_path}")
            return LoRADepthAnything3(base_model=args.model_name, lora_path=lora_path).to(device), lora_path

        from depth_anything_3.api import DepthAnything3

        print(f"\nLoading DA3 base model for {dataset_name}: {args.model_name}")
        return DepthAnything3.from_pretrained(args.model_name).to(device), None

    if lora_path:
        print(f"\nLoading VGGT LoRA model for {dataset_name}: {lora_path}")
        return (
            LoRAVGGT(
                base_model=args.model_name,
                lora_path=lora_path,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_layers_start=args.lora_layers_start,
                image_size=args.image_size,
            ).to(device),
            lora_path,
        )

    print(f"\nLoading VGGT base model for {dataset_name}: {args.model_name}")
    return BaseVGGT(model_name=args.model_name, image_size=args.image_size).to(device), None


def main():
    args = _parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    view_counts = sorted({int(v) for v in args.view_counts})
    experiments = [exp_key(v) for v in view_counts]

    if args.work_dir is None:
        args.work_dir = os.path.join(args.results_root, f"multiview_{args.model_family}_4datasets")

    if args.model_name is None:
        args.model_name = "depth-anything/DA3-GIANT-1.1" if args.model_family == "da3" else "facebook/vggt-1b"

    if args.lora_root is None and not args.no_lora and args.lora_path is None:
        args.lora_root = _default_lora_root(args.model_family)

    fixed_view_indices = None
    if args.view_indices_file:
        if len(view_counts) != 1:
            raise ValueError("--view_indices_file requires exactly one --view_counts value")
        fixed_view_indices = load_view_index_file(args.view_indices_file, 32)

    if args.report_only:
        scenes_by_dataset = {}
        for dataset_name in args.datasets:
            scenes = _scenes_from_existing_metrics(args.work_dir, dataset_name, experiments)
            if scenes:
                scenes_by_dataset[dataset_name] = scenes
        if not scenes_by_dataset:
            raise RuntimeError(
                "No scenes discovered from existing metric files. "
                "Run benchmark/evaluation first or pass the correct --work_dir/--datasets."
            )
        write_summary_report(args.work_dir, scenes_by_dataset, experiments)
        return

    if args.model_family == "da3":
        from depth_anything_3.bench.registries import MV_REGISTRY
    else:
        from vggt.vggt.bench.registries import VGGT_MV_REGISTRY as MV_REGISTRY

    scenes_by_dataset: Dict[str, List[str]] = {}
    datasets = []
    for dataset_name in args.datasets:
        if not MV_REGISTRY.has(dataset_name):
            print(f"[WARN] Unknown dataset: {dataset_name}")
            continue
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = _resolve_scenes(dataset_name, dataset, args)
        if not scenes:
            print(f"[WARN] No scenes selected for dataset={dataset_name}; skipping")
            continue
        datasets.append((dataset_name, dataset))
        scenes_by_dataset[dataset_name] = scenes

    if fixed_view_indices is not None:
        for dataset_name, scenes in list(scenes_by_dataset.items()):
            eligible_scenes = [scene for scene in scenes if (scene, 0) in fixed_view_indices]
            skipped_scenes = sorted(set(scenes) - set(eligible_scenes))
            if skipped_scenes:
                print(f"[WARN] {dataset_name}: skipping scenes absent from fixed P32 file: {skipped_scenes}")
            scenes_by_dataset[dataset_name] = eligible_scenes
        datasets = [(name, dataset) for name, dataset in datasets if scenes_by_dataset[name]]

    if not datasets:
        raise RuntimeError("No valid dataset/scene combination selected.")

    write_manifest(
        os.path.join(args.work_dir, "benchmark_manifest.json"),
        model_family=args.model_family,
        model_name=args.model_name,
        lora_path=args.lora_path,
        datasets=[d for d, _ in datasets],
        view_counts=view_counts,
        scenes_by_dataset=scenes_by_dataset,
    )

    if args.eval_only:
        run_evaluation(
            args.model_family,
            args.work_dir,
            scenes_by_dataset,
            experiments,
            num_fusion_workers=args.num_fusion_workers,
        )
        write_summary_report(args.work_dir, scenes_by_dataset, experiments)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name, dataset in datasets:
        model, resolved_lora_path = _load_model_for_dataset(args, dataset_name, device)
        scenes = scenes_by_dataset[dataset_name]
        print(f"\n{'#' * 72}")
        print(f"Dataset: {dataset_name} ({len(scenes)} scenes)")
        if resolved_lora_path:
            print(f"LoRA path: {resolved_lora_path}")
        else:
            print("LoRA path: base model only")
        print(f"{'#' * 72}")

        for scene in scenes:
            try:
                scene_data = dataset.get_data(scene)
                ordered_image_files, ordered_extrinsics, ordered_intrinsics = _canonical_scene_views(scene_data)
                num_frames = len(ordered_image_files)
                print(f"\nScene {scene}: {num_frames} total frames")

                for view_count in view_counts:
                    experiment = exp_key(view_count)
                    selected_indices = sample_view_indices(
                        num_frames=num_frames,
                        target_views=view_count,
                        dataset_name=dataset_name,
                        scene=scene,
                        seed=args.seed,
                    )
                    if fixed_view_indices is not None:
                        try:
                            selected_indices = fixed_view_indices[(scene, 0)]
                        except KeyError as exc:
                            raise KeyError(f"No fixed evaluation pool for scene={scene!r}, sample_idx=0") from exc
                        if max(selected_indices) >= num_frames:
                            raise ValueError(f"Fixed evaluation pool references unavailable frame in {scene}")
                        if view_count < len(selected_indices):
                            selected_indices = [
                                selected_indices[index * len(selected_indices) // view_count]
                                for index in range(view_count)
                            ]
                    if not selected_indices:
                        print(f"  [{experiment}] Skip: no sampled indices available")
                        continue

                    image_files = [ordered_image_files[i] for i in selected_indices]
                    result_npz = _results_npz_path(args.work_dir, experiment, dataset_name, scene)
                    meta_path = _gt_meta_path(args.work_dir, experiment, dataset_name, scene)

                    print(
                        f"  [{experiment}] requested={view_count}, actual={len(selected_indices)}, "
                        f"indices={selected_indices}"
                    )

                    if args.skip_existing and os.path.exists(result_npz) and os.path.exists(meta_path):
                        print(f"  [{experiment}] Skipping existing outputs")
                        continue

                    if args.model_family == "da3":
                        run_da3_inference(
                            model,
                            image_files,
                            SimpleNamespace(extrinsics=ordered_extrinsics, intrinsics=ordered_intrinsics),
                            selected_indices,
                            args.work_dir,
                            experiment,
                            dataset_name,
                            scene,
                        )
                    else:
                        run_vggt_inference(
                            model,
                            image_files,
                            SimpleNamespace(extrinsics=ordered_extrinsics, intrinsics=ordered_intrinsics),
                            selected_indices,
                            args.work_dir,
                            experiment,
                            dataset_name,
                            scene,
                        )

                    print(f"  [{experiment}] Saved predictions and visualizations")
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[ERROR] {dataset_name}/{scene} failed:\n{e}")
                import traceback

                traceback.print_exc()
                continue

        del model
        torch.cuda.empty_cache()

    run_evaluation(
        args.model_family,
        args.work_dir,
        scenes_by_dataset,
        experiments,
        num_fusion_workers=args.num_fusion_workers,
    )
    write_summary_report(args.work_dir, scenes_by_dataset, experiments)


if __name__ == "__main__":
    main()
