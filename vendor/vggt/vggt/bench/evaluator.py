"""
VGGT Evaluator for Benchmark Evaluation.

Adapted from DA3 Evaluator to handle VGGT's inference API and output format.
Reuses generic utilities from DA3 for pose and reconstruction evaluation.
"""

import json
import os
import random
import time
from typing import Dict as TDict, Iterable, List, Optional

import numpy as np
import torch
from addict import Dict
from tqdm import tqdm

# Reuse generic utilities from DA3
from depth_anything_3.bench.print_metrics import MetricsPrinter
from depth_anything_3.bench.utils import compute_pose, evaluate_3d_reconstruction
from depth_anything_3.utils.parallel_utils import parallel_execution
from depth_anything_3.utils.geometry import as_homogeneous

from .registries import VGGT_MV_REGISTRY


class VGGTEvaluator:
    """
    Evaluation orchestrator for VGGT benchmarks.

    Supports multiple datasets and evaluation modes:
    - pose: Camera pose estimation (AUC metrics)
    - recon_unposed: 3D reconstruction with predicted poses
    - recon_posed: 3D reconstruction with GT poses

    Usage:
        evaluator = VGGTEvaluator(
            work_dir="./eval_workspace",
            datas=["eth3d"],
            modes=["pose", "recon_unposed"],
        )
        api = LoRAVGGT(...)
        evaluator.infer(api)
        metrics = evaluator.eval()
        evaluator.print_metrics()
    """

    VALID_MODES = {"pose", "recon_unposed", "recon_posed"}

    def __init__(
        self,
        work_dir: str = "./eval_workspace",
        datas: List[str] = ("eth3d",),
        modes: List[str] = ("pose",),
        scenes: List[str] = None,
        debug: bool = False,
        num_fusion_workers: int = 4,
        max_frames: int = 100,
        image_size: int = 518,
        seed: int = 42,
        eval_frames: int = None,
        subset_sampling: bool = False,
        subset_ratio: float = 0.1,
    ):
        """
        Initialize the VGGT evaluator.

        Args:
            work_dir: Base directory for model outputs and metric files
            datas: List of dataset names (must be registered in VGGT_MV_REGISTRY)
            modes: List of evaluation modes to run
            scenes: Specific scenes to evaluate (None = all scenes)
            debug: Enable verbose debug output
            num_fusion_workers: Number of parallel workers for TSDF fusion
            max_frames: Maximum number of frames per scene
            image_size: VGGT input image size (default: 518)
            seed: Random seed for frame sampling (default: 42)
            eval_frames: If set, use even-indexed frames from max_frames sample
            subset_sampling: If True, use consecutive window sampling from subset
            subset_ratio: Ratio of frames to include in subset (default: 0.1 = 10%)
        """
        self.work_dir = work_dir
        self.datas = list(datas)
        self.modes = set(modes)
        self.scenes_filter = scenes
        self.debug = debug
        self.num_fusion_workers = num_fusion_workers
        self.max_frames = max_frames
        self.image_size = image_size
        self.seed = seed
        self.eval_frames = eval_frames
        self.subset_sampling = subset_sampling
        self.subset_ratio = subset_ratio

        # Validate modes
        unknown = self.modes - self.VALID_MODES
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}. Valid: {sorted(self.VALID_MODES)}")

        os.makedirs(self.work_dir, exist_ok=True)

        # Initialize datasets
        self.datasets = Dict()
        for data in self.datas:
            if not VGGT_MV_REGISTRY.has(data):
                available = list(VGGT_MV_REGISTRY.all().keys())
                raise ValueError(f"Dataset '{data}' not found. Available: {available}")
            self.datasets[data] = VGGT_MV_REGISTRY.get(data)()

        # Initialize metrics printer
        self._printer = MetricsPrinter()

    def _get_scenes(self, dataset) -> List[str]:
        """Get list of scenes to evaluate, optionally filtered."""
        all_scenes = dataset.SCENES
        if self.scenes_filter:
            scenes = [s for s in all_scenes if s in self.scenes_filter]
            return scenes
        return all_scenes

    def infer(self, api) -> None:
        """
        Run VGGT inference on all scenes.

        Args:
            api: VGGT API instance with inference() method
        """
        need_unposed = {"pose", "recon_unposed"} & self.modes
        need_posed = {"recon_posed"} & self.modes

        # Collect all tasks
        all_tasks = []
        for data in self.datas:
            dataset = self.datasets[data]
            for scene in self._get_scenes(dataset):
                all_tasks.append((data, scene))

        print(f"[INFO] Total inference tasks: {len(all_tasks)}")

        # Timing and memory tracking
        scene_times = {}
        total_start = time.time()
        peak_memory_mb = 0.0

        for data, scene in tqdm(all_tasks, desc="Inference"):
            dataset = self.datasets[data]
            scene_data = dataset.get_data(scene)
            scene_data = self._sample_frames(scene_data, scene)

            scene_start = time.time()

            if need_unposed:
                export_dir = self._export_dir(data, scene, posed=False)
                self._run_inference(api, scene_data, export_dir, use_gt_poses=False)
                self._save_gt_meta(export_dir, scene_data)

            if need_posed:
                export_dir = self._export_dir(data, scene, posed=True)
                self._run_inference(api, scene_data, export_dir, use_gt_poses=True)
                self._save_gt_meta(export_dir, scene_data)

            scene_time = time.time() - scene_start
            scene_times[f"{data}/{scene}"] = scene_time

            # Track peak GPU memory
            if torch.cuda.is_available():
                current_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        total_time = time.time() - total_start

        # Save timing and memory stats
        timing_file = os.path.join(self.work_dir, "timing_stats.json")
        timing_stats = {
            "total_time_seconds": total_time,
            "peak_memory_mb": peak_memory_mb,
            "scene_times": scene_times,
            "num_scenes": len(scene_times),
            "avg_time_per_scene": total_time / len(scene_times) if scene_times else 0,
        }
        os.makedirs(os.path.dirname(timing_file), exist_ok=True)
        with open(timing_file, 'w') as f:
            json.dump(timing_stats, f, indent=2)

        print(f"\n[TIMING] Total: {total_time:.2f}s, Avg per scene: {timing_stats['avg_time_per_scene']:.2f}s")
        print(f"[MEMORY] Peak GPU memory: {peak_memory_mb:.2f} MB")
        print(f"[TIMING] Stats saved to: {timing_file}")

    def _run_inference(
        self,
        api,
        scene_data: Dict,
        export_dir: str,
        use_gt_poses: bool = False,
    ) -> None:
        """
        Run VGGT inference and save results.

        Args:
            api: VGGT API instance
            scene_data: Scene data with image_files, extrinsics, intrinsics
            export_dir: Directory to save results
            use_gt_poses: Whether to use GT poses (for recon_posed mode)
        """
        # Run inference
        if use_gt_poses:
            predictions = api.inference(
                scene_data.image_files,
                extrinsics=scene_data.extrinsics,
                intrinsics=scene_data.intrinsics,
            )
        else:
            predictions = api.inference(scene_data.image_files)

        # Save results to npz
        results_dir = os.path.join(export_dir, "exports", "mini_npz")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.npz")

        # Convert predictions to numpy and save
        save_dict = {}

        # Extrinsics and intrinsics (from pose_enc or direct)
        if "extrinsics" in predictions:
            save_dict["extrinsics"] = predictions["extrinsics"].cpu().numpy()
        if "intrinsics" in predictions:
            save_dict["intrinsics"] = predictions["intrinsics"].cpu().numpy()

        # Fallback: convert pose_enc if extrinsics not already set
        if "extrinsics" not in save_dict and "pose_enc" in predictions:
            # Convert pose_enc to extrinsics using VGGT utility
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(self.image_size, self.image_size),
                pose_encoding_type="absT_quaR_FoV",
            )
            save_dict["extrinsics"] = extrinsics.squeeze(0).cpu().numpy()
            save_dict["intrinsics"] = intrinsics.squeeze(0).cpu().numpy()

        # Depth maps
        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 5:  # [B, S, H, W, 1]
                depth = depth.squeeze(0).squeeze(-1)  # [S, H, W]
            elif depth.dim() == 4:  # [B, S, H, W] or [S, H, W, 1]
                if depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                else:
                    depth = depth.squeeze(0)
            save_dict["depth"] = depth.cpu().numpy()

        # Depth confidence
        if "depth_conf" in predictions:
            depth_conf = predictions["depth_conf"]
            if depth_conf.dim() == 4:
                depth_conf = depth_conf.squeeze(0)
            save_dict["depth_conf"] = depth_conf.cpu().numpy()

        # World points (3D reconstruction)
        if "world_points" in predictions:
            world_points = predictions["world_points"]
            if world_points.dim() == 5:  # [B, S, H, W, 3]
                world_points = world_points.squeeze(0)  # [S, H, W, 3]
            save_dict["world_points"] = world_points.cpu().numpy()

        # World points confidence
        if "world_points_conf" in predictions:
            world_points_conf = predictions["world_points_conf"]
            if world_points_conf.dim() == 4:
                world_points_conf = world_points_conf.squeeze(0)
            save_dict["world_points_conf"] = world_points_conf.cpu().numpy()

        np.savez_compressed(results_path, **save_dict)

    def eval(self) -> TDict[str, dict]:
        """
        Evaluate for all configured modes.

        Returns:
            Summary mapping: {"<data>_<mode>": metrics_dict}
        """
        summary: TDict[str, dict] = {}

        if "pose" in self.modes:
            print(f"\n{'='*60}")
            print(f"Evaluating POSE for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_pose():
                summary[f"{data}_pose"] = result

        if "recon_unposed" in self.modes:
            print(f"\n{'='*60}")
            print(f"Evaluating RECON_UNPOSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_unposed"):
                summary[f"{data}_recon_unposed"] = result

        if "recon_posed" in self.modes:
            print(f"\n{'='*60}")
            print(f"Evaluating RECON_POSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_posed"):
                summary[f"{data}_recon_posed"] = result

        return summary

    def _eval_pose(self) -> Iterable[tuple]:
        """Compute pose-estimation metrics for each dataset and scene."""
        os.makedirs(self._metric_dir, exist_ok=True)

        for data in tqdm(self.datas, desc="Datasets (pose eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            for scene in tqdm(scenes, desc=f"{data} scenes", leave=False):
                export_dir = self._export_dir(data, scene, posed=False)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")

                if not os.path.exists(result_path):
                    print(f"\n[ERROR] Result file not found: {result_path}")
                    continue

                try:
                    gt_meta = self._load_gt_meta(export_dir)
                    if gt_meta is not None:
                        result = self._compute_pose_with_gt(result_path, gt_meta)
                    else:
                        result = dataset.eval_pose(scene, result_path)
                    dataset_results[scene] = self._to_float_dict(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to evaluate pose for {data}/{scene}: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            if not dataset_results:
                print(f"[WARNING] No valid results for {data}")
                continue

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_pose.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    def _eval_reconstruction(self, mode: str) -> Iterable[tuple]:
        """Compute reconstruction metrics for each dataset and scene."""
        assert mode in {"recon_unposed", "recon_posed"}
        os.makedirs(self._metric_dir, exist_ok=True)

        posed_flag = mode == "recon_posed"

        for data in tqdm(self.datas, desc=f"Datasets ({mode} eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            # Prepare paths for all scenes
            scene_list = []
            result_paths = []
            fuse_paths = []
            for scene in scenes:
                export_dir = self._export_dir(data, scene, posed=posed_flag)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
                fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
                scene_list.append(scene)
                result_paths.append(result_path)
                fuse_paths.append(fuse_path)

            # Parallel fusion
            parallel_execution(
                scene_list,
                result_paths,
                fuse_paths,
                action=lambda s, rp, fp: dataset.fuse3d(s, rp, fp, mode),
                num_processes=self.num_fusion_workers,
                print_progress=True,
                desc=f"{data} fusion",
            )

            # Sequential evaluation
            for scene, fuse_path in zip(scene_list, fuse_paths):
                result = dataset.eval3d(scene, fuse_path)
                dataset_results[scene] = self._to_float_dict(result)
                print(f"  {mode} | {data} | {scene}: {result}")

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    def print_metrics(self, metrics: TDict[str, dict] = None) -> None:
        """Print evaluation metrics in a tabular format."""
        if metrics is None:
            metrics = self._load_metrics()
        self._printer.print_results(metrics)

    # -------------------- Helpers -------------------- #

    def _save_gt_meta(self, export_dir: str, scene_data: Dict) -> None:
        """Save GT extrinsics/intrinsics for evaluation."""
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.savez_compressed(
            meta_path,
            extrinsics=scene_data.extrinsics,
            intrinsics=scene_data.intrinsics,
            image_files=np.array(scene_data.image_files, dtype=object),
        )

    def _load_gt_meta(self, export_dir: str) -> Optional[Dict]:
        """Load saved GT extrinsics/intrinsics for evaluation."""
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if os.path.exists(meta_path):
            data = np.load(meta_path)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
            })
        return None

    def _compute_pose_with_gt(self, result_path: str, gt_meta: Dict) -> TDict[str, float]:
        """Compute pose metrics using saved GT meta."""
        pred = np.load(result_path)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
        )

    def _sample_frames(self, scene_data: Dict, scene: str) -> Dict:
        """Sample frames if scene has more than max_frames.

        If subset_sampling is enabled, uses consecutive window sampling:
        1. First uniformly sample subset_ratio of frames
        2. Then take consecutive window from subset based on seed
        """
        if self.max_frames <= 0:
            return scene_data

        num_frames = len(scene_data.image_files)

        # Subset sampling: consecutive windows from uniformly sampled subset
        if self.subset_sampling:
            # Get subset indices (uniform sampling)
            subset_size = max(self.max_frames, int(num_frames * self.subset_ratio))
            if subset_size >= num_frames:
                subset_indices = list(range(num_frames))
            else:
                step = num_frames / subset_size
                subset_indices = [int(i * step) for i in range(subset_size)]

            # Use seed to determine which consecutive window to use
            num_full_windows = len(subset_indices) // self.max_frames
            window_idx = self.seed % max(1, num_full_windows * 2)  # Allow offset windows too

            if len(subset_indices) <= self.max_frames:
                # Subset too small, use all
                sampled_indices = subset_indices
            elif window_idx < num_full_windows:
                # Non-overlapping window
                start = window_idx * self.max_frames
                sampled_indices = subset_indices[start:start + self.max_frames]
            else:
                # Offset window (wrap around)
                offset = self.max_frames // 2
                wrap_idx = window_idx - num_full_windows
                max_start = len(subset_indices) - self.max_frames
                start = (offset + wrap_idx * self.max_frames) % (max_start + 1)
                sampled_indices = subset_indices[start:start + self.max_frames]

            print(f"  [Subset Sampling] {scene}: {num_frames} -> subset {len(subset_indices)} -> window {len(sampled_indices)} frames (seed={self.seed})")

            # Apply eval_frames if set (take even-indexed from window)
            if self.eval_frames and self.eval_frames < len(sampled_indices):
                even_indices = [sampled_indices[i] for i in range(0, len(sampled_indices), 2)]
                sampled_indices = even_indices[:self.eval_frames]
                print(f"    -> {self.eval_frames} frames (even-indexed)")
        else:
            # Original random sampling logic
            if num_frames <= self.max_frames:
                # If eval_frames is set and we have enough frames, still apply even-indexing
                if self.eval_frames and self.eval_frames < num_frames:
                    random.seed(self.seed)
                    indices = list(range(num_frames))
                    random.shuffle(indices)
                    sampled_indices = sorted(indices[:self.max_frames])
                    # Take even-indexed frames
                    even_indices = [sampled_indices[i] for i in range(0, len(sampled_indices), 2)]
                    sampled_indices = even_indices[:self.eval_frames]
                    print(f"  [Sampling] {scene}: {num_frames} -> {self.eval_frames} frames (even-indexed, seed={self.seed})")
                else:
                    return scene_data
            else:
                random.seed(self.seed)
                indices = list(range(num_frames))
                random.shuffle(indices)
                sampled_indices = sorted(indices[:self.max_frames])

                # If eval_frames is set, take even-indexed frames from the sampled set
                if self.eval_frames and self.eval_frames < self.max_frames:
                    even_indices = [sampled_indices[i] for i in range(0, len(sampled_indices), 2)]
                    sampled_indices = even_indices[:self.eval_frames]
                    print(f"  [Sampling] {scene}: {num_frames} -> {self.max_frames} -> {self.eval_frames} frames (even-indexed, seed={self.seed})")
                else:
                    print(f"  [Sampling] {scene}: {num_frames} -> {self.max_frames} frames (seed={self.seed})")

        print(f"    indices: {sampled_indices}")

        sampled = Dict()
        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]

        sampled.aux = Dict()
        for key, val in scene_data.aux.items():
            if isinstance(val, list) and len(val) == num_frames:
                sampled.aux[key] = [val[i] for i in sampled_indices]
            elif isinstance(val, np.ndarray) and len(val) == num_frames:
                sampled.aux[key] = val[sampled_indices]
            else:
                sampled.aux[key] = val

        return sampled

    @property
    def _metric_dir(self) -> str:
        """Directory for storing metric JSON files."""
        return os.path.join(self.work_dir, "metric_results")

    def _export_dir(self, data: str, scene: str, posed: bool) -> str:
        """Get export directory path."""
        suffix = "posed" if posed else "unposed"
        export_dir = os.path.join(self.work_dir, "model_results", data, scene, suffix)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    @staticmethod
    def _to_float_dict(d: TDict[str, float]) -> dict:
        """Convert numpy scalars to plain Python floats."""
        return {k: float(v) for k, v in d.items()}

    @staticmethod
    def _mean_of_dicts(dicts: Iterable[dict]) -> dict:
        """Compute finite-only elementwise means without hiding scene metrics."""
        dicts = list(dicts)
        if not dicts:
            return {}
        keys = dicts[0].keys()
        means = {}
        for key in keys:
            values = [float(item[key]) for item in dicts if np.isfinite(item[key])]
            if len(values) != len(dicts):
                print(f"[WARNING] Excluding {len(dicts) - len(values)} non-finite {key} scene metric(s) from mean")
            means[key] = float(np.mean(values).item()) if values else float("nan")
        return means

    @staticmethod
    def _dump_json(path: str, obj: dict, indent: int = 4) -> None:
        """Write JSON with UTF-8 and pretty indentation."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    def _load_metrics(self) -> TDict[str, dict]:
        """Load evaluation metrics from JSON files."""
        metrics = {}
        metric_dir = self._metric_dir

        if not os.path.exists(metric_dir):
            return metrics

        for filename in os.listdir(metric_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metric_dir, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    key = filename[:-5]
                    metrics[key] = data
                except Exception as e:
                    print(f"Warning: Failed to read metrics file: {filename} - {e}")

        return metrics
