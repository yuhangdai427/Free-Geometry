#!/usr/bin/env python3
"""Compute per-scene, depth-verified co-visibility matrices on a GPU.

This is deliberately independent of the benchmark dataset loaders.  It reads
the released depth/pose files directly, so it can also run when RGB images are
not present (as is the case for the ScanNet++ depth-only checkout).
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from depth_anything_3.utils.constants import (  # noqa: E402
    ETH3D_FILTER_KEYS,
    ETH3D_SCENES,
    HIROOM_SCENE_LIST_PATH,
    SCANNETPP_SCENES,
    SEVENSCENES_SCENES,
)
from depth_anything_3.utils.read_write_model import read_model  # noqa: E402
from depth_anything_3.bench.utils import quat2rotmat  # noqa: E402


DEFAULT_ROOTS = {
    "7scenes": REPO_ROOT / "workspace/benchmark_dataset/7scenes",
    "scannetpp": REPO_ROOT / "workspace/benchmark_dataset/scannetpp",
    "eth3d": REPO_ROOT / "workspace/benchmark_dataset/eth3d",
    "hiroom": REPO_ROOT / "workspace/benchmark_dataset/hiroom",
}
DEFAULT_STRIDES = {"7scenes": 10, "scannetpp": 5, "eth3d": 1, "hiroom": 1}
DEFAULT_THRESHOLDS = {"7scenes": 0.10, "scannetpp": 0.25, "eth3d": 0.025, "hiroom": 0.10}


@dataclass
class Frame:
    identifier: str
    depth_path: Path
    w2c: np.ndarray
    intrinsic: np.ndarray
    aliasing_path: Optional[Path] = None
    image_shape: Optional[Tuple[int, int]] = None


def _w2c_from_colmap(image) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = image.qvec2rotmat()
    result[:3, 3] = image.tvec
    return result


def _intrinsic_from_camera(camera) -> np.ndarray:
    p = camera.params
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = p[0]
        cx, cy = p[1:3]
    else:
        fx, fy, cx, cy = p[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def seven_scenes_frames(root: Path, scene: str) -> List[Frame]:
    sequence = "seq-02" if scene == "stairs" else "seq-01"
    directory = root / "7Scenes" / scene / sequence
    intrinsic = np.array([[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    frames = []
    for pose_path in sorted(directory.glob("*.pose.txt")):
        stem = pose_path.name.replace(".pose.txt", "")
        depth_path = directory / f"{stem}.depth.png"
        if depth_path.exists():
            frames.append(Frame(stem, depth_path, np.linalg.inv(np.loadtxt(pose_path)).astype(np.float32), intrinsic.copy()))
    return frames


def scannetpp_frames(root: Path, scene: str) -> List[Frame]:
    base = root / scene / "merge_dslr_iphone"
    cameras, images, _ = read_model(base / "colmap/sparse_render_rgb")
    records = []
    for image in images.values():
        if not image.name.startswith("iphone/"):
            continue
        frame = Path(image.name).stem
        depth = base / "render_depth" / f"{frame}.png"
        if not depth.exists():
            continue
        camera = cameras[image.camera_id]
        if camera.model != "OPENCV":
            raise ValueError(f"{scene}/{frame}: expected OPENCV camera, got {camera.model}")
        records.append((frame, depth, _w2c_from_colmap(image), _intrinsic_from_camera(camera), camera.params[4:8]))
    # The names encode capture time; retain temporal order before stride sampling.
    return [Frame(name, depth, pose, intrinsic, None) for name, depth, pose, intrinsic, _ in sorted(records)]


def eth3d_frames(root: Path, scene: str) -> List[Frame]:
    calibration = root / scene / "dslr_calibration_jpg"
    camera_params: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
    for line in (calibration / "cameras.txt").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        intrinsic = np.array([[float(parts[4]), 0, float(parts[6])], [0, float(parts[5]), float(parts[7])], [0, 0, 1]], dtype=np.float32)
        camera_params[parts[0]] = (intrinsic, (int(parts[3]), int(parts[2])))
    lines = [line.strip() for line in (calibration / "images.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    frames = []
    for line in lines[::2]:
        p = line.split()
        # COLMAP names include "dslr_images/" while the local image/depth
        # directories contain only the basename.
        name, camera_id = Path(p[9]).name, p[8]
        if any(name.endswith(key) for key in ETH3D_FILTER_KEYS.get(scene, [])):
            continue
        depth = root / scene / "ground_truth_depth" / "dslr_images" / name
        if not depth.exists():
            continue
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = quat2rotmat([float(x) for x in p[1:5]])
        pose[:3, 3] = [float(x) for x in p[5:8]]
        intrinsic, image_shape = camera_params[camera_id]
        frames.append(Frame(name, depth, pose, intrinsic.copy(), image_shape=image_shape))
    return frames


def hiroom_frames(root: Path, scene: str) -> List[Frame]:
    base = root / "data" / scene
    intrinsic = np.load(base / "cam_K.npy").astype(np.float32)
    frames = []
    for image in sorted((base / "image").iterdir()):
        name = image.stem
        depth, pose = base / "depth" / f"{name}.png", base / "pose" / f"{name}.npy"
        if depth.exists() and pose.exists():
            frames.append(Frame(name, depth, np.load(pose).astype(np.float32), intrinsic.copy(), base / "aliasing_mask" / f"{name}.png"))
    return frames


def get_frames(dataset: str, root: Path, scene: str) -> List[Frame]:
    return {"7scenes": seven_scenes_frames, "scannetpp": scannetpp_frames, "eth3d": eth3d_frames, "hiroom": hiroom_frames}[dataset](root, scene)


def read_depth(frame: Frame, dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return metric depth, validity mask and an intrinsic in matching pixels."""
    if dataset == "eth3d":
        # ETH3D stores native float32 maps without a PNG header.
        if frame.image_shape is None:
            raise ValueError(f"ETH3D frame {frame.identifier} lacks calibrated image shape")
        height, width = frame.image_shape
        raw = np.fromfile(frame.depth_path, dtype=np.float32)
        if raw.size != height * width:
            raise ValueError(f"{frame.depth_path}: {raw.size} values, expected {height}x{width}")
        depth = raw.reshape(height, width)
    else:
        raw = iio.imread(frame.depth_path)
        if dataset == "7scenes":
            depth = raw.astype(np.float32) / 1000.0
            depth[raw == 65535] = 0.0
        elif dataset == "scannetpp":
            depth = raw.astype(np.float32) / 1000.0
        else:
            depth = raw.astype(np.float32) / 65535.0 * 100.0
    valid = np.isfinite(depth) & (depth > 0)
    if dataset == "hiroom" and frame.aliasing_path is not None and frame.aliasing_path.exists():
        valid &= iio.imread(frame.aliasing_path) == 0
    return depth.astype(np.float32), valid, frame.intrinsic.copy()


def resize_frame(depth: np.ndarray, valid: np.ndarray, intrinsic: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    scale = min(1.0, float(max_side) / max(h, w))
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    intrinsic = intrinsic.copy()
    intrinsic[0, :] *= new_w / w
    intrinsic[1, :] *= new_h / h
    intrinsic[2, 2] = 1.0
    return depth, valid, intrinsic


@torch.inference_mode()
def covisibility_row(source_depth: torch.Tensor, source_valid: torch.Tensor, source_k: torch.Tensor, source_w2c: torch.Tensor, target_depths: torch.Tensor, target_valids: torch.Tensor, target_ks: torch.Tensor, target_w2cs: torch.Tensor) -> torch.Tensor:
    """Return C_i,j for one source and a batch of target cameras on one device."""
    h, w = source_depth.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=source_depth.device), torch.arange(w, device=source_depth.device), indexing="ij")
    pixels = torch.stack((xx.reshape(-1), yy.reshape(-1), torch.ones(h * w, device=source_depth.device)))
    z = source_depth.reshape(-1)
    source_points = torch.linalg.solve(source_k, pixels) * z
    world = torch.linalg.inv(source_w2c) @ torch.cat((source_points, torch.ones(1, h * w, device=source_depth.device)))
    cameras = target_w2cs @ world.unsqueeze(0)
    projected_z = cameras[:, 2]
    uv = torch.bmm(target_ks, cameras[:, :3])
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    inside = (projected_z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    grid = torch.stack((2 * u / max(w - 1, 1) - 1, 2 * v / max(h - 1, 1) - 1), dim=-1).view(len(target_depths), h, w, 2)
    sampled_depth = F.grid_sample(target_depths[:, None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0].reshape(len(target_depths), -1)
    sampled_valid = F.grid_sample(target_valids[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0].reshape(len(target_depths), -1) > 0.5
    consistent = (sampled_depth - projected_z).abs() < (0.1693 + 0.005 * projected_z)
    matched = inside & sampled_valid & consistent & source_valid.reshape(1, -1)
    return matched.float().sum(dim=1) / float(h * w)


def compute_matrix(depths: np.ndarray, valids: np.ndarray, intrinsics: np.ndarray, w2cs: np.ndarray, device: str, target_batch_size: int) -> np.ndarray:
    dev = torch.device(device)
    target_depths = torch.from_numpy(depths).to(dev)
    target_valids = torch.from_numpy(valids).to(dev)
    target_ks = torch.from_numpy(intrinsics).to(dev)
    target_w2cs = torch.from_numpy(w2cs).to(dev)
    n = len(depths)
    matrix = np.zeros((n, n), dtype=np.float32)
    for source_index in range(n):
        for start in range(0, n, target_batch_size):
            end = min(n, start + target_batch_size)
            matrix[source_index, start:end] = covisibility_row(target_depths[source_index], target_valids[source_index], target_ks[source_index], target_w2cs[source_index], target_depths[start:end], target_valids[start:end], target_ks[start:end], target_w2cs[start:end]).cpu().numpy()
    return matrix


def scene_names(dataset: str, root: Path) -> Sequence[str]:
    if dataset == "7scenes": return SEVENSCENES_SCENES
    if dataset == "scannetpp": return SCANNETPP_SCENES
    if dataset == "eth3d": return ETH3D_SCENES
    return [line.strip() for line in (root / "selected_scene_list_val.txt").read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_ROOTS, default=list(DEFAULT_ROOTS))
    parser.add_argument("--scenes", nargs="*", help="Optional scene names; only applies to a single dataset.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts/covisibility")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-side", type=int, default=224)
    parser.add_argument("--stride", type=int, default=None, help="Override the dataset default candidate-frame stride; use 1 for every valid frame.")
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None, help="Override the dataset-specific adjacency threshold.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenes and len(args.datasets) != 1:
        raise ValueError("--scenes requires exactly one dataset")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError(f"CUDA is unavailable; requested {args.device}")
    if args.stride is not None and args.stride < 1:
        raise ValueError("--stride must be at least 1")
    for dataset in args.datasets:
        root = DEFAULT_ROOTS[dataset]
        names = args.scenes or scene_names(dataset, root)
        threshold = DEFAULT_THRESHOLDS[dataset] if args.threshold is None else args.threshold
        for scene in names:
            output = args.output_root / dataset / f"{scene.replace('/', '__')}.npz"
            if args.resume and output.exists():
                print(f"[skip] {dataset}/{scene}: {output}")
                continue
            start = time.monotonic()
            frames = get_frames(dataset, root, scene)
            stride = DEFAULT_STRIDES[dataset] if args.stride is None else args.stride
            candidate_indices = np.arange(0, len(frames), stride, dtype=np.int32)
            frames = [frames[i] for i in candidate_indices]
            if not frames:
                print(f"[skip] {dataset}/{scene}: no valid frames")
                continue
            prepared = [resize_frame(*read_depth(frame, dataset), args.max_side) for frame in frames]
            shapes = {depth.shape for depth, _, _ in prepared}
            if len(shapes) != 1:
                raise ValueError(f"{dataset}/{scene}: mixed resized shapes {shapes}; use a fixed resize policy")
            depths, valids, intrinsics = (np.stack(values) for values in zip(*prepared))
            w2cs = np.stack([frame.w2c for frame in frames])
            raw = compute_matrix(depths, valids, intrinsics, w2cs, args.device, args.target_batch_size)
            directional = raw / (np.diag(raw)[:, None] + 1e-8)
            # The maximum makes the score truly symmetric and conservative:
            # neither view may substantially see the other for a selected pair.
            symmetric = np.maximum(directional, directional.T)
            adjacency = symmetric >= threshold
            np.fill_diagonal(adjacency, True)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, raw_covisibility=raw, max_directional_overlap=symmetric.astype(np.float32), adjacency=adjacency, candidate_indices=candidate_indices, frame_ids=np.asarray([f.identifier for f in frames]), depth_paths=np.asarray([str(f.depth_path) for f in frames]), intrinsics=intrinsics, extrinsics_w2c=w2cs, dataset=dataset, scene=scene, stride=stride, max_side=args.max_side, depth_tolerance_base=0.1693, depth_tolerance_slope=0.005, adjacency_threshold=threshold)
            print(f"[done] {dataset}/{scene}: {len(frames)} candidates, {raw.shape}, {time.monotonic() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
