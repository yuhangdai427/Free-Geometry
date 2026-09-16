#!/usr/bin/env python3
"""Build textureless and occlusion-hard 8V manifests from real RGB/depth/poses.

The script is independent of benchmark loaders.  It reads the full-frame
Co-visibility artifacts as the canonical depth/pose ordering, produces a
scene-balanced Q4 label, and writes normal VGGT 8V/4V manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "vggt")]
from vggt.vggt.test_time_adaption.covisibility_sequences import _image_from_depth


STUDENT = [0, 2, 4, 6]
MATRIX_ROOTS = {
    "eth3d": ROOT / "artifacts/covisibility_eth3d_all/eth3d",
    "scannetpp": ROOT / "artifacts/covisibility_scannetpp_all_frames/scannetpp",
}


@dataclass
class Scene:
    dataset: str
    name: str
    ids: List[str]
    images: List[str]
    depths: np.ndarray
    valids: np.ndarray
    intrinsics: np.ndarray
    w2cs: np.ndarray


def stable_seed(*parts: object, seed: int) -> int:
    value = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def load_scenes(dataset: str, need_geometry: bool, selected: Sequence[str] | None = None) -> List[Scene]:
    scenes = []
    for path in sorted(MATRIX_ROOTS[dataset].glob("*.npz")):
        if selected is not None and path.stem not in selected:
            continue
        with np.load(path, allow_pickle=False) as data:
            ids = [str(item) for item in data["frame_ids"]]
            images = [_image_from_depth(dataset, str(item)) for item in data["depth_paths"]]
            missing = [path for path in images if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(f"{dataset}/{path.stem}: missing RGB {missing[0]}")
            depths = valids = None
            if need_geometry:
                # Depth paths are native resolution while saved intrinsics use
                # the matrix script's max_side grid. Reapply that resize.
                raw_depths = [read_depth(dataset, str(depth_path), identifier)
                              for depth_path, identifier in zip(data["depth_paths"], ids)]
                max_side = int(data["max_side"])
                h0, w0 = raw_depths[0].shape
                scale = min(1.0, max_side / max(h0, w0))
                out_h, out_w = max(1, round(h0 * scale)), max(1, round(w0 * scale))
                depths = np.stack([cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                                   for depth in raw_depths]).astype(np.float32)
                valids = np.isfinite(depths) & (depths > 0)
            scenes.append(Scene(dataset, str(data["scene"].item()), ids, images,
                                depths, valids, np.asarray(data["intrinsics"], dtype=np.float32),
                                np.asarray(data["extrinsics_w2c"], dtype=np.float32)))
            if need_geometry:
                print(f"[loaded] {dataset}/{path.stem}: {len(ids)} depths at {depths.shape[-2:]} ", flush=True)
    return scenes


def read_depth(dataset: str, path: str, identifier: str) -> np.ndarray:
    depth_path = Path(path)
    if dataset == "eth3d":
        image_path = Path(_image_from_depth(dataset, path))
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)
        raw = np.fromfile(depth_path, dtype=np.float32)
        expected = image.shape[0] * image.shape[1]
        if raw.size != expected:
            raise ValueError(f"{identifier}: ETH3D depth length {raw.size}, expected {expected}")
        return raw.reshape(image.shape[:2])
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_path)
    return raw.astype(np.float32) / 1000.0


def texture_raw_scores(paths: Sequence[str], device: str, image_size: int = 504) -> tuple[List[np.ndarray], List[float]]:
    """Compute Sobel patch scores and Laplacian variance in one CUDA batch.

    RGB decoding/resizing remains CPU-bound; the gradient, 14x14 patch pooling,
    and robustness metric run on the requested GPU, typically GPU0.
    """
    arrays = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = image_size / max(h, w)
        new_h, new_w = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
        arrays.append(cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA))
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        # Mixed camera resolutions are uncommon here; retain a correct GPU path.
        outputs, laps = [], []
        for path in paths:
            one, lap = texture_raw_scores([path], device, image_size)
            outputs.extend(one)
            laps.extend(lap)
        return outputs, laps
    images = torch.from_numpy(np.stack(arrays).astype(np.float32) / 255.0).unsqueeze(1).to(device)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=images.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    dx, dy = F.conv2d(images, sobel_x, padding=1), F.conv2d(images, sobel_y, padding=1)
    patches = F.avg_pool2d(torch.sqrt(dx.square() + dy.square()), kernel_size=14, stride=14).flatten(1)
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=images.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(images, lap_kernel, padding=1).flatten(1).var(dim=1, unbiased=False)
    return [row.cpu().numpy() for row in patches], [float(value) for value in laplacian.cpu()]


def balanced_threshold(per_scene_values: Dict[str, np.ndarray], percentile: float) -> float:
    # Equal-size deterministic draws make every scene contribute equally.
    count = min(len(values) for values in per_scene_values.values())
    equal = []
    for name, values in sorted(per_scene_values.items()):
        if len(values) == count:
            equal.append(values)
        else:
            indices = np.linspace(0, len(values) - 1, count, dtype=int)
            equal.append(values[indices])
    return float(np.percentile(np.concatenate(equal), percentile))


def random_sequences(scene: Scene, count: int, seed: int, tag: str) -> List[List[int]]:
    rng = np.random.default_rng(stable_seed(scene.dataset, scene.name, tag, seed=seed))
    available = list(range(len(scene.ids)))
    sequences = []
    for _ in range(count):
        if len(available) >= 8:
            sequences.append(rng.choice(available, 8, replace=False).tolist())
        else:
            sequences.append(available + [available[0]] * (8 - len(available)))
    return sequences


@torch.inference_mode()
def normalized_occlusion_scores(scene: Scene, sequences: Sequence[Sequence[int]], device: str) -> List[List[float]]:
    """Occlusion fraction normalized over geometrically testable target views.

    A target only enters a source pixel's denominator when the projection is in
    its image plane and it has valid GT depth. Pixels with fewer than two such
    candidates are excluded from the frame denominator. This keeps view-frustum
    separation distinct from genuine depth-discontinuity occlusion.
    """
    dev = torch.device(device)
    depths = torch.from_numpy(scene.depths).to(dev)
    valids = torch.from_numpy(scene.valids).to(dev)
    intrinsics = torch.from_numpy(scene.intrinsics).to(dev)
    w2cs = torch.from_numpy(scene.w2cs).to(dev)
    h, w = scene.depths.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    pixels = torch.stack((xx.reshape(-1), yy.reshape(-1), torch.ones(h * w, device=dev)))
    values = []
    for sequence in sequences:
        view_count = len(sequence)
        if view_count < 3:
            raise ValueError("Normalized occlusion needs a source plus at least two target views")
        per_position = []
        for pos, source_idx in enumerate(sequence):
            target_indices = [idx for offset, idx in enumerate(sequence) if offset != pos]
            z = depths[source_idx].reshape(-1)
            points = torch.linalg.solve(intrinsics[source_idx], pixels) * z
            world = torch.linalg.inv(w2cs[source_idx]) @ torch.cat((points, torch.ones(1, h * w, device=dev)))
            cameras = w2cs[target_indices] @ world.unsqueeze(0)
            projected_z = cameras[:, 2]
            uv = torch.bmm(intrinsics[target_indices], cameras[:, :3])
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            inside = (projected_z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
            grid = torch.stack((2 * u / max(w - 1, 1) - 1, 2 * v / max(h - 1, 1) - 1), -1).view(view_count - 1, h, w, 2)
            sampled_depth = F.grid_sample(depths[target_indices, None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0].reshape(view_count - 1, -1)
            sampled_valid = F.grid_sample(valids[target_indices, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=True)[:, 0].reshape(view_count - 1, -1) > 0.5
            candidates = inside & sampled_valid
            seen = candidates & ((sampled_depth - projected_z).abs() <= 0.02 * projected_z)
            source_valid = valids[source_idx].reshape(-1)
            candidate_count = candidates.sum(0)
            testable = source_valid & (candidate_count >= 2)
            denominator = testable.sum().clamp_min(1)
            occluded = testable & (seen.sum(0) < 2)
            per_position.append(float(occluded.sum().float() / denominator))
        values.append(per_position)
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def manifest_record(scene: Scene, sequence: List[int], split: str, sample_idx: int, epoch: int | None,
                    criterion: str, hard_positions: List[int], score: float, repeated: bool) -> Dict:
    four = [sequence[position] for position in STUDENT]
    return {
        "dataset": scene.dataset, "scene": scene.name, "split": split, "epoch": epoch, "sample_idx": sample_idx,
        "criterion": criterion, "hard_student_positions": hard_positions, "difficulty_score": score, "repeated_candidate": repeated,
        "eight_local_indices": sequence, "four_local_indices": four,
        "eight_frame_indices": sequence, "four_frame_indices": four,
        "eight_frame_ids": [scene.ids[i] for i in sequence], "four_frame_ids": [scene.ids[i] for i in four],
        "eight_image_files": [scene.images[i] for i in sequence], "four_image_files": [scene.images[i] for i in four],
    }


def write_source_frames(directory: Path, scenes: Sequence[Scene], hard_names: Sequence[str]) -> None:
    """Write all source-frame identities used by hard sequence indices."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "source_frames.jsonl", "w", encoding="utf-8") as handle:
        for scene in scenes:
            if scene.name not in hard_names:
                continue
            matrix_path = MATRIX_ROOTS[scene.dataset] / f"{scene.name}.npz"
            count = len(scene.ids)
            handle.write(json.dumps({
                "dataset": scene.dataset, "scene": scene.name, "total_frames": count,
                "selected_count": count, "selected_frame_indices": list(range(count)),
                "selected_frame_ids": scene.ids, "matrix_path": str(matrix_path),
            }) + "\n")


def build_for_criterion(dataset: str, criterion: str, scenes: List[Scene], output_root: Path,
                        device: str, seed: int, candidate_count: int, eval_samples_per_scene: int) -> None:
    texture_scores, laplacian_scores = {}, {}
    if criterion == "texture":
        patch_scores, per_image_patches = {}, {}
        for scene in scenes:
            per_frame, lap = [], []
            frame_patches, frame_laplacian = texture_raw_scores(scene.images, device)
            for patches, lap_var in zip(frame_patches, frame_laplacian):
                per_frame.append(patches)
                lap.append(lap_var)
            per_image_patches[scene.name] = per_frame
            patch_scores[scene.name] = np.concatenate(per_frame)
            laplacian_scores[scene.name] = np.asarray(lap)
        q30 = balanced_threshold(patch_scores, 30)
        for scene in scenes:
            texture_scores[scene.name] = np.asarray([float((patches < q30).mean())
                                                     for patches in per_image_patches[scene.name]])
        threshold_meta = {"patch_gradient_q30": q30, "laplacian_spearman": spearman(
            np.concatenate([texture_scores[s.name] for s in scenes]),
            -np.concatenate([laplacian_scores[s.name] for s in scenes]),
        )}
        train_sequences = {scene.name: random_sequences(scene, candidate_count, seed, "texture-train") for scene in scenes}
        eval_sequences = {scene.name: random_sequences(scene, candidate_count, seed + 1, "texture-eval") for scene in scenes}
        train_values = {scene.name: np.asarray([[texture_scores[scene.name][idx] for idx in seq] for seq in train_sequences[scene.name]]) for scene in scenes}
        eval_values = {scene.name: np.asarray([[texture_scores[scene.name][idx] for idx in seq] for seq in eval_sequences[scene.name]]) for scene in scenes}
    else:
        train_sequences = {scene.name: random_sequences(scene, candidate_count, seed, "occlusion-train") for scene in scenes}
        eval_sequences = {scene.name: random_sequences(scene, candidate_count, seed + 1, "occlusion-eval") for scene in scenes}
        train_values, eval_values = {}, {}
        for scene in scenes:
            print(f"[occlusion] {dataset}/{scene.name}: {candidate_count} 8V windows", flush=True)
            train_values[scene.name] = np.asarray(occlusion_scores(scene, train_sequences[scene.name], device))
            eval_values[scene.name] = np.asarray(occlusion_scores(scene, eval_sequences[scene.name], device))
        threshold_meta = {"visibility_count_threshold": 2, "relative_depth_tolerance": 0.02}

    q4 = balanced_threshold({scene.name: train_values[scene.name].reshape(-1) for scene in scenes}, 75)
    scene_scores = {scene.name: float((train_values[scene.name][:, STUDENT] >= q4).mean()) for scene in scenes}
    hard_names = [name for name, _ in sorted(scene_scores.items(), key=lambda item: (-item[1], item[0]))[:5]]
    hard_scenes = [scene for scene in scenes if scene.name in hard_names]
    records = []
    for split, per_scene_sequences, per_scene_values, target_count in (
        ("train", train_sequences, train_values, 10), ("eval", eval_sequences, eval_values, eval_samples_per_scene),
    ):
        for scene in hard_scenes:
            values = per_scene_values[scene.name]
            eligible = []
            for sequence, scores in zip(per_scene_sequences[scene.name], values):
                hard_positions = [position for position in STUDENT if scores[position] >= q4]
                if hard_positions:
                    eligible.append((float(scores[STUDENT].mean()), sequence, hard_positions))
            if not eligible:
                raise RuntimeError(f"{dataset}/{criterion}/{scene.name}: no sequence with a Q4 student frame")
            eligible.sort(key=lambda item: -item[0])
            for epoch in range(3 if split == "train" else 1):
                for sample_idx in range(target_count):
                    score, sequence, positions = eligible[sample_idx % len(eligible)]
                    records.append(manifest_record(scene, sequence, split, sample_idx, epoch if split == "train" else None,
                                                   criterion, positions, score, sample_idx >= len(eligible)))
    directory = output_root / dataset / criterion
    directory.mkdir(parents=True, exist_ok=True)
    write_source_frames(directory, scenes, hard_names)
    for split in ("train", "eval"):
        with open(directory / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record) + "\n")
    report = {"dataset": dataset, "criterion": criterion, "q4_threshold": q4, "top5_scenes": hard_names,
              "scene_scores": scene_scores, "candidate_sequences_per_scene": candidate_count, **threshold_meta}
    with open(directory / "selection_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"[done] {dataset}/{criterion}: Q4={q4:.6f}, Top-5={hard_names}")


def build_occlusion_streaming(dataset: str, output_root: Path, device: str, seed: int,
                              candidate_count: int, selected: Sequence[str] | None,
                              eval_samples_per_scene: int) -> None:
    """Avoid retaining every native-depth scene while retaining exact scoring."""
    scene_names = [path.stem for path in sorted(MATRIX_ROOTS[dataset].glob("*.npz"))
                   if selected is None or path.stem in selected]
    values, sequences, scenes = {}, {}, {}
    for name in scene_names:
        scene = load_scenes(dataset, need_geometry=True, selected=[name])[0]
        scenes[name] = scene
        train = random_sequences(scene, candidate_count, seed, "occlusion-train")
        eval_ = random_sequences(scene, candidate_count, seed + 1, "occlusion-eval")
        print(f"[normalized-occlusion] {dataset}/{name}: {candidate_count} 8V windows", flush=True)
        values[name] = (np.asarray(normalized_occlusion_scores(scene, train, device)),
                        np.asarray(normalized_occlusion_scores(scene, eval_, device)))
        sequences[name] = (train, eval_)
        del scene.depths, scene.valids
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    q4 = balanced_threshold({name: pair[0].reshape(-1) for name, pair in values.items()}, 75)
    scene_scores = {name: float((pair[0][:, STUDENT] >= q4).mean()) for name, pair in values.items()}
    hard_names = [name for name, _ in sorted(scene_scores.items(), key=lambda item: (-item[1], item[0]))[:5]]
    records = []
    for split_index, split, target_count in ((0, "train", 10), (1, "eval", eval_samples_per_scene)):
        for name in hard_names:
            score_rows, sequence_rows = values[name][split_index], sequences[name][split_index]
            eligible = []
            for sequence, scores in zip(sequence_rows, score_rows):
                positions = [position for position in STUDENT if scores[position] >= q4]
                if positions:
                    eligible.append((float(scores[STUDENT].mean()), sequence, positions))
            eligible.sort(key=lambda item: -item[0])
            for epoch in range(3 if split == "train" else 1):
                for sample_idx in range(target_count):
                    score, sequence, positions = eligible[sample_idx % len(eligible)]
                    records.append(manifest_record(scenes[name], sequence, split, sample_idx,
                                                   epoch if split == "train" else None, "normalized_occlusion",
                                                   positions, score, sample_idx >= len(eligible)))
    directory = output_root / dataset / "occlusion"
    directory.mkdir(parents=True, exist_ok=True)
    write_source_frames(directory, list(scenes.values()), hard_names)
    for split in ("train", "eval"):
        with open(directory / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record) + "\n")
    report = {"dataset": dataset, "criterion": "occlusion", "q4_threshold": q4,
              "top5_scenes": hard_names, "scene_scores": scene_scores,
              "candidate_sequences_per_scene": candidate_count, "visibility_count_threshold": 2,
              "relative_depth_tolerance": 0.02,
              "candidate_definition": "inside target image and target GT depth valid",
              "frame_denominator": "source-valid pixels with at least two candidate target views",
              "terminology": "Q4 normalized occlusion; legacy all-other-view score is low multi-view support"}
    with open(directory / "selection_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"[done] {dataset}/occlusion: Q4={q4:.6f}, Top-5={hard_names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(MATRIX_ROOTS), default=sorted(MATRIX_ROOTS))
    parser.add_argument("--scenes", nargs="*", help="Optional scene names; requires one dataset.")
    parser.add_argument("--criteria", nargs="+", choices=["texture", "occlusion"], default=["texture", "occlusion"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/hard_view_subsets")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--eval-samples-per-scene", type=int, default=5)
    args = parser.parse_args()
    if args.scenes and len(args.datasets) != 1:
        parser.error("--scenes requires exactly one dataset")
    if "occlusion" in args.criteria and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Occlusion requested on CUDA but CUDA is unavailable")
    for dataset in args.datasets:
        if "texture" in args.criteria:
            scenes = load_scenes(dataset, need_geometry=False, selected=args.scenes)
            build_for_criterion(dataset, "texture", scenes,
                                args.output_root, args.device, args.seed, args.candidate_count, args.eval_samples_per_scene)
        if "occlusion" in args.criteria:
            build_occlusion_streaming(dataset, args.output_root, args.device, args.seed,
                                      args.candidate_count, args.scenes, args.eval_samples_per_scene)


if __name__ == "__main__":
    main()
