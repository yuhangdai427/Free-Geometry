"""Reproducible low-overlap view sequences for the VGGT ablation protocol.

This deliberately reads the Co-visibility artifacts instead of benchmark data
loaders.  The artifacts are the source of truth for the depth/pose frame order
used to make the 0.1 selection, and this also handles ScanNet++'s iphone RGB
subdirectory without changing its existing benchmark loader.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


STUDENT_POSITIONS = [0, 2, 4, 6]


@dataclass
class ResolvedScene:
    dataset: str
    scene: str
    frame_indices: List[int]
    frame_ids: List[str]
    image_files: List[str]
    extrinsics: np.ndarray
    intrinsics: np.ndarray


def _stable_seed(*parts: object, seed: int) -> int:
    value = "|".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _image_from_depth(dataset: str, depth_path: str) -> str:
    depth = Path(depth_path)
    if dataset == "eth3d":
        # .../<scene>/ground_truth_depth/dslr_images/<frame>.JPG
        return str(depth.parents[2] / "images" / depth.parent.name / depth.name)
    if dataset == "scannetpp":
        # .../<scene>/merge_dslr_iphone/render_depth/frame_x.png
        root = depth.parents[1]
        stem = depth.stem
        candidates = [root / "images" / "iphone" / f"{stem}{suffix}" for suffix in (".jpg", ".JPG", ".png")]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        matches = list((root / "images" / "iphone").glob(f"{stem}.*"))
        if matches:
            return str(matches[0])
    if dataset == "7scenes":
        return str(depth.with_name(depth.name.replace(".depth.png", ".color.png")))
    if dataset == "hiroom":
        matches = sorted((depth.parents[1] / "image").glob(f"{depth.stem}.*"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Cannot resolve RGB image for {dataset}: {depth_path}")


def load_selected_scenes(selection_jsonl: str, datasets: Iterable[str]) -> List[ResolvedScene]:
    """Load 0.1-selected Co frames plus exact calibration from their NPZ files."""
    wanted = set(datasets)
    resolved: List[ResolvedScene] = []
    # The selection manifest lives at <repo>/artifacts/<run>/manifest.jsonl.
    # Resolve its repository-relative matrix paths independently of the caller's cwd.
    manifest_path = Path(selection_jsonl).resolve()
    repository_root = manifest_path.parents[2]
    with open(selection_jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["dataset"] not in wanted:
                continue
            matrix_path = Path(record["matrix_path"])
            if not matrix_path.is_absolute():
                matrix_path = repository_root / matrix_path
            matrix = np.load(matrix_path, allow_pickle=False)
            all_ids = [str(x) for x in matrix["frame_ids"]]
            id_to_index = {frame_id: index for index, frame_id in enumerate(all_ids)}
            selected_ids = [str(x) for x in record["selected_frame_ids"]]
            try:
                indices = [id_to_index[frame_id] for frame_id in selected_ids]
            except KeyError as exc:
                raise ValueError(f"{record['dataset']}/{record['scene']}: missing artifact ID {exc}") from exc
            depth_paths = matrix["depth_paths"]
            images = [_image_from_depth(record["dataset"], str(depth_paths[index])) for index in indices]
            missing = [path for path in images if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(f"{record['dataset']}/{record['scene']}: missing RGB: {missing[0]}")
            resolved.append(ResolvedScene(
                dataset=record["dataset"], scene=record["scene"], frame_indices=indices,
                frame_ids=selected_ids, image_files=images,
                extrinsics=np.asarray(matrix["extrinsics_w2c"])[indices],
                intrinsics=np.asarray(matrix["intrinsics"])[indices],
            ))
    if not resolved:
        raise ValueError(f"No selected scenes for {sorted(wanted)} in {selection_jsonl}")
    return resolved


def make_sequence(scene: ResolvedScene, seed: int, split: str, sample_idx: int, epoch: int | None = None) -> Dict:
    """Randomly pick eight distinct, pairwise low-overlap frames.

    The sparse-view protocol is an audited stress test, not an input-padding
    protocol.  A scene without eight unique candidates is therefore ineligible
    rather than being repaired by repeating a camera.
    """
    rng = np.random.default_rng(_stable_seed(scene.dataset, scene.scene, split, epoch, sample_idx, seed=seed))
    available = list(range(len(scene.frame_indices)))
    if len(available) < 8:
        raise ValueError(
            f"{scene.dataset}/{scene.scene} has only {len(available)} low-overlap "
            "frames; an 8V strict sequence cannot be constructed"
        )
    local_eight = rng.choice(available, size=8, replace=False).tolist()
    if len(set(local_eight)) != 8:
        raise AssertionError("Strict 8V sequence unexpectedly contains a repeated frame")
    local_four = [local_eight[position] for position in STUDENT_POSITIONS]
    assert local_four == [local_eight[i] for i in STUDENT_POSITIONS]
    return {
        "dataset": scene.dataset, "scene": scene.scene, "split": split, "epoch": epoch,
        "sample_idx": sample_idx, "eight_local_indices": local_eight,
        "four_local_indices": local_four,
        "eight_frame_indices": [scene.frame_indices[i] for i in local_eight],
        "four_frame_indices": [scene.frame_indices[i] for i in local_four],
        "eight_frame_ids": [scene.frame_ids[i] for i in local_eight],
        "four_frame_ids": [scene.frame_ids[i] for i in local_four],
        "eight_image_files": [scene.image_files[i] for i in local_eight],
        "four_image_files": [scene.image_files[i] for i in local_four],
    }


def write_sequences(selection_jsonl: str, output_jsonl: str, datasets: Iterable[str], split: str,
                    samples_per_scene: int, seed: int, epochs: int = 1) -> int:
    scenes = load_selected_scenes(selection_jsonl, datasets)
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for scene in scenes:
            for epoch in range(epochs):
                for sample_idx in range(samples_per_scene):
                    record = make_sequence(scene, seed, split, sample_idx, epoch if split == "train" else None)
                    handle.write(json.dumps(record) + "\n")
                    count += 1
    return count


def sequence_calibration(scene: ResolvedScene, record: Dict, arm: str):
    """Return image/calibration arrays for an arm, keeping 4V positions locked."""
    if arm in {"base_8v", "lora_8v"}:
        local = record["eight_local_indices"]
    elif arm in {"base_16v", "lora_16v"}:
        local = record["sixteen_local_indices"]
    elif arm in {"base_4v", "base_8v_extract_4v", "lora_4v"}:
        local = record["four_local_indices"]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return ([scene.image_files[i] for i in local], scene.extrinsics[local], scene.intrinsics[local])
