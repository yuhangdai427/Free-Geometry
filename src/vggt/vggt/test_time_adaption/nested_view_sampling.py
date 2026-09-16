"""Reproducible nested frame pools for view-count ablations."""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


TRAIN_POOL_SIZE = 16


def stable_seed(seed: int, scene_id: str, sample_idx: int, pool_size: int) -> int:
    """Return a process-independent seed for one scene/sample pool."""
    key = f"e6::{seed}::{scene_id}::{sample_idx}::{pool_size}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def sample_pool(
    num_frames: int,
    pool_size: int,
    seed: int,
    scene_id: str,
    sample_idx: int,
    allow_short: bool = False,
) -> List[int]:
    """Sample ordered unique source-frame indices, preserving frame zero if present."""
    if num_frames < pool_size:
        if allow_short:
            return list(range(num_frames))
        raise ValueError(
            f"{scene_id} has only {num_frames} frames, but E6 requires {pool_size} unique frames"
        )
    rng = random.Random(stable_seed(seed, scene_id, sample_idx, pool_size))
    candidates = list(range(1, num_frames))
    chosen = [0] + rng.sample(candidates, pool_size - 1)
    return [chosen[0]] + sorted(chosen[1:])


def nested_subset(pool16: List[int], num_views: int) -> List[int]:
    """Return the 4/8/16 teacher subset at fixed positions in a P16 pool."""
    if len(pool16) != TRAIN_POOL_SIZE:
        raise ValueError(f"Expected P16, got {len(pool16)} entries")
    if num_views not in (4, 8, 16):
        raise ValueError(f"E6 supports teacher views 4, 8, or 16; got {num_views}")
    return pool16[:: TRAIN_POOL_SIZE // num_views]


def make_records(
    scene_frame_counts: Mapping[str, int],
    *,
    seed: int,
    samples_per_scene: int,
    pool_size: int = TRAIN_POOL_SIZE,
    allow_short: bool = False,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for scene_id in sorted(scene_frame_counts):
        for sample_idx in range(samples_per_scene):
            records.append(
                {
                    "scene_id": scene_id,
                    "sample_idx": sample_idx,
                    "frame_indices": sample_pool(
                        scene_frame_counts[scene_id], pool_size, seed, scene_id, sample_idx, allow_short
                    ),
                }
            )
    return records


def write_jsonl(records: Iterable[Mapping[str, object]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: str, expected_pool_size: int = TRAIN_POOL_SIZE) -> Dict[tuple[str, int], List[int]]:
    pools: Dict[tuple[str, int], List[int]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                key = (str(record["scene_id"]), int(record["sample_idx"]))
                frames = [int(value) for value in record["frame_indices"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid nested pool at {path}:{line_number}") from exc
            if len(frames) != expected_pool_size or len(set(frames)) != expected_pool_size:
                raise ValueError(f"Invalid P{expected_pool_size} at {path}:{line_number}")
            if frames[0] != 0:
                raise ValueError(f"P{expected_pool_size} must retain reference frame 0 at {path}:{line_number}")
            if key in pools:
                raise ValueError(f"Duplicate nested pool for {key} in {path}")
            pools[key] = frames
    return pools
