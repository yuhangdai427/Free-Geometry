"""Deterministic non-overlapping view selection from co-visibility matrices.

The selector is intentionally loader-agnostic: benchmark loaders remain the
source of poses/images, while this module supplies their selected indices.
"""

import json
from copy import copy
from pathlib import Path
from typing import List, Optional

import numpy as np


def select_non_overlapping_indices(overlap: np.ndarray, threshold: float, *, required_index: Optional[int] = 0) -> List[int]:
    """Return a deterministic maximal set with pairwise overlap <= threshold.

    ``overlap`` must be a symmetric [N, N] score where larger values mean a
    conflict.  The greedy order prefers views with fewer conflicts, which
    preserves more candidates than simply traversing temporal frame order.
    ``required_index=0`` keeps a stable first/reference view when available.
    """
    overlap = np.asarray(overlap, dtype=np.float32)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError(f"Expected a square overlap matrix, got {overlap.shape}")
    if not np.allclose(overlap, overlap.T, atol=1e-5):
        raise ValueError("Non-overlap selection requires a symmetric overlap matrix")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    count = len(overlap)
    if not count:
        return []
    conflict = overlap > threshold
    np.fill_diagonal(conflict, False)
    degrees = conflict.sum(axis=1)
    order = sorted(range(count), key=lambda index: (int(degrees[index]), index))
    selected: List[int] = []
    if required_index is not None:
        if not 0 <= required_index < count:
            raise ValueError(f"required_index {required_index} is outside 0..{count - 1}")
        selected.append(required_index)
    for index in order:
        if index not in selected and all(not conflict[index, chosen] for chosen in selected):
            selected.append(index)
    return sorted(selected)


def selected_indices_from_npz(path: str, threshold: float, *, required_index: Optional[int] = 0) -> List[int]:
    """Load a matrix artifact and select its non-overlapping candidate indices."""
    with np.load(path) as data:
        if "max_directional_overlap" not in data:
            raise ValueError(f"{path} lacks max_directional_overlap; recompute with the current script")
        local_indices = select_non_overlapping_indices(data["max_directional_overlap"], threshold, required_index=required_index)
        return [int(data["candidate_indices"][index]) for index in local_indices]


class CovisibilitySubsetDataset:
    """Read-only dataset wrapper that exposes one non-overlapping view subset per scene.

    ``manifest_path`` is produced by ``select_nonoverlapping_views.py``.  The
    wrapped dataset is never modified; its original frames are selected by the
    stored original frame indices, so all existing image/depth handling stays
    intact.
    """

    def __init__(self, dataset, manifest_path: str):
        self.dataset = dataset
        self.SCENES = list(dataset.SCENES)
        self._selected = {}
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                self._selected[str(record["scene"])] = [int(index) for index in record["selected_frame_indices"]]

    def get_data(self, scene: str):
        source = self.dataset.get_data(scene)
        if scene not in self._selected:
            raise KeyError(f"No non-overlap selection for scene {scene!r}")
        indices = self._selected[scene]
        count = len(source.image_files)
        if not indices or min(indices) < 0 or max(indices) >= count:
            raise ValueError(f"Invalid selected indices for {scene!r}: {indices}")
        subset = copy(source)
        subset.image_files = [source.image_files[index] for index in indices]
        subset.extrinsics = np.asarray(source.extrinsics)[indices]
        subset.intrinsics = np.asarray(source.intrinsics)[indices]
        subset.aux = copy(source.aux)
        for key, value in source.aux.items():
            if isinstance(value, list) and len(value) == count:
                subset.aux[key] = [value[index] for index in indices]
            elif isinstance(value, np.ndarray) and len(value) == count:
                subset.aux[key] = value[indices]
        return subset

    def __getattr__(self, name):
        return getattr(self.dataset, name)
