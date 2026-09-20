"""
Dataset for VGGT Free-Geometry.

This module provides a dataset loader that can load data from benchmark datasets
for VGGT Free-Geometry training. Images are returned in [0, 1] range since
VGGT's Aggregator handles normalization internally.
"""

import os
import random
import csv
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Import benchmark dataset classes
from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.bench.datasets.sevenscenes import SevenScenes
from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
from depth_anything_3.bench.datasets.dtu import DTU


# Dataset registry
DATASET_REGISTRY = {
    'eth3d': ETH3D,
    '7scenes': SevenScenes,
    'scannetpp': ScanNetPP,
    'hiroom': HiRoomDataset,
    'dtu': DTU,
}


class VGGTFreeGeometryDataset(Dataset):
    """
    Dataset for VGGT Free-Geometry.

    Returns 8 uniformly sampled views per scene for teacher,
    and corresponding 4 views (indices [0,2,4,6]) for student.

    Note: Images are returned in [0, 1] range (NOT normalized).
    VGGT's Aggregator handles normalization internally.

    Args:
        dataset_name: Name of benchmark dataset ('eth3d', '7scenes', 'scannetpp', 'hiroom', 'dtu')
        num_views: Number of views for teacher (default: 8)
        image_size: Target size for longest side (default 504, DA3-style)
        student_indices: Which teacher views to use for student (default: [0,2,4,6])
        augment: Whether to apply data augmentation
        samples_per_scene: Number of samples to generate per scene (default: 2)
        seed: Random seed for sampling
        seeds_list: Optional list of seeds for deterministic sampling
        first_frame_ref: Force first frame as reference after sampling
        subset_sampling: Enable subset-based frame sampling (recommended for large scenes)
        subset_ratio: Ratio of frames to include in subset (default: 0.05 = 5%)
        stride_sampling: Enable stride-based anchor sampling (random anchor + stride window)
        stride: File-list stride for consecutive views (default: 2, i.e. every 2nd file)
        paired_sampling: Enable paired anchor+companion frame sampling
        paired_gap: Gap between anchor and companion in sorted file list (default: 3)
        fixed_subset_seed: If set, restrict training to a fixed frame subset per scene
            computed using the same shuffle logic as the benchmark evaluator
        fixed_subset_max_frames: Max frames for the fixed subset (default: 100)
    """

    def __init__(
        self,
        dataset_name: str,
        num_views: int = 8,
        image_size: int = 504,
        student_indices: List[int] = None,
        augment: bool = True,
        samples_per_scene: int = 2,
        seed: int = 42,
        seeds_list: Optional[List[int]] = None,
        first_frame_ref: bool = False,
        subset_sampling: bool = False,
        subset_ratio: float = 0.05,
        stride_sampling: bool = False,
        stride: int = 2,
        paired_sampling: bool = False,
        paired_gap: int = 3,
        fixed_subset_seed: Optional[int] = None,
        fixed_subset_max_frames: int = 100,
        scenes: Optional[List[str]] = None,
        pair_manifest: Optional[str] = None,
    ):
        super().__init__()

        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )

        self.dataset_name = dataset_name
        self.num_views = num_views
        self.image_size = image_size
        self.student_indices = student_indices or [0, 2, 4, 6]
        self.augment = augment
        self.seeds_list = seeds_list
        self.samples_per_scene = samples_per_scene
        self.seed = seed
        self.first_frame_ref = first_frame_ref
        self.subset_sampling = subset_sampling
        self.subset_ratio = subset_ratio
        self.stride_sampling = stride_sampling
        self.stride = stride
        self.paired_sampling = paired_sampling
        self.paired_gap = paired_gap
        self.fixed_subset_seed = fixed_subset_seed
        self.fixed_subset_max_frames = fixed_subset_max_frames
        self.scene_subset_indices = {}  # Cache subset indices per scene
        self._fixed_subset_cache = {}  # Cache fixed subset indices per scene
        self.current_epoch = 0  # Mixed into sample seed for per-epoch diversity
        self.pair_manifest = pair_manifest

        # Initialize benchmark dataset
        self.benchmark_dataset = DATASET_REGISTRY[dataset_name]()

        # Use all scenes unless a per-scene TTA run selected a subset.
        available_scenes = list(self.benchmark_dataset.SCENES)
        if scenes is None:
            self.scenes = available_scenes
        else:
            missing = sorted(set(scenes) - set(available_scenes))
            if missing:
                raise ValueError(
                    f"Unknown scene(s) for {dataset_name}: {missing}. "
                    f"Available: {available_scenes}"
                )
            self.scenes = [scene for scene in available_scenes if scene in scenes]

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found for {dataset_name}")

        self.pairs = None
        if pair_manifest is not None:
            with open(pair_manifest, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            pairs = []
            for row in rows:
                if row.get("dataset") != dataset_name:
                    continue
                scene = row.get("scene")
                if scene not in self.scenes:
                    raise ValueError(f"Pair manifest scene is unavailable for {dataset_name}: {scene}")
                try:
                    sample_seed = int(str(row.get("seed", row.get("window_seed", ""))).removeprefix("seed"))
                except ValueError as exc:
                    raise ValueError(f"Invalid pair-manifest seed in {pair_manifest}") from exc
                pairs.append((scene, sample_seed))
            if not pairs:
                raise ValueError(f"No pairs for dataset={dataset_name} in {pair_manifest}")
            self.pairs = pairs

        print(f"Loaded {len(self.scenes)} scenes for {dataset_name} (all scenes for training)")
        if self.pairs is None:
            print(f"Generating {samples_per_scene} samples per scene = {len(self.scenes) * samples_per_scene} total samples")
        else:
            print(f"Loaded {len(self.pairs)} exact scene/seed pairs from {pair_manifest}")
        if self.subset_sampling:
            print(f"Using subset sampling: {subset_ratio*100:.1f}% of frames per scene")
        if self.stride_sampling:
            print(f"Using stride sampling: random anchor + {num_views} views at stride {stride}")
        if self.paired_sampling:
            print(f"Using paired sampling: {num_views // 2} anchors + companions at gap {paired_gap}")
        if self.fixed_subset_seed is not None:
            print(f"Using fixed subset sampling: seed={fixed_subset_seed}, max_frames={fixed_subset_max_frames}")
            # Eagerly compute and print fixed subset for each scene
            for scene in self.scenes:
                data = self.benchmark_dataset.get_data(scene)
                num_frames = len(data['image_files']) if isinstance(data, dict) else len(data.image_files)
                subset = self._get_fixed_subset_indices(scene, num_frames)
                print(f"  [Fixed Subset] {scene}: {num_frames} -> {len(subset)} frames")
                print(f"    indices: {subset}")

        # Cache for scene data
        self._scene_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        if self.pairs is not None:
            return len(self.pairs)
        return len(self.scenes) * self.samples_per_scene

    def _load_scene_data(self, scene: str) -> Dict:
        """Load scene data from benchmark dataset."""
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        # Get data from benchmark dataset
        data = self.benchmark_dataset.get_data(scene)

        self._scene_cache[scene] = data
        return data

    def _get_subset_indices(self, scene: str, num_available: int) -> List[int]:
        """
        Get or compute the subset of frame indices for a scene.

        Uniformly samples subset_ratio of frames, keeping temporal order.

        Args:
            scene: Scene name (for caching)
            num_available: Total number of frames in scene

        Returns:
            List of frame indices in the subset
        """
        if scene in self.scene_subset_indices:
            return self.scene_subset_indices[scene]

        # Compute subset size (at least num_views frames)
        subset_size = max(self.num_views, int(num_available * self.subset_ratio))

        # Uniformly sample indices (keep order)
        if subset_size >= num_available:
            indices = list(range(num_available))
        else:
            # Uniform spacing
            step = num_available / subset_size
            indices = [int(i * step) for i in range(subset_size)]

        self.scene_subset_indices[scene] = indices
        return indices

    def _get_fixed_subset_indices(self, scene: str, num_available: int) -> List[int]:
        """
        Get or compute the fixed frame subset for a scene.

        Replicates the benchmark evaluator's sampling logic:
        random.seed(seed) -> shuffle -> take first max_frames -> sort.

        Args:
            scene: Scene name (for caching)
            num_available: Total number of frames in scene

        Returns:
            Sorted list of frame indices in the fixed subset
        """
        if scene in self._fixed_subset_cache:
            return self._fixed_subset_cache[scene]

        if num_available <= self.fixed_subset_max_frames:
            indices = list(range(num_available))
        else:
            # Replicate benchmark evaluator logic (evaluator.py:464-468)
            rng_fix = random.Random(self.fixed_subset_seed)
            all_indices = list(range(num_available))
            rng_fix.shuffle(all_indices)
            indices = sorted(all_indices[:self.fixed_subset_max_frames])

        self._fixed_subset_cache[scene] = indices
        return indices

    def _sample_view_indices(self, num_available: int, rng: random.Random,
                             scene: str = None, sample_idx: int = 0) -> List[int]:
        """
        Sample view indices from available images.

        If subset_sampling is enabled, uses consecutive windows from the subset.
        Windows are non-overlapping (0-7, 8-15, 16-23, ...) and wrap with offset
        when all full windows are exhausted.

        Args:
            num_available: Number of available images in scene
            rng: Random number generator
            scene: Scene name (required if subset_sampling is enabled)
            sample_idx: Sample index within the scene (for consecutive window selection)

        Returns:
            List of indices to sample
        """
        # Fixed subset sampling: randomly sample num_views from the fixed subset
        if self.fixed_subset_seed is not None and scene is not None:
            subset = self._get_fixed_subset_indices(scene, num_available)
            subset_size = len(subset)
            if subset_size <= self.num_views:
                indices = list(subset)
                while len(indices) < self.num_views:
                    indices.append(subset[rng.randint(0, subset_size - 1)])
            else:
                indices = sorted(rng.sample(subset, self.num_views))
            return indices

        # Stride sampling: random anchor + num_views at stride from file list
        if self.stride_sampling:
            # Max valid anchor: anchor + (num_views-1)*stride < num_available
            max_anchor = num_available - 1 - (self.num_views - 1) * self.stride
            if max_anchor < 0:
                # Scene too small for stride window, fall back to all available
                indices = list(range(num_available))
                while len(indices) < self.num_views:
                    indices.append(rng.randint(0, num_available - 1))
            else:
                anchor = rng.randint(0, max_anchor)
                indices = [anchor + i * self.stride for i in range(self.num_views)]
            return indices

        # Paired anchor + companion sampling
        if self.paired_sampling:
            num_anchors = self.num_views // 2  # 4 for 8 views
            gap = self.paired_gap

            max_anchor_pos = num_available - 1 - gap
            if max_anchor_pos < 0:
                # Scene too small, fall back to all available with repetition
                indices = list(range(num_available))
                while len(indices) < self.num_views:
                    indices.append(rng.randint(0, num_available - 1))
                return indices

            valid_positions = list(range(max_anchor_pos + 1))
            if len(valid_positions) < num_anchors:
                anchors = sorted([rng.choice(valid_positions) for _ in range(num_anchors)])
            else:
                anchors = sorted(rng.sample(valid_positions, num_anchors))

            # Interleave: [anchor0, companion0, anchor1, companion1, ...]
            indices = []
            used = set()
            for anchor in anchors:
                companion = anchor + gap
                companion = min(companion, num_available - 1)
                # Ensure companion is unique (shift forward until unique or hit boundary)
                while companion in used and companion < num_available - 1:
                    companion += 1
                used.add(anchor)
                used.add(companion)
                indices.append(anchor)
                indices.append(companion)

            return indices

        # If subset sampling enabled, use consecutive windows from subset
        if self.subset_sampling and scene is not None:
            subset_indices = self._get_subset_indices(scene, num_available)
            subset_size = len(subset_indices)

            if subset_size <= self.num_views:
                # Subset too small, use all with repetition if needed
                indices = list(subset_indices)
                while len(indices) < self.num_views:
                    indices.append(subset_indices[rng.randint(0, subset_size - 1)])
            else:
                # Calculate how many full non-overlapping windows we can have
                num_full_windows = subset_size // self.num_views

                if sample_idx < num_full_windows:
                    # Use non-overlapping window
                    start = sample_idx * self.num_views
                    indices = list(subset_indices[start:start + self.num_views])
                else:
                    # Wrap around with offset
                    # After using all full windows, start with offset of num_views // 2
                    offset = self.num_views // 2  # e.g., 4 for 8 views
                    wrap_idx = sample_idx - num_full_windows
                    # Calculate start position with wrapping
                    max_start = subset_size - self.num_views
                    start = (offset + wrap_idx * self.num_views) % (max_start + 1)
                    indices = list(subset_indices[start:start + self.num_views])
        else:
            # Original logic: sample from all frames
            if num_available <= self.num_views:
                indices = list(range(num_available))
                while len(indices) < self.num_views:
                    indices.append(rng.randint(0, num_available - 1))
            else:
                # Random sample without replacement
                indices = rng.sample(range(num_available), self.num_views)

            # Sort for non-subset sampling
            indices = sorted(indices)

        # Optionally force first frame as ref
        if self.first_frame_ref and len(indices) > 0:
            ref = indices[0]
            others = indices[1:]
            indices = [ref] + others

        return indices

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image using DA3-style preprocessing.

        - Resize longest side to image_size (default 504)
        - Preserve aspect ratio
        - Round dimensions to nearest multiple of PATCH_SIZE (14)
        """
        PATCH_SIZE = 14

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Resize longest side to image_size (DA3 upper_bound_resize)
        scale = self.image_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        # Round to nearest multiple of PATCH_SIZE
        new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
        new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

        # Use appropriate interpolation
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Return in [0, 1] range - VGGT normalizes internally
        img = img.astype(np.float32) / 255.0

        return img

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        # Color jitter
        if random.random() < 0.5:
            brightness = 0.8 + 0.4 * random.random()
            img = img * brightness
            img = np.clip(img, 0, 1)

        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
                - teacher_images: [8, 3, H, W] tensor in [0, 1] range
                - student_images: [4, 3, H, W] tensor in [0, 1] range
                - scene_id: str
        """
        # Map idx to scene and sample number
        if self.pairs is None:
            scene_idx = idx // self.samples_per_scene
            sample_idx = idx % self.samples_per_scene
            scene = self.scenes[scene_idx]
            pair_seed = None
        else:
            scene, pair_seed = self.pairs[idx]
            sample_idx = 0
        scene_data = self._load_scene_data(scene)

        # Get image files
        image_files = scene_data['image_files']

        # Set random seed for this specific sample (deterministic but different per sample and epoch)
        if pair_seed is not None:
            sample_seed = pair_seed
        elif self.seeds_list is not None:
            if sample_idx < len(self.seeds_list):
                sample_seed = self.seeds_list[sample_idx]
            else:
                sample_seed = self.seeds_list[sample_idx % len(self.seeds_list)]
        else:
            sample_seed = self.seed + idx
        # Exact pair studies keep the same teacher/student frame tuple every epoch.
        if pair_seed is None:
            sample_seed = sample_seed + self.current_epoch * 1000000
        rng = random.Random(sample_seed)
        random.seed(sample_seed)
        np.random.seed(sample_seed)

        # Sample view indices (pass sample_idx for consecutive window selection)
        view_indices = self._sample_view_indices(len(image_files), rng, scene=scene, sample_idx=sample_idx)

        # Determine if we should flip (consistent across all views)
        do_flip = self.augment and random.random() < 0.5

        # Load images
        images = []
        for view_idx in view_indices:
            img_path = image_files[view_idx]

            # Load image
            img = self._load_and_preprocess_image(img_path)

            # Apply augmentation
            if self.augment:
                img = self._augment_image(img)

            # Apply flip
            if do_flip:
                img = np.fliplr(img).copy()

            images.append(img)

        # Stack images: [8, H, W, 3] -> [8, 3, H, W]
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        # Teacher gets all 8 views
        teacher_images = images

        # Student gets views at specified indices
        student_images = images[self.student_indices]

        result = {
            'teacher_images': teacher_images,
            'student_images': student_images,
            'scene_id': f"{scene}_sample{sample_idx}",
            'student_frame_indices': torch.tensor(self.student_indices),
        }

        return result


def create_vggt_dataloader(
    dataset_name: str,
    batch_size: int = 2,
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the VGGT Free-Geometry dataset.

    Args:
        dataset_name: Name of benchmark dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for VGGTFreeGeometryDataset

    Returns:
        DataLoader instance
    """
    dataset = VGGTFreeGeometryDataset(
        dataset_name=dataset_name,
        **dataset_kwargs,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
