"""
Generic Free-Geometry dataset for benchmark datasets.

This module provides a unified dataset loader that can load data from any
of the benchmark datasets (eth3d, 7scenes, scannetpp, hiroom, dtu) for
Free-Geometry training.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Import benchmark dataset classes
from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.bench.datasets.sevenscenes import SevenScenes
from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
from depth_anything_3.bench.datasets.dtu import DTU
from depth_anything_3.bench.datasets.dtu64 import DTU64


# ImageNet normalization (same as DA3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Dataset registry
DATASET_REGISTRY = {
    'eth3d': ETH3D,
    '7scenes': SevenScenes,
    'scannetpp': ScanNetPP,
    'hiroom': HiRoomDataset,
    'dtu': DTU,
    'dtu64': DTU64,
}


class BenchmarkFreeGeometryDataset(Dataset):
    """
    Generic Free-Geometry dataset that loads from benchmark datasets.

    Returns 8 uniformly sampled views per scene for teacher,
    and corresponding 4 views (indices [0,2,4,6]) for student.

    Args:
        dataset_name: Name of benchmark dataset ('eth3d', '7scenes', 'scannetpp', 'hiroom', 'dtu')
        num_views: Number of views for teacher (default: 8)
        image_size: Target image size (H, W), default (518, 518)
        student_indices: Which teacher views to use for student (default: [0,2,4,6])
        augment: Whether to apply data augmentation
        samples_per_scene: Number of samples to generate per scene (default: 2)
        seed: Random seed for sampling
    """

    def __init__(
        self,
        dataset_name: str,
        num_views: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        student_indices: List[int] = None,
        augment: bool = True,
        samples_per_scene: int = 2,
        seed: int = 42,
        seeds_list: Optional[List[int]] = None,
        first_frame_ref: bool = False,
        scenes: Optional[List[str]] = None,
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

        # Initialize benchmark dataset
        self.benchmark_dataset = DATASET_REGISTRY[dataset_name]()

        # Get all scenes (no train/val split - use all scenes)
        available_scenes = list(self.benchmark_dataset.SCENES)
        if scenes is None:
            self.scenes = available_scenes
        else:
            requested = list(scenes)
            unknown = sorted(set(requested) - set(available_scenes))
            if unknown:
                raise ValueError(
                    f"Unknown scenes for {dataset_name}: {unknown}. "
                    f"Available: {available_scenes}"
                )
            self.scenes = requested

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found for {dataset_name}")

        print(f"Loaded {len(self.scenes)} scenes for {dataset_name} (all scenes for training)")
        print(f"Generating {samples_per_scene} samples per scene = {len(self.scenes) * samples_per_scene} total samples")

        # Image transforms
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Cache for scene data
        self._scene_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self.scenes) * self.samples_per_scene

    def _load_scene_data(self, scene: str) -> Dict:
        """Load scene data from benchmark dataset."""
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        # Get data from benchmark dataset
        data = self.benchmark_dataset.get_data(scene)

        self._scene_cache[scene] = data
        return data

    def _sample_view_indices(self, num_available: int, rng: random.Random) -> List[int]:
        """
        Uniformly sample view indices from available images.

        Args:
            num_available: Number of available images in scene

        Returns:
            List of indices to sample
        """
        if num_available <= self.num_views:
            indices = list(range(num_available))
            while len(indices) < self.num_views:
                indices.append(rng.randint(0, num_available - 1))
        else:
            # Random sample without replacement
            indices = rng.sample(range(num_available), self.num_views)
        # Ensure deterministic order and optionally force first frame as ref
        indices = sorted(indices)
        if self.first_frame_ref and len(indices) > 0:
            ref = indices[0]
            others = indices[1:]
            indices = [ref] + others
        return indices

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
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
                - teacher_images: [8, 3, H, W] tensor
                - student_images: [4, 3, H, W] tensor
                - scene_id: str
        """
        # Map idx to scene and sample number
        scene_idx = idx // self.samples_per_scene
        sample_idx = idx % self.samples_per_scene

        scene = self.scenes[scene_idx]
        scene_data = self._load_scene_data(scene)

        # Get image files
        image_files = scene_data['image_files']

        # Set random seed for this specific sample (deterministic but different per sample)
        if self.seeds_list is not None:
            if sample_idx < len(self.seeds_list):
                sample_seed = self.seeds_list[sample_idx]
            else:
                sample_seed = self.seeds_list[sample_idx % len(self.seeds_list)]
        else:
            sample_seed = self.seed + idx
        rng = random.Random(sample_seed)
        random.seed(sample_seed)
        np.random.seed(sample_seed)

        # Sample view indices
        view_indices = self._sample_view_indices(len(image_files), rng)

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

        # Normalize
        images = self.normalize(images)

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


def create_benchmark_dataloader(
    dataset_name: str,
    batch_size: int = 2,
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the benchmark Free-Geometry dataset.

    Args:
        dataset_name: Name of benchmark dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for BenchmarkFreeGeometryDataset

    Returns:
        DataLoader instance
    """
    dataset = BenchmarkFreeGeometryDataset(
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
