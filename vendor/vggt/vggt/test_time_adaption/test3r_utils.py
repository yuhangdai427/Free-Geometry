"""Test3R-style prompt adaptation utilities for VGGT-test3r."""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import torch

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

Triplet = Tuple[int, int, int]


@dataclass
class AdaptationStats:
    num_triplets: int
    optimizer_steps: int
    loss_mean: float
    loss_last: Optional[float]
    adaptation_time_sec: float
    time_per_triplet_sec: float
    cuda_peak_allocated_mb: Optional[float]
    cuda_peak_reserved_mb: Optional[float]
    cuda_mean_allocated_mb: Optional[float]
    cuda_last_allocated_mb: Optional[float]
    cuda_num_memory_samples: int


def build_test3r_triplets(
    num_views: int,
    seed: int = 43,
    max_triplets: Optional[int] = None,
    remove_degenerate: bool = False,
) -> List[Triplet]:
    """Build Test3R TTT triplets.

    The literal Test3R code path creates all N^3 ordered triples, shuffles
    them, and then optionally uses only a prefix if the caller caps it. Keep
    the exact all-triplets behavior by default.
    """
    triplets: List[Triplet] = []
    for i, j, k in itertools.product(range(num_views), repeat=3):
        if remove_degenerate and (i == j or i == k or j == k):
            continue
        triplets.append((i, j, k))

    rng = random.Random(seed)
    rng.shuffle(triplets)
    if max_triplets is not None:
        triplets = triplets[:max_triplets]
    return triplets


def set_trainable_prompt_only(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if not hasattr(model, "aggregator") or getattr(model.aggregator, "ttt_prompt", None) is None:
        raise RuntimeError("Expected a VGGT-test3r model with aggregator.ttt_prompt")
    model.aggregator.ttt_prompt.requires_grad_(True)


def depth_to_camera_points(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth.squeeze(0)
        elif depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
    if depth.ndim != 2:
        raise ValueError(f"Expected a single depth map, got shape {tuple(depth.shape)}")

    h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    z = depth
    x_cam = (x - cx) / fx * z
    y_cam = (y - cy) / fy * z
    return torch.stack([x_cam, y_cam, z], dim=-1)


def reference_camera_points_from_vggt(
    pred: dict, image_hw: Tuple[int, int]
) -> torch.Tensor:
    """Unproject the reference-view depth into reference-camera coordinates.

    Using camera-coord points (not world points) keeps the two pair forwards
    (I_i, I_j) and (I_i, I_k) comparable: they share the reference image
    I_i and therefore the same reference camera frame, regardless of each
    pair's own predicted world gauge.
    """
    depth = pred.get("depth")
    pose_enc = pred.get("pose_enc")
    if depth is None or pose_enc is None:
        raise KeyError("Expected VGGT predictions to contain 'depth' and 'pose_enc'")

    _, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_hw, pose_encoding_type="absT_quaR_FoV"
    )

    depth_ref = depth[0, 0]
    if depth_ref.ndim == 3:
        if depth_ref.shape[-1] == 1:
            depth_ref = depth_ref.squeeze(-1)
        elif depth_ref.shape[0] == 1:
            depth_ref = depth_ref.squeeze(0)
    if depth_ref.ndim != 2:
        raise ValueError(
            f"Unexpected reference-view depth shape: {tuple(depth_ref.shape)}"
        )

    return depth_to_camera_points(depth_ref, intrinsic[0, 0])


def ttt_consistency_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    triplet: Triplet,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
) -> torch.Tensor:
    i, j, k = triplet
    pair_ij = images[[i, j]][None]
    pair_ik = images[[i, k]][None]
    H, W = images.shape[-2:]

    device_type = images.device.type
    autocast_enabled = bool(use_amp and device_type == "cuda")
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
        pred_ij = model(pair_ij)
        pred_ik = model(pair_ik)

    points_ij = reference_camera_points_from_vggt(pred_ij, (H, W)).float()
    points_ik = reference_camera_points_from_vggt(pred_ik, (H, W)).float()

    valid = torch.isfinite(points_ij).all(dim=-1) & torch.isfinite(points_ik).all(dim=-1)
    valid = valid & (points_ij[..., 2] > 1e-4) & (points_ik[..., 2] > 1e-4)
    if int(valid.sum()) < 10:
        return (points_ij - points_ik).abs().mean() * 0.0
    return (points_ij[valid] - points_ik[valid]).abs().mean()


def _reference_camera_points_batched(
    pred: dict, image_hw: Tuple[int, int]
) -> torch.Tensor:
    """Batched version of reference_camera_points_from_vggt.

    ``pred['depth']`` has shape (M, 2, H, W, 1) or (M, 2, 1, H, W);
    ``pred['pose_enc']`` has shape (M, 2, 9). Returns (M, H, W, 3) — the
    reference-view (index 0) camera-coord pointmap for each of the M pairs.
    """
    depth = pred.get("depth")
    pose_enc = pred.get("pose_enc")
    if depth is None or pose_enc is None:
        raise KeyError("Expected VGGT predictions to contain 'depth' and 'pose_enc'")

    _, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_hw, pose_encoding_type="absT_quaR_FoV"
    )  # (M, 2, 3, 3)

    depth_ref = depth[:, 0]  # (M, H, W, 1) or (M, 1, H, W)
    if depth_ref.ndim == 4:
        if depth_ref.shape[-1] == 1:
            depth_ref = depth_ref.squeeze(-1)   # (M, H, W)
        elif depth_ref.shape[1] == 1:
            depth_ref = depth_ref.squeeze(1)    # (M, H, W)
    if depth_ref.ndim != 3:
        raise ValueError(
            f"Unexpected reference-view depth shape: {tuple(depth_ref.shape)}"
        )

    K_ref = intrinsic[:, 0]                     # (M, 3, 3)
    M, H, W = depth_ref.shape
    device = depth_ref.device
    dtype = depth_ref.dtype

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # (H, W)

    fx = K_ref[:, 0, 0].view(M, 1, 1)
    fy = K_ref[:, 1, 1].view(M, 1, 1)
    cx = K_ref[:, 0, 2].view(M, 1, 1)
    cy = K_ref[:, 1, 2].view(M, 1, 1)

    z = depth_ref
    x_cam = (x[None] - cx) / fx * z
    y_cam = (y[None] - cy) / fy * z
    return torch.stack([x_cam, y_cam, z], dim=-1)  # (M, H, W, 3)


def ttt_consistency_loss_batched(
    model: torch.nn.Module,
    images: torch.Tensor,
    triplets: Sequence[Triplet],
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
) -> torch.Tensor:
    """Mini-batched Test3R triplet loss.

    Stacks ``M = len(triplets)`` (i, j, k) triples into two pair tensors of
    shape ``[M, 2, 3, H, W]`` and runs them through VGGT in a single forward
    per side (concatenated along batch: ``[2M, 2, 3, H, W]``). Returns the
    mean reference-camera-coord L1 consistency over all M triplets.
    """
    if len(triplets) == 0:
        return images.new_zeros(())

    H, W = images.shape[-2:]
    idx_ij = torch.tensor(
        [[t[0], t[1]] for t in triplets], device=images.device, dtype=torch.long
    )  # (M, 2)
    idx_ik = torch.tensor(
        [[t[0], t[2]] for t in triplets], device=images.device, dtype=torch.long
    )  # (M, 2)

    pair_ij = images[idx_ij]   # (M, 2, 3, H, W)
    pair_ik = images[idx_ik]   # (M, 2, 3, H, W)

    M = pair_ij.shape[0]
    pair_both = torch.cat([pair_ij, pair_ik], dim=0)  # (2M, 2, 3, H, W)

    device_type = images.device.type
    autocast_enabled = bool(use_amp and device_type == "cuda")
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
        pred = model(pair_both)

    # Split pred back into the "_ij" half and the "_ik" half.
    def _split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:M], x[M:]

    pred_ij = {"depth": _split(pred["depth"])[0], "pose_enc": _split(pred["pose_enc"])[0]}
    pred_ik = {"depth": _split(pred["depth"])[1], "pose_enc": _split(pred["pose_enc"])[1]}

    points_ij = _reference_camera_points_batched(pred_ij, (H, W)).float()   # (M, H, W, 3)
    points_ik = _reference_camera_points_batched(pred_ik, (H, W)).float()

    valid = torch.isfinite(points_ij).all(dim=-1) & torch.isfinite(points_ik).all(dim=-1)
    valid = valid & (points_ij[..., 2] > 1e-4) & (points_ik[..., 2] > 1e-4)
    if int(valid.sum()) < 10:
        return (points_ij - points_ik).abs().mean() * 0.0
    return (points_ij[valid] - points_ik[valid]).abs().mean()


def adapt_prompt_one_scene(
    model: torch.nn.Module,
    images: torch.Tensor,
    seed: int = 43,
    epochs: int = 1,
    lr: float = 1e-5,
    max_triplets: Optional[int] = None,
    accum_iter: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    remove_degenerate_triplets: bool = False,
    triplet_batch_size: int = 1,
) -> AdaptationStats:
    """Run Test3R-style prompt TTA for one scene.

    When ``triplet_batch_size > 1`` the per-step loss is computed on a
    mini-batch of triplets stacked along the batch dim (one VGGT forward per
    chunk instead of two per triplet). ``accum_iter`` still controls the
    number of chunks per optimizer step.
    """
    model.train()
    model.aggregator.reset_ttt_prompt()
    set_trainable_prompt_only(model)
    device = images.device
    track_cuda = device.type == "cuda" and torch.cuda.is_available()
    if track_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    triplets = build_test3r_triplets(
        num_views=images.shape[0],
        seed=seed,
        max_triplets=max_triplets,
        remove_degenerate=remove_degenerate_triplets,
    )
    optimizer = torch.optim.AdamW(
        [model.aggregator.ttt_prompt],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)

    tbs = max(1, int(triplet_batch_size))
    chunks: List[List[Triplet]] = [triplets[i : i + tbs] for i in range(0, len(triplets), tbs)]

    losses: List[float] = []
    mem_samples_mb: List[float] = []
    optimizer_steps = 0
    for _ in range(epochs):
        for step_idx, chunk in enumerate(chunks):
            if tbs == 1:
                loss = ttt_consistency_loss(
                    model,
                    images,
                    chunk[0],
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                )
            else:
                loss = ttt_consistency_loss_batched(
                    model,
                    images,
                    chunk,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                )
            (loss / accum_iter).backward()
            losses.append(float(loss.detach().cpu()))
            if track_cuda:
                mem_samples_mb.append(torch.cuda.memory_allocated(device) / 1024 / 1024)

            is_update = (step_idx + 1) % accum_iter == 0 or (step_idx + 1) == len(chunks)
            if is_update:
                torch.nn.utils.clip_grad_norm_([model.aggregator.ttt_prompt], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    if track_cuda:
        torch.cuda.synchronize(device)
    adaptation_time_sec = time.perf_counter() - start_time
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if track_cuda else None
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024 if track_cuda else None
    mean_allocated_mb = sum(mem_samples_mb) / len(mem_samples_mb) if mem_samples_mb else None
    last_allocated_mb = mem_samples_mb[-1] if mem_samples_mb else None

    model.eval()
    return AdaptationStats(
        num_triplets=len(triplets),
        optimizer_steps=optimizer_steps,
        loss_mean=sum(losses) / max(len(losses), 1),
        loss_last=losses[-1] if losses else None,
        adaptation_time_sec=adaptation_time_sec,
        time_per_triplet_sec=adaptation_time_sec / max(len(triplets) * max(epochs, 1), 1),
        cuda_peak_allocated_mb=peak_allocated_mb,
        cuda_peak_reserved_mb=peak_reserved_mb,
        cuda_mean_allocated_mb=mean_allocated_mb,
        cuda_last_allocated_mb=last_allocated_mb,
        cuda_num_memory_samples=len(mem_samples_mb),
    )


def resize_image_to_vggt(
    image_path: Union[str, Path],
    image_size: int = 504,
    patch_size: int = 14,
) -> torch.Tensor:
    """Preprocess one image to match ``scripts/benchmark_vggt.BaseVGGT._load_images``.

    Must stay bit-identical to BaseVGGT's preprocessing so the adapted and
    baseline paths see the same input tensors.
    """
    import cv2 as _cv2
    import numpy as _np

    img = _cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    new_h = max(patch_size, round(new_h / patch_size) * patch_size)
    new_w = max(patch_size, round(new_w / patch_size) * patch_size)
    interp = _cv2.INTER_CUBIC if scale > 1.0 else _cv2.INTER_AREA
    img = _cv2.resize(img, (new_w, new_h), interpolation=interp)
    img = img.astype(_np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float()


def load_images_for_vggt(
    image_files: Sequence[Union[str, Path]],
    image_size: int = 504,
    patch_size: int = 14,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """Load a scene's images as a [S, 3, H, W] tensor on ``device``.

    Uses preprocessing that is bit-identical to
    ``scripts/benchmark_vggt.BaseVGGT._load_images`` so baseline and adapted
    runs see the same input tensors.
    """
    images = [resize_image_to_vggt(p, image_size=image_size, patch_size=patch_size) for p in image_files]
    return torch.stack(images, dim=0).to(device)


def save_prompt_checkpoint(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    meta: dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "VGGT-test3r",
            "prompt": model.aggregator.ttt_prompt.detach().cpu(),
            "meta": meta,
        },
        output_path,
    )
