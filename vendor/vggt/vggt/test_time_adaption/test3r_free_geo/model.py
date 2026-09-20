"""VGGT Test3R-style per-scene LoRA adaptation.

This module intentionally does not use the VGGT-test3r prompt path. It keeps the
base VGGT weights frozen, inserts PEFT LoRA adapters into the alternating
frame/global encoder blocks, and updates only those LoRA weights with a
Test3R-style triplet point consistency loss from VGGT's point head.
"""

from __future__ import annotations

import itertools
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from huggingface_hub import snapshot_download

from vggt.models.vggt import VGGT

Triplet = Tuple[int, int, int]


def build_test3r_triplets(
    num_views: int,
    seed: int = 43,
    max_triplets: Optional[int] = None,
    remove_degenerate: bool = False,
) -> List[Triplet]:
    """Build the ordered N^3 Test3R triplet set and shuffle it deterministically."""
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


@dataclass
class FreeGeoLoRAAdaptationStats:
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


class LoRALinear(nn.Module):
    """LoRA adapter around a frozen Linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling


class VGGTTest3RFreeGeoLoRAModel(nn.Module):
    """Frozen VGGT with trainable LoRA adapters in the alternating encoder."""

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_layers: Optional[Sequence[int]] = None,
        lora_target: str = "attention_mlp",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_layers = list(lora_layers) if lora_layers is not None else list(range(24))
        self.lora_target = lora_target

        print(f"Loading VGGT base model: {model_name}")
        self.vggt = self._load_vggt(model_name)
        for param in self.vggt.parameters():
            param.requires_grad_(False)

        target_modules = self._build_lora_targets()
        print(
            "Applying Free-Geo LoRA: "
            f"rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
            f"layers={self.lora_layers[0]}-{self.lora_layers[-1]}, target={lora_target}"
        )
        self._inject_lora_modules(target_modules)
        self.set_lora_trainable_only()
        self._initial_lora_state = self._clone_trainable_state()

        total = sum(p.numel() for p in self.vggt.parameters())
        trainable = sum(p.numel() for p in self.vggt.parameters() if p.requires_grad)
        print(f"VGGT Free-Geo LoRA parameters: {trainable:,} trainable / {total:,} total")

    @staticmethod
    def _load_vggt(model_name: str) -> VGGT:
        """Load VGGT from cached model.pt when available, avoiding safetensors issues."""
        if os.path.isdir(model_name):
            model_pt = os.path.join(model_name, "model.pt")
            if os.path.exists(model_pt):
                model = VGGT()
                state = torch.load(model_pt, map_location="cpu", weights_only=False)
                model.load_state_dict(state, strict=True)
                return model

        try:
            local_snapshot = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
                allow_patterns=["model.pt", "config.json"],
            )
            model_pt = os.path.join(local_snapshot, "model.pt")
            if os.path.exists(model_pt):
                model = VGGT()
                state = torch.load(model_pt, map_location="cpu", weights_only=False)
                model.load_state_dict(state, strict=True)
                return model
        except Exception as exc:
            print(f"Local model.pt load failed, falling back to from_pretrained: {exc}")

        return VGGT.from_pretrained(model_name)

    def _build_lora_targets(self) -> List[str]:
        if self.lora_target not in {"attention", "attention_mlp"}:
            raise ValueError("--lora_target must be one of: attention, attention_mlp")

        targets: List[str] = []
        for layer_idx in self.lora_layers:
            for block_name in ("frame_blocks", "global_blocks"):
                prefix = f"aggregator.{block_name}.{layer_idx}"
                targets.extend([f"{prefix}.attn.qkv", f"{prefix}.attn.proj"])
                if self.lora_target == "attention_mlp":
                    targets.extend([f"{prefix}.mlp.fc1", f"{prefix}.mlp.fc2"])
        return targets

    def _resolve_parent_module(self, dotted_path: str) -> tuple[nn.Module, str]:
        parts = dotted_path.split(".")
        parent: nn.Module = self.vggt
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]  # type: ignore[index]
            else:
                parent = getattr(parent, part)
        return parent, parts[-1]

    def _inject_lora_modules(self, target_modules: Sequence[str]) -> None:
        replaced = 0
        for module_path in target_modules:
            parent, child_name = self._resolve_parent_module(module_path)
            base_layer = getattr(parent, child_name)
            if not isinstance(base_layer, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {module_path}, got {type(base_layer).__name__}")
            setattr(
                parent,
                child_name,
                LoRALinear(
                    base_layer=base_layer,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                ),
            )
            replaced += 1
        print(f"Inserted local LoRA adapters into {replaced} linear modules")

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.vggt(images)

    def set_lora_trainable_only(self) -> None:
        for param in self.vggt.parameters():
            param.requires_grad_(False)
        for name, param in self.vggt.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    def trainable_lora_parameters(self) -> List[nn.Parameter]:
        return [param for param in self.vggt.parameters() if param.requires_grad]

    def _clone_trainable_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().clone().cpu()
            for name, param in self.vggt.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def reset_lora_parameters(self) -> None:
        state_by_name = dict(self.vggt.named_parameters())
        for name, value in self._initial_lora_state.items():
            state_by_name[name].copy_(value.to(device=state_by_name[name].device, dtype=state_by_name[name].dtype))

    def save_lora_checkpoint(
        self,
        output_path: Union[str, Path],
        meta: Optional[dict] = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": "VGGT-test3r-free-geo-lora",
                "meta": meta or {},
                "lora_state": self._clone_trainable_state(),
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "lora_layers": self.lora_layers,
                "lora_target": self.lora_target,
            },
            output_path,
        )


def _reference_world_points(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
    world_points = predictions.get("world_points")
    if world_points is None:
        raise KeyError("Expected VGGT point head output 'world_points'")
    if world_points.ndim == 5:
        return world_points[0, 0]
    if world_points.ndim == 4:
        return world_points[0]
    if world_points.ndim == 3:
        return world_points
    raise ValueError(f"Unexpected world_points shape: {tuple(world_points.shape)}")


def point_head_triplet_loss(
    model: VGGTTest3RFreeGeoLoRAModel,
    images: torch.Tensor,
    triplet: Triplet,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
) -> torch.Tensor:
    """Compare reference-frame point-head outputs from (i, j) and (i, k)."""
    i, j, k = triplet
    pair_ij = images[[i, j]][None]
    pair_ik = images[[i, k]][None]

    device_type = images.device.type
    autocast_enabled = bool(use_amp and device_type == "cuda")
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
        pred_ij = model(pair_ij)
        pred_ik = model(pair_ik)

    points_ij = _reference_world_points(pred_ij).float()
    points_ik = _reference_world_points(pred_ik).float()
    valid = torch.isfinite(points_ij).all(dim=-1) & torch.isfinite(points_ik).all(dim=-1)
    valid = valid & (points_ij[..., 2] > 1e-4) & (points_ik[..., 2] > 1e-4)

    if int(valid.sum()) < 10:
        return (points_ij - points_ik).abs().mean() * 0.0
    return (points_ij[valid] - points_ik[valid]).abs().mean()


def adapt_lora_one_scene(
    model: VGGTTest3RFreeGeoLoRAModel,
    images: torch.Tensor,
    seed: int = 43,
    epochs: int = 1,
    lr: float = 1e-5,
    max_triplets: Optional[int] = None,
    accum_iter: int = 2,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    remove_degenerate_triplets: bool = False,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> FreeGeoLoRAAdaptationStats:
    """Reset and adapt LoRA weights for one scene using Test3R triplets."""
    model.train()
    model.reset_lora_parameters()
    model.set_lora_trainable_only()

    trainable_params = model.trainable_lora_parameters()
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found")

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
        trainable_params,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    losses: List[float] = []
    mem_samples_mb: List[float] = []
    optimizer_steps = 0
    for _ in range(epochs):
        for step_idx, triplet in enumerate(triplets):
            loss = point_head_triplet_loss(
                model=model,
                images=images,
                triplet=triplet,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            (loss / accum_iter).backward()
            losses.append(float(loss.detach().cpu()))
            if track_cuda:
                mem_samples_mb.append(torch.cuda.memory_allocated(device) / 1024 / 1024)

            is_update = (step_idx + 1) % accum_iter == 0 or (step_idx + 1) == len(triplets)
            if is_update:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
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
    return FreeGeoLoRAAdaptationStats(
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
