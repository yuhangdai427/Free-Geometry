#!/usr/bin/env python3
"""VGGT benchmark-dataset trainer for the active Free-Geometry LoRA workflow."""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))

from vggt.vggt.test_time_adaption.dataset import VGGTFreeGeometryDataset
from vggt.vggt.test_time_adaption.models import VGGTFreeGeometryOutput, VGGTStudentModel, VGGTTeacherModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
    steps_per_epoch: int | None = None,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if scheduler_type in (None, "none"):
        return None

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = None
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps - warmup_steps),
            eta_min=eta_min,
        )
        if warmup_scheduler is None:
            return cosine_scheduler
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    if scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR

        step_size = max(1, steps_per_epoch or num_training_steps)
        return StepLR(optimizer, step_size=step_size, gamma=0.7)

    if scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, num_training_steps),
        )

    if scheduler_type == "constant":
        from torch.optim.lr_scheduler import ConstantLR

        return ConstantLR(optimizer, factor=1.0, total_iters=max(1, num_training_steps))

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class VGGTPatchHuberCosineLoss(nn.Module):
    """Per-patch Huber + cosine loss on full VGGT tokens."""

    def __init__(
        self,
        student_frame_indices: Optional[List[int]] = None,
        target_layers: Optional[List[int]] = None,
        huber_weight: float = 1.0,
        cos_weight: float = 2.0,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_layers = target_layers or [23]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.huber_weight = huber_weight
        self.cos_weight = cos_weight
        self.huber = nn.SmoothL1Loss(reduction="mean", beta=delta)
        print(
            f"VGGTPatchHuberCosineLoss: layers={self.target_layers}, "
            f"huber_w={huber_weight}, cos_w={cos_weight}, delta={delta}"
        )

    def forward(
        self,
        teacher_output: VGGTFreeGeometryOutput,
        student_output: VGGTFreeGeometryOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_huber = 0.0
        total_cos = 0.0

        for layer in self.target_layers:
            teacher_all = teacher_output.layer_features[layer]
            student_all = student_output.layer_features[layer]
            teacher_selected = teacher_all[:, self.student_frame_indices, :, :].detach()

            huber = self.huber(student_all, teacher_selected)
            teacher_norm = torch.nn.functional.normalize(teacher_selected, dim=-1)
            student_norm = torch.nn.functional.normalize(student_all, dim=-1)
            cos = (teacher_norm * student_norm).sum(dim=-1).mean()

            total_huber = total_huber + huber
            total_cos = total_cos + cos

        n_layers = len(self.target_layers)
        avg_huber = total_huber / n_layers
        avg_cos = total_cos / n_layers
        loss = self.huber_weight * avg_huber + self.cos_weight * (1.0 - avg_cos)

        return loss, {
            "patch_huber": avg_huber.item(),
            "patch_huber_cos": avg_cos.item(),
            "patch_huber_total": loss.item(),
            "total_loss": loss.item(),
        }


class VGGTCrossFrameCFAngleLoss(nn.Module):
    """Cross-frame CF angle loss for VGGT."""

    def __init__(
        self,
        student_frame_indices: Optional[List[int]] = None,
        num_teacher_views: int = 8,
        target_layer: int = 23,
        topk: int = 4,
        num_ref_samples: int = 128,
        num_shared_samples: int = 128,
        angle1_weight: float = 1.0,
        angle2_weight: float = 1.0,
        angle3_weight: float = 1.0,
        huber_delta: float = 1.0,
        shared_chunk_size: int = 64,
        selection_mode: str = "topk",
    ) -> None:
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.num_teacher_views = num_teacher_views
        self.target_layer = target_layer
        self.topk = topk
        self.num_ref_samples = num_ref_samples
        self.num_shared_samples = num_shared_samples
        self.angle1_weight = angle1_weight
        self.angle2_weight = angle2_weight
        self.angle3_weight = angle3_weight
        self.huber_loss = nn.HuberLoss(reduction="none", delta=huber_delta)
        self.shared_chunk_size = shared_chunk_size
        self.selection_mode = selection_mode

        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)
        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]
        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(f"VGGTCrossFrameCFAngleLoss: layer={target_layer}, topk={topk}, selection={selection_mode}")
        print(
            f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, "
            f"shared_chunk={shared_chunk_size}"
        )
        print(f"  angle weights: a1={angle1_weight}, a2={angle2_weight}, a3={angle3_weight}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")

    @staticmethod
    def _cos_angle(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
        a_norm = torch.nn.functional.normalize(vec_a, dim=-1, eps=1e-8)
        b_norm = torch.nn.functional.normalize(vec_b, dim=-1, eps=1e-8)
        return (a_norm * b_norm).sum(dim=-1)

    def forward(
        self,
        teacher_output: VGGTFreeGeometryOutput,
        student_output: VGGTFreeGeometryOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]
        student_feats = student_output.layer_features[self.target_layer]

        batch_size, _, num_patches, dim = teacher_feats.shape
        num_ref = min(self.num_ref_samples, num_patches)
        num_shared = min(self.num_shared_samples, num_patches)
        ref_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_shared]

        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]

        with torch.no_grad():
            extra_t = torch.cat(
                [teacher_feats[:, frame_idx, :, :] for frame_idx in self.extra_frame_indices],
                dim=1,
            )

            if self.selection_mode == "random":
                ep = extra_t.shape[1]
                rand_indices = torch.randint(0, ep, (batch_size, num_ref, self.topk), device=teacher_feats.device)
                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = rand_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(num_ref, self.topk, dim)
            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))

                if self.selection_mode == "mixed":
                    k_top = self.topk // 2
                    k_bot = self.topk - k_top
                    _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                    _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                    gather_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                else:
                    _, gather_indices = sim_matrix.topk(self.topk, dim=-1)

                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = gather_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(num_ref, self.topk, dim)

        sim_high_t = sim_high_t.detach()

        sum_angle1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle3 = torch.tensor(0.0, device=teacher_feats.device)
        total_elements = 0
        chunk_size = self.shared_chunk_size

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]
            shared_t_full = teacher_feats[:, shared_teacher_idx, shared_perm, :].detach()
            shared_s_full = student_feats[:, shared_student_idx, shared_perm, :]

            for ref_start in range(0, num_ref, chunk_size):
                ref_end = min(ref_start + chunk_size, num_ref)
                ref_count = ref_end - ref_start

                ref_t_4d = ref_t_sampled[:, ref_start:ref_end, :].detach().unsqueeze(2).unsqueeze(3)
                ref_s_4d = ref_s_sampled[:, ref_start:ref_end, :].unsqueeze(2).unsqueeze(3)
                sim_high_4d = sim_high_t[:, ref_start:ref_end, :, :].unsqueeze(2)

                for shared_start in range(0, num_shared, chunk_size):
                    shared_end = min(shared_start + chunk_size, num_shared)
                    shared_count = shared_end - shared_start

                    shared_t_4d = shared_t_full[:, shared_start:shared_end, :].unsqueeze(1).unsqueeze(3)
                    shared_s_4d = shared_s_full[:, shared_start:shared_end, :].unsqueeze(1).unsqueeze(3)

                    angle1_t = self._cos_angle(shared_t_4d - ref_t_4d, sim_high_4d - ref_t_4d)
                    angle1_s = self._cos_angle(shared_s_4d - ref_s_4d, sim_high_4d - ref_s_4d)
                    sum_angle1 = sum_angle1 + self.huber_loss(angle1_s, angle1_t.detach()).sum()

                    angle2_t = self._cos_angle(ref_t_4d - sim_high_4d, shared_t_4d - sim_high_4d)
                    angle2_s = self._cos_angle(ref_s_4d - sim_high_4d, shared_s_4d - sim_high_4d)
                    sum_angle2 = sum_angle2 + self.huber_loss(angle2_s, angle2_t.detach()).sum()

                    angle3_t = self._cos_angle(ref_t_4d - shared_t_4d, sim_high_4d - shared_t_4d)
                    angle3_s = self._cos_angle(ref_s_4d - shared_s_4d, sim_high_4d - shared_s_4d)
                    sum_angle3 = sum_angle3 + self.huber_loss(angle3_s, angle3_t.detach()).sum()

                    total_elements += batch_size * ref_count * shared_count * self.topk

        total_angle1_loss = sum_angle1 / total_elements
        total_angle2_loss = sum_angle2 / total_elements
        total_angle3_loss = sum_angle3 / total_elements
        loss = (
            self.angle1_weight * total_angle1_loss
            + self.angle2_weight * total_angle2_loss
            + self.angle3_weight * total_angle3_loss
        )

        return loss, {
            "cf_angle1_loss": total_angle1_loss.item(),
            "cf_angle2_loss": total_angle2_loss.item(),
            "cf_angle3_loss": total_angle3_loss.item(),
            "cf_total_loss": loss.item(),
        }


class VGGTCrossFrameCFDistanceLoss(nn.Module):
    """Cross-frame CF distance loss for VGGT."""

    def __init__(
        self,
        student_frame_indices: Optional[List[int]] = None,
        num_teacher_views: int = 8,
        target_layer: int = 23,
        topk: int = 4,
        num_ref_samples: int = 256,
        num_shared_samples: int = 256,
        d1_weight: float = 1.0,
        d2_weight: float = 1.0,
        d3_weight: float = 1.0,
        shared_chunk_size: int = 64,
        distance_chunk_size: int = 16,
        distance_type: str = "l2",
        normalize_distance: bool = True,
        temperature: float = 1.0,
        distance_mode: str = "kl",
        huber_beta: float = 0.5,
        selection_mode: str = "topk",
    ) -> None:
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.num_teacher_views = num_teacher_views
        self.target_layer = target_layer
        self.topk = topk
        self.num_ref_samples = num_ref_samples
        self.num_shared_samples = num_shared_samples
        self.d1_weight = d1_weight
        self.d2_weight = d2_weight
        self.d3_weight = d3_weight
        self.shared_chunk_size = shared_chunk_size
        self.distance_chunk_size = distance_chunk_size
        self.distance_type = distance_type
        self.normalize_distance = normalize_distance
        self.temperature = temperature
        self.distance_mode = distance_mode
        self.huber_beta = huber_beta
        self.selection_mode = selection_mode

        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)
        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]
        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(
            f"VGGTCrossFrameCFDistanceLoss: layer={target_layer}, topk={topk}, "
            f"temp={temperature}, mode={distance_mode}, selection={selection_mode}"
        )
        print(
            f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, "
            f"shared_chunk={shared_chunk_size}, distance_chunk={distance_chunk_size}"
        )
        print(f"  distance weights: d1={d1_weight}, d2={d2_weight}, d3={d3_weight}")
        print(f"  distance_type={distance_type}, normalize={normalize_distance}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")

    def forward(
        self,
        teacher_output: VGGTFreeGeometryOutput,
        student_output: VGGTFreeGeometryOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]
        student_feats = student_output.layer_features[self.target_layer]

        batch_size, _, num_patches, dim = teacher_feats.shape
        num_ref = min(self.num_ref_samples, num_patches)
        num_shared = min(self.num_shared_samples, num_patches)
        ref_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_shared]

        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]

        with torch.no_grad():
            extra_t = torch.cat(
                [teacher_feats[:, frame_idx, :, :] for frame_idx in self.extra_frame_indices],
                dim=1,
            )

            if self.selection_mode == "random":
                ep = extra_t.shape[1]
                rand_indices = torch.randint(0, ep, (batch_size, num_ref, self.topk), device=teacher_feats.device)
                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = rand_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(num_ref, self.topk, dim)
            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))

                if self.selection_mode == "mixed":
                    k_top = self.topk // 2
                    k_bot = self.topk - k_top
                    _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                    _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                    gather_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                else:
                    _, gather_indices = sim_matrix.topk(self.topk, dim=-1)

                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = gather_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(num_ref, self.topk, dim)

        sim_high_t = sim_high_t.detach()

        paired_count = min(num_ref, num_shared)
        chunk_size = self.distance_chunk_size
        huber_outer = nn.SmoothL1Loss(reduction="none", beta=0.5)
        huber_direct = nn.SmoothL1Loss(reduction="none", beta=self.huber_beta)

        sum_d1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d3 = torch.tensor(0.0, device=teacher_feats.device)
        total_d1_elements = 0
        total_d2_elements = 0
        total_d3_elements = 0

        ref_t_paired = ref_t_sampled[:, :paired_count, :]
        ref_s_paired = ref_s_sampled[:, :paired_count, :]
        sim_high_paired = sim_high_t[:, :paired_count, :, :]

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]
            shared_t = teacher_feats[:, shared_teacher_idx, shared_perm[:paired_count], :].detach()
            shared_s = student_feats[:, shared_student_idx, shared_perm[:paired_count], :]

            for start in range(0, paired_count, chunk_size):
                end = min(start + chunk_size, paired_count)
                ref_t = ref_t_paired[:, start:end, :].detach()
                ref_s = ref_s_paired[:, start:end, :]
                shared_t_chunk = shared_t[:, start:end, :]
                shared_s_chunk = shared_s[:, start:end, :]
                sim_high = sim_high_paired[:, start:end, :, :]

                diff_d1_t = ref_t - shared_t_chunk
                diff_d1_s = ref_s - shared_s_chunk
                diff_d2_t = ref_t.unsqueeze(2) - sim_high
                diff_d2_s = ref_s.unsqueeze(2) - sim_high
                diff_d3_t = shared_t_chunk.unsqueeze(2) - sim_high
                diff_d3_s = shared_s_chunk.unsqueeze(2) - sim_high

                if self.distance_mode == "huber":
                    loss_d1_chunk = huber_direct(diff_d1_s, diff_d1_t).mean(dim=-1)
                    loss_d2_chunk = huber_direct(diff_d2_s, diff_d2_t).mean(dim=-1)
                    loss_d3_chunk = huber_direct(diff_d3_s, diff_d3_t).mean(dim=-1)
                    sum_d1 = sum_d1 + loss_d1_chunk.sum()
                    sum_d2 = sum_d2 + loss_d2_chunk.sum()
                    sum_d3 = sum_d3 + loss_d3_chunk.sum()
                    total_d1_elements += loss_d1_chunk.numel()
                    total_d2_elements += loss_d2_chunk.numel()
                    total_d3_elements += loss_d3_chunk.numel()
                else:
                    log_p_d1 = torch.log_softmax(diff_d1_t / self.temperature, dim=-1)
                    log_q_d1 = torch.log_softmax(diff_d1_s / self.temperature, dim=-1)
                    kl_d1 = (log_p_d1.exp() * (log_p_d1 - log_q_d1)).sum(dim=-1)
                    sum_d1 = sum_d1 + huber_outer(kl_d1, torch.zeros_like(kl_d1)).sum()
                    total_d1_elements += kl_d1.numel()

                    log_p_d2 = torch.log_softmax(diff_d2_t / self.temperature, dim=-1)
                    log_q_d2 = torch.log_softmax(diff_d2_s / self.temperature, dim=-1)
                    kl_d2 = (log_p_d2.exp() * (log_p_d2 - log_q_d2)).sum(dim=-1)
                    sum_d2 = sum_d2 + huber_outer(kl_d2, torch.zeros_like(kl_d2)).sum()
                    total_d2_elements += kl_d2.numel()

                    log_p_d3 = torch.log_softmax(diff_d3_t / self.temperature, dim=-1)
                    log_q_d3 = torch.log_softmax(diff_d3_s / self.temperature, dim=-1)
                    kl_d3 = (log_p_d3.exp() * (log_p_d3 - log_q_d3)).sum(dim=-1)
                    sum_d3 = sum_d3 + huber_outer(kl_d3, torch.zeros_like(kl_d3)).sum()
                    total_d3_elements += kl_d3.numel()

        loss_d1 = sum_d1 / max(total_d1_elements, 1)
        loss_d2 = sum_d2 / max(total_d2_elements, 1)
        loss_d3 = sum_d3 / max(total_d3_elements, 1)
        loss = self.d1_weight * loss_d1 + self.d2_weight * loss_d2 + self.d3_weight * loss_d3

        return loss, {
            "cf_dist_d1_loss": loss_d1.item(),
            "cf_dist_d2_loss": loss_d2.item(),
            "cf_dist_d3_loss": loss_d3.item(),
            "cf_dist_total_loss": loss.item(),
        }


def save_checkpoint(
    student: VGGTStudentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
    }

    lora_path = os.path.join(output_dir, filename.replace(".pt", "_lora.pt"))
    student.save_lora_weights(lora_path)
    checkpoint["lora_path"] = lora_path

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    latest_path = os.path.join(output_dir, "latest.pt")
    latest_lora_path = os.path.join(output_dir, "latest_lora.pt")
    torch.save(checkpoint, latest_path)
    student.save_lora_weights(latest_lora_path)

    return checkpoint_path


def _to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def train_epoch(
    teacher: VGGTTeacherModel,
    student: VGGTStudentModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    args,
    writer: Optional[SummaryWriter] = None,
    cf_criterion: Optional[nn.Module] = None,
    cf_distance_criterion: Optional[nn.Module] = None,
) -> Tuple[int, Dict[str, float]]:
    student.train()
    teacher.eval()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    teacher_peak_bytes = 0
    training_peak_bytes = 0

    for batch_idx, batch in enumerate(train_loader):
        teacher_images = batch["teacher_images"].to(args.device)
        student_images = batch["student_images"].to(args.device)

        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)

        with autocast(enabled=args.use_amp):
            with torch.no_grad():
                teacher_output = teacher(teacher_images)
            if args.device.type == "cuda":
                teacher_peak_bytes = max(teacher_peak_bytes, torch.cuda.max_memory_allocated(args.device))
            student_output = student(student_images)

            feat_loss, feat_details = criterion(teacher_output, student_output)
            combined_loss = feat_loss
            cf_loss = torch.tensor(0.0, device=args.device)
            cf_dist_loss = torch.tensor(0.0, device=args.device)
            cf_details: Dict[str, float] = {}
            cf_dist_details: Dict[str, float] = {}

            if cf_criterion is not None:
                cf_loss, cf_details = cf_criterion(teacher_output, student_output)
                combined_loss = combined_loss + args.cf_weight * cf_loss

            if cf_distance_criterion is not None:
                cf_dist_loss, cf_dist_details = cf_distance_criterion(teacher_output, student_output)
                combined_loss = combined_loss + args.cf_distance_weight * cf_dist_loss

            loss = combined_loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.get_trainable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        total_loss += _to_float(combined_loss)
        num_batches += 1
        if args.device.type == "cuda":
            training_peak_bytes = max(training_peak_bytes, torch.cuda.max_memory_allocated(args.device))

        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * args.batch_size / max(elapsed, 1e-6)

            detail_parts = [
                f"feat: {_to_float(feat_loss):.4f}",
                f"cf: {_to_float(cf_loss):.4f}",
                f"cf_dist: {_to_float(cf_dist_loss):.4f}",
                f"combined: {_to_float(combined_loss):.4f}",
            ]
            for key in ("patch_huber", "patch_huber_cos", "patch_huber_total"):
                if key in feat_details:
                    detail_parts.append(f"{key}: {_to_float(feat_details[key]):.4f}")
            for key in ("cf_angle1_loss", "cf_angle2_loss", "cf_angle3_loss"):
                if key in cf_details:
                    detail_parts.append(f"{key}: {_to_float(cf_details[key]):.4f}")
            for key in ("cf_dist_d1_loss", "cf_dist_d2_loss", "cf_dist_d3_loss"):
                if key in cf_dist_details:
                    detail_parts.append(f"{key}: {_to_float(cf_dist_details[key]):.4f}")

            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )
            print(f"  {' | '.join(detail_parts)}")

            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/feat_loss", _to_float(feat_loss), global_step)
                writer.add_scalar("train/cf_loss", _to_float(cf_loss), global_step)
                writer.add_scalar("train/cf_dist_loss", _to_float(cf_dist_loss), global_step)
                writer.add_scalar("train/combined_loss", _to_float(combined_loss), global_step)
                for key, value in feat_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in cf_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in cf_dist_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)

        if args.save_interval > 0 and global_step % args.save_interval == 0:
            save_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                total_loss / num_batches,
                args.output_dir,
                f"checkpoint_step{global_step}.pt",
            )

    return global_step, {
        "wall_clock_seconds": time.time() - start_time,
        "teacher_forward_peak_bytes": teacher_peak_bytes,
        "training_peak_bytes": training_peak_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT Free-Geometry training on benchmark datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        default="scannetpp",
        choices=["scannetpp", "eth3d", "7scenes", "hiroom", "dtu"],
    )
    parser.add_argument("--samples_per_scene", type=int, default=4)
    parser.add_argument("--seeds_list", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument(
        "--student_indices",
        type=int,
        nargs="+",
        default=None,
        help="Teacher-frame positions for the student; default is all even positions.",
    )
    parser.add_argument(
        "--nested_indices_file",
        type=str,
        default=None,
        help="E6 P16 JSONL generated by scripts/generate_e6_view_indices.py.",
    )
    parser.add_argument("--image_size", type=int, default=504)

    parser.add_argument("--model_name", type=str, default="facebook/vggt-1b")
    parser.add_argument("--output_layers", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_layers_start", type=int, default=12)
    parser.add_argument("--train_camera_token", action="store_true", default=True)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["cosine", "linear", "constant", "step", "none"],
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vggt_tta")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)

    parser.add_argument("--patch_huber_weight", type=float, default=1.0)
    parser.add_argument("--patch_huber_cos_weight", type=float, default=2.0)
    parser.add_argument("--patch_huber_delta", type=float, default=1.0)

    parser.add_argument("--cf_weight", type=float, default=1.0)
    parser.add_argument("--cf_topk", type=int, default=4)
    parser.add_argument("--cf_num_ref_samples", type=int, default=128)
    parser.add_argument("--cf_num_shared_samples", type=int, default=128)
    parser.add_argument("--cf_angle1_weight", type=float, default=1.0)
    parser.add_argument("--cf_angle2_weight", type=float, default=1.0)
    parser.add_argument("--cf_angle3_weight", type=float, default=1.0)
    parser.add_argument("--cf_shared_chunk_size", type=int, default=64)
    parser.add_argument(
        "--cf_selection_mode",
        type=str,
        default="topk",
        choices=["topk", "random", "mixed"],
    )
    parser.add_argument("--use_cf_distance", action="store_true")
    parser.add_argument("--cf_distance_weight", type=float, default=1.0)
    parser.add_argument("--cf_distance_chunk_size", type=int, default=16)
    parser.add_argument("--cf_distance_type", type=str, default="l2", choices=["l2", "cosine"])
    parser.add_argument("--cf_normalize_distance", action="store_true", default=True)
    parser.add_argument("--cf_no_normalize_distance", action="store_true")
    parser.add_argument("--cf_d1_weight", type=float, default=1.0)
    parser.add_argument("--cf_d2_weight", type=float, default=1.0)
    parser.add_argument("--cf_d3_weight", type=float, default=1.0)
    parser.add_argument("--cf_distance_temperature", type=float, default=1.0)
    parser.add_argument("--cf_distance_mode", type=str, default="kl", choices=["kl", "huber"])
    parser.add_argument("--cf_distance_huber_beta", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.cf_no_normalize_distance:
        args.cf_normalize_distance = False

    student_indices = args.student_indices if args.student_indices is not None else list(range(0, args.num_views, 2))
    if not student_indices or any(index < 0 or index >= args.num_views for index in student_indices):
        parser.error("--student_indices must be non-empty positions in [0, --num_views)")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {args.device}")
    print("\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Teacher views: {args.num_views}")
    print(f"  Student views: {len(student_indices)} (indices: {student_indices})")
    print(f"  Output layers: {args.output_layers}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  LoRA layers: {args.lora_layers_start}-23")

    print("\nCreating dataset...")
    train_dataset = VGGTFreeGeometryDataset(
        dataset_name=args.dataset,
        num_views=args.num_views,
        image_size=args.image_size,
        student_indices=student_indices,
        augment=True,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
        seeds_list=args.seeds_list,
        nested_indices_file=args.nested_indices_file,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train samples: {len(train_dataset)}")

    print("\nCreating models...")
    teacher = VGGTTeacherModel(
        model_name=args.model_name,
        output_layers=args.output_layers,
    ).to(args.device)
    student = VGGTStudentModel(
        model_name=args.model_name,
        output_layers=args.output_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_camera_token=args.train_camera_token,
        lora_layers=list(range(args.lora_layers_start, 24)),
    ).to(args.device)

    criterion = VGGTPatchHuberCosineLoss(
        student_frame_indices=student_indices,
        target_layers=args.output_layers,
        huber_weight=args.patch_huber_weight,
        cos_weight=args.patch_huber_cos_weight,
        delta=args.patch_huber_delta,
    )
    target_layer = args.output_layers[-1]
    cf_criterion = VGGTCrossFrameCFAngleLoss(
        student_frame_indices=student_indices,
        num_teacher_views=args.num_views,
        target_layer=target_layer,
        topk=args.cf_topk,
        num_ref_samples=args.cf_num_ref_samples,
        num_shared_samples=args.cf_num_shared_samples,
        angle1_weight=args.cf_angle1_weight,
        angle2_weight=args.cf_angle2_weight,
        angle3_weight=args.cf_angle3_weight,
        shared_chunk_size=args.cf_shared_chunk_size,
        selection_mode=args.cf_selection_mode,
    )

    cf_distance_criterion = None
    if args.use_cf_distance:
        cf_distance_criterion = VGGTCrossFrameCFDistanceLoss(
            student_frame_indices=student_indices,
            num_teacher_views=args.num_views,
            target_layer=target_layer,
            topk=args.cf_topk,
            num_ref_samples=args.cf_num_ref_samples,
            num_shared_samples=args.cf_num_shared_samples,
            d1_weight=args.cf_d1_weight,
            d2_weight=args.cf_d2_weight,
            d3_weight=args.cf_d3_weight,
            shared_chunk_size=args.cf_shared_chunk_size,
            distance_chunk_size=args.cf_distance_chunk_size,
            distance_type=args.cf_distance_type,
            normalize_distance=args.cf_normalize_distance,
            temperature=args.cf_distance_temperature,
            distance_mode=args.cf_distance_mode,
            huber_beta=args.cf_distance_huber_beta,
            selection_mode=args.cf_selection_mode,
        )

    optimizer = torch.optim.AdamW(
        student.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        warmup_steps = int(num_training_steps * args.warmup_ratio)
        print(f"Warmup ratio {args.warmup_ratio} -> {warmup_steps} steps (of {num_training_steps} total)")
    scheduler = get_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        num_training_steps,
        warmup_steps,
        len(train_loader),
        eta_min=args.eta_min,
    )
    scaler = GradScaler(enabled=args.use_amp)

    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    print("\nStarting training...")
    run_started_at = time.time()
    telemetry = {
        "schema_version": 1,
        "seed": args.seed,
        "teacher_views": args.num_views,
        "student_indices": student_indices,
        "epochs": [],
        "notes": [],
    }
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        train_loader.dataset.current_epoch = epoch
        global_step, epoch_telemetry = train_epoch(
            teacher,
            student,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
            writer,
            cf_criterion=cf_criterion,
            cf_distance_criterion=cf_distance_criterion,
        )
        telemetry["epochs"].append(epoch_telemetry)

        save_checkpoint(
            student,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            0.0,
            args.output_dir,
            f"epoch_{epoch}.pt",
        )

    telemetry["total_wall_clock_seconds"] = time.time() - run_started_at
    telemetry["training_peak_bytes"] = max(
        (epoch["training_peak_bytes"] for epoch in telemetry["epochs"]), default=0
    )
    telemetry["teacher_forward_peak_bytes"] = max(
        (epoch["teacher_forward_peak_bytes"] for epoch in telemetry["epochs"]), default=0
    )
    with open(os.path.join(args.output_dir, "telemetry.json"), "w", encoding="utf-8") as handle:
        json.dump(telemetry, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
