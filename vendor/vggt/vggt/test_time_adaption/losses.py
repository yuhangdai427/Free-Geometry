"""SelfEvo and output-distillation losses for VGGT Free-Geometry TTA."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import VGGTFreeGeometryOutput


class VGGTSelfEvoRepresentationLoss(nn.Module):
    """Four-layer per-frame patch-token representation MSE used by SelfEvo."""

    def __init__(
        self,
        student_frame_indices: List[int] | None = None,
        target_layers: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.target_layers = target_layers or [4, 11, 17, 23]

    def forward(
        self, teacher_output: VGGTFreeGeometryOutput, student_output: VGGTFreeGeometryOutput
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = []
        for layer in self.target_layers:
            # Exclude camera/register tokens: SelfEvo pools only patch tokens.
            teacher = teacher_output.global_features[layer][:, self.student_frame_indices, teacher_output.patch_start_idx:].detach().float().mean(dim=2)
            student = student_output.global_features[layer][:, :, student_output.patch_start_idx:].float().mean(dim=2)
            losses.append(F.mse_loss(student, teacher))
        loss = torch.stack(losses).mean()
        return loss, {"self_evo_mse": loss}


class VGGTOutputDistillLoss(nn.Module):
    """VGGT's official confidence-regression objective with teacher pseudo-labels.

    This preserves the official point/depth form ``error * confidence - 0.2
    log(confidence)`` while replacing GT targets by frozen 8-view predictions.
    """

    def __init__(self, point_weight: float = 1.0, depth_weight: float = 1.0, alpha: float = 0.2) -> None:
        super().__init__()
        self.point_weight = point_weight
        self.depth_weight = depth_weight
        self.alpha = alpha

    def _confidence_regression(
        self, prediction: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        error = torch.linalg.vector_norm(prediction.float() - target.detach().float(), dim=-1)
        confidence = confidence.float().clamp_min(1e-6)
        return (error * confidence - self.alpha * torch.log(confidence)).mean()

    def forward(
        self, student: Dict[str, torch.Tensor], teacher: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        required = ("world_points", "world_points_conf", "depth", "depth_conf")
        missing = [key for key in required if key not in student or key not in teacher]
        if missing:
            raise KeyError(f"VGGT output loss requires {required}; missing {missing}")

        point_loss = self._confidence_regression(
            student["world_points"], teacher["world_points"], student["world_points_conf"]
        )
        depth_loss = self._confidence_regression(
            student["depth"], teacher["depth"], student["depth_conf"]
        )
        total = self.point_weight * point_loss + self.depth_weight * depth_loss
        return total, {
            "vggt_output_point": point_loss,
            "vggt_output_depth": depth_loss,
            "vggt_output_total": total,
        }
