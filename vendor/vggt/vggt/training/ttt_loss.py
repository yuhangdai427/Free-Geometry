"""
Test-Time Training (TTT) Loss for VGGT.

Implements self-supervised consistency loss inspired by Test3R.
The key idea: predictions for the same anchor image should be consistent
regardless of which companion image it's paired with.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def ttt_consistency_loss(
    pred1: Dict[str, torch.Tensor],
    pred2: Dict[str, torch.Tensor],
    use_confidence: bool = False,
    depth_weight: float = 1.0,
    point_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute consistency loss between predictions of anchor image from two pairs.

    This is a self-supervised loss that doesn't require ground truth.
    The anchor image should produce consistent predictions regardless of
    which companion image it's paired with.

    Args:
        pred1: Predictions from pair (anchor, img_j)
            - 'world_points': [B, S, H, W, 3] 3D world coordinates (optional)
            - 'world_points_conf': [B, S, H, W] confidence (optional)
            - 'depth': [B, S, H, W, 1] depth values (optional)
            - 'depth_conf': [B, S, H, W] depth confidence (optional)
        pred2: Predictions from pair (anchor, img_k)
        use_confidence: Whether to weight loss by prediction confidence
        depth_weight: Weight for depth consistency loss
        point_weight: Weight for 3D point consistency loss

    Returns:
        Scalar consistency loss
    """
    total_loss = 0.0

    # 3D point consistency (if available)
    if 'world_points' in pred1 and 'world_points' in pred2:
        pts1 = pred1['world_points']
        pts2 = pred2['world_points']

        if use_confidence and 'world_points_conf' in pred1 and 'world_points_conf' in pred2:
            # Confidence-weighted consistency
            conf1 = pred1['world_points_conf'].unsqueeze(-1)  # [B, S, H, W, 1]
            conf2 = pred2['world_points_conf'].unsqueeze(-1)

            # Normalize confidence to [0, 1] range
            conf1 = torch.sigmoid(conf1)
            conf2 = torch.sigmoid(conf2)

            # Weight by geometric mean of confidences
            conf_weight = torch.sqrt(conf1 * conf2)
            loss_pts3d = (torch.abs(pts1 - pts2) * conf_weight).mean()
        else:
            # Simple L1 consistency (Test3R style)
            loss_pts3d = torch.abs(pts1 - pts2).mean()

        total_loss = total_loss + point_weight * loss_pts3d

    # Depth consistency (if available)
    if 'depth' in pred1 and 'depth' in pred2:
        depth1 = pred1['depth']
        depth2 = pred2['depth']

        if use_confidence and 'depth_conf' in pred1 and 'depth_conf' in pred2:
            conf1 = torch.sigmoid(pred1['depth_conf'].unsqueeze(-1))
            conf2 = torch.sigmoid(pred2['depth_conf'].unsqueeze(-1))
            conf_weight = torch.sqrt(conf1 * conf2)
            loss_depth = (torch.abs(depth1 - depth2) * conf_weight).mean()
        else:
            loss_depth = torch.abs(depth1 - depth2).mean()

        total_loss = total_loss + depth_weight * loss_depth

    # If neither depth nor points available, raise error
    if isinstance(total_loss, float) and total_loss == 0.0:
        raise ValueError("No valid predictions found for TTT loss. Need either 'world_points' or 'depth'.")

    return total_loss


def extract_anchor_predictions(
    predictions: Dict[str, torch.Tensor],
    anchor_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract predictions for the anchor image from a batch.

    Args:
        predictions: Full predictions dict with shape [B, S, ...]
        anchor_idx: Index of anchor image in sequence dimension

    Returns:
        Predictions for anchor image only
    """
    anchor_pred = {}

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            # Extract anchor frame (index along sequence dimension)
            anchor_pred[key] = value[:, anchor_idx:anchor_idx+1]
        else:
            anchor_pred[key] = value

    return anchor_pred


def make_triplet_pairs(num_images: int, max_triplets: Optional[int] = None) -> list:
    """
    Create triplet pairs (anchor, img_j, img_k) for TTT.

    Each triplet consists of:
    - anchor: the reference image
    - img_j: first companion image
    - img_k: second companion image (different from img_j)

    Args:
        num_images: Total number of images in sequence
        max_triplets: Maximum number of triplets to generate (None = all)

    Returns:
        List of (anchor_idx, j_idx, k_idx) tuples
    """
    if num_images < 3:
        raise ValueError(f"Need at least 3 images for triplets, got {num_images}")

    triplets = []

    # Generate all possible triplets
    for anchor in range(num_images):
        for j in range(num_images):
            if j == anchor:
                continue
            for k in range(j + 1, num_images):
                if k == anchor:
                    continue
                triplets.append((anchor, j, k))

    # Optionally limit number of triplets
    if max_triplets is not None and len(triplets) > max_triplets:
        import random
        triplets = random.sample(triplets, max_triplets)

    return triplets


class TTTLoss(nn.Module):
    """
    Test-Time Training loss module for VGGT.

    Wraps the consistency loss function in a PyTorch module for easier integration.
    """

    def __init__(
        self,
        use_confidence: bool = False,
        depth_weight: float = 0.1,
    ):
        super().__init__()
        self.use_confidence = use_confidence
        self.depth_weight = depth_weight

    def forward(
        self,
        pred1: Dict[str, torch.Tensor],
        pred2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute TTT consistency loss."""
        return ttt_consistency_loss(
            pred1, pred2,
            use_confidence=self.use_confidence,
            depth_weight=self.depth_weight,
        )
