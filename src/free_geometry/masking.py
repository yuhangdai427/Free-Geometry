"""Normalized-zero block masking on [0,1] input images.

All three new models (VGGT-Omega / Pi3 / DVLT) normalize inputs INSIDE the
model with ImageNet stats, so masking must fill the ImageNet mean color in
[0,1] space (normalized-zero), not black — matching what the DA3 pipeline does
by zeroing already-normalized pixels.
"""
from typing import Tuple

import torch

IMAGENET_MEAN_01 = (0.485, 0.456, 0.406)


def mask_image_blocks(images: torch.Tensor, ratio: float, patch_hw: Tuple[int, int],
                      gen: torch.Generator, fill=IMAGENET_MEAN_01):
    """Zero-fill random patch-aligned blocks with the ImageNet mean color.

    images: [B,S,3,H,W] in [0,1]; patch_hw: (ph, pw) = model patch grid.
    Returns (masked_images, patch_mask [1,S,P], 1=masked). Independent Bernoulli
    per patch (expected ratio), same convention as the DA3/VGGT pipelines.
    """
    B, S, C, H, W = images.shape
    ph, pw = patch_hw
    block = torch.rand(S, ph, pw, generator=gen, device=images.device) < ratio
    m = block.repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)  # [S,H,W]
    fill_t = torch.tensor(fill, device=images.device, dtype=images.dtype).view(1, 1, C, 1, 1)
    out = torch.where(m[None, :, None], fill_t.expand(B, S, C, H, W), images)
    return out, block.reshape(1, S, ph * pw).float()
