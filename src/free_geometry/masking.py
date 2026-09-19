"""Patch-aligned input corruption, independent of the supervised patch region."""

import torch


def mask_images(images, patch_hw, ratio, seed, fill):
    b, s, c, h, w = images.shape
    ph, pw = patch_hw
    if b != 1 or h % ph or w % pw or not 0 <= ratio <= 1:
        raise ValueError("invalid image mask dimensions/ratio")
    # CPU generator makes masks identical across devices and preserves global RNG.
    mask = torch.rand(s, ph, pw, generator=torch.Generator().manual_seed(seed)) < ratio
    pixel = (
        mask.repeat_interleave(h // ph, 1)
        .repeat_interleave(w // pw, 2)
        .to(images.device)
    )
    color = images.new_tensor(fill).view(1, 1, c, 1, 1)
    return torch.where(pixel[None, :, None], color, images), mask
