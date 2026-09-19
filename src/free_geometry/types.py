"""Explicit frame-aligned adapter output and immutable teacher cache helpers."""

from dataclasses import dataclass, replace

import torch


@dataclass
class ModelOutput:
    frame_ids: tuple
    readouts: dict  # each [1,S,P,C], in the model's native normalized readout space
    depth: torch.Tensor  # [S,H,W]
    conf: torch.Tensor
    ext_w2c: torch.Tensor  # [S,3,4] or [S,4,4]
    centers: torch.Tensor  # [S,3]
    valid: torch.Tensor  # [S,H,W], real image support (no padding)
    patch_hw: tuple

    def check(self):
        s = len(self.frame_ids)
        if (
            len(set(self.frame_ids)) != s
            or self.depth.ndim != 3
            or self.depth.shape[0] != s
        ):
            raise ValueError("invalid frame IDs or depth shape")
        if self.conf.shape != self.depth.shape or self.valid.shape != self.depth.shape:
            raise ValueError("depth/conf/support shape mismatch")
        if self.ext_w2c.shape not in ((s, 3, 4), (s, 4, 4)) or self.centers.shape != (
            s,
            3,
        ):
            raise ValueError("invalid geometry shape")
        if not self.readouts:
            raise ValueError("adapter has no feature readouts")
        for name, value in self.readouts.items():
            if value.ndim != 4 or value.shape[:3] != (
                1,
                s,
                self.patch_hw[0] * self.patch_hw[1],
            ):
                raise ValueError(f"invalid patch grid for {name}")
        return self

    def to(self, device, detach=False):
        def move(x):
            return (x.detach() if detach else x).to(device)

        return replace(
            self,
            readouts={k: move(v) for k, v in self.readouts.items()},
            **{
                k: move(getattr(self, k))
                for k in ("depth", "conf", "ext_w2c", "centers", "valid")
            },
        )


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    return value
