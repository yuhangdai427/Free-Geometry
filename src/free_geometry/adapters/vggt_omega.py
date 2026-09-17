"""VGGT-Omega adapter (LoRA r32 on aggregator frame/inter-frame blocks).

Verified against /root/autodl-tmp/vggt-omega source (2026-09-17):
- cached layers [4,11,17,23], each 2048-d = concat(frame_out, inter_frame_out)
- patch_token_start = 17 (1 camera + 16 registers); first frame uses slot-0
  special tokens (satisfied by "student frames ascending, shared[0] first")
- dense_head.norm = shared entry LayerNorm over patch tokens -> readout
- camera head consumes N-frame camera/register tokens jointly -> teacher MUST
  run the head on the full N-frame forward, then slice shared views
- LinearKMaskedBias is an nn.Linear subclass; PEFT 0.18 wraps it calling the
  base forward, preserving the (checkpoint-all-zero) bias_mask buffers.
"""
import sys
from typing import Dict, List

import torch

OMEGA_ROOT = "/root/autodl-tmp/vggt-omega"
OMEGA_CKPT = "/root/autodl-tmp/models/vggt-omega/vggt_omega_1b_416_reproduce.pt"
OMEGA_RES = 416
OMEGA_PATCH = 16
TAP_LAYERS = [4, 11, 17, 23]

sys.path.insert(0, OMEGA_ROOT)

from .base import BaseAdapter  # noqa: E402


def _lora_targets() -> List[str]:
    targets = []
    for i in range(24):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            targets.append(f"aggregator.frame_blocks.{i}.{sub}")
            targets.append(f"aggregator.inter_frame_blocks.{i}.{sub}")
    return targets


class VGGTOmegaAdapter(BaseAdapter):
    model_key = "omega"
    uses_lora = True

    def __init__(self, lora_rank: int = 32, lora_alpha: float = 32.0):
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self._ph = self._pw = None

    # ---------------- model ----------------
    def _fresh_model(self):
        from vggt_omega.models import VGGTOmega
        m = VGGTOmega()
        m.load_state_dict(torch.load(OMEGA_CKPT, map_location="cpu"))
        m.eval()
        return m

    def load(self, device="cuda"):
        self.teacher = self._fresh_model().to(device)
        for p in self.teacher.parameters():
            p.requires_grad = False

    def reset_student(self, device="cuda"):
        import math

        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        base = self._fresh_model()
        cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                         lora_dropout=0.0, target_modules=_lora_targets(),
                         bias="none")
        self.net = get_peft_model(base, cfg)
        self.net.eval()  # NO train-time randomness exists; keep eval determinism
        self.model = self.net.base_model.model  # underlying VGGTOmega
        self.model.to(device)
        # PEFT-faithful re-init: A kaiming, B zeros -> student == frozen baseline
        count = 0
        for m in self.model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                for a in m.lora_A.values():
                    nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                    count += 1
                for b in m.lora_B.values():
                    nn.init.zeros_(b.weight)
        assert count == 192, f"expected 192 LoRA modules, got {count}"
        return self.net

    def student_params(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def student_label(self):
        return f"lora_r{self.lora_rank}_48blk"

    # ---------------- forward ----------------
    def _full_forward(self, model, images):
        """Mirror VGGTOmega.forward: bf16-autocast aggregator, fp32 heads.
        Returns (tokens_list, psi, pose_enc (B,S,9), depth (B,S,H,W), conf)."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=images.is_cuda):
            tokens_list, psi = model.aggregator(images)
        with torch.autocast(device_type="cuda", enabled=False):
            pose_enc = model.camera_head(tokens_list,
                                         patch_token_start=psi)  # joint over ALL frames
            depth, conf = model.dense_head(tokens_list, images=images,
                                           patch_token_start=psi)
        self._ph, self._pw = images.shape[-2] // OMEGA_PATCH, images.shape[-1] // OMEGA_PATCH
        return tokens_list, psi, pose_enc, depth.squeeze(-1), conf

    def _readouts(self, model, tokens_list):
        """dense_head.norm readout on patch tokens for the 4 cached layers."""
        out = {}
        for layer in TAP_LAYERS:
            t = tokens_list[layer]                       # [B,S,T,2048]
            patch = t[:, :, model.aggregator.patch_token_start:, :]
            B, S, P, C = patch.shape
            out[f"tap{layer}"] = model.dense_head.norm(
                patch.reshape(B * S, P, C).float()).reshape(B, S, P, C)
        return out

    def _ext_w2c(self, pose_enc, H, W):
        """[S,3,4] w2c extrinsics, differentiable (frozen head, grads flow)."""
        from vggt_omega.utils.pose_enc import encoding_to_camera
        ext, _ = encoding_to_camera(pose_enc.float(), (H, W))
        return ext[0]

    def _centers(self, model, pose_enc, H, W):
        ext = self._ext_w2c(pose_enc, H, W)
        R, t = ext[..., :3, :3], ext[..., :3, 3]
        return (-R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)  # [S,3]

    def forward_teacher(self, images_N, shared_slots):
        model = self.teacher
        with torch.no_grad():
            tokens_list, psi, pose_enc, depth, conf = self._full_forward(model, images_N)
            readouts = {k: v[:, shared_slots] for k, v in self._readouts(model, tokens_list).items()}
            out = {
                "readouts": {k: v.detach() for k, v in readouts.items()},
                "depth": depth[0, shared_slots].detach(),
                "conf": conf[0, shared_slots].detach(),
                "centers": self._centers(model, pose_enc, images_N.shape[-2], images_N.shape[-1])[shared_slots].detach(),
                "ext_w2c": self._ext_w2c(pose_enc, images_N.shape[-2], images_N.shape[-1])[shared_slots].detach(),
            }
        return out

    def forward_student(self, images_M):
        model = self.model
        tokens_list, psi, pose_enc, depth, conf = self._full_forward(model, images_M)
        return {
            "readouts": self._readouts(model, tokens_list),
            "depth": depth[0],
            "conf": conf[0],
            "centers": self._centers(model, pose_enc, images_M.shape[-2], images_M.shape[-1]),
            "ext_w2c": self._ext_w2c(pose_enc, images_M.shape[-2], images_M.shape[-1]),
            "pose_enc": pose_enc.detach(),  # eval reuse only
        }

    def export_eval(self, images_eval):
        from vggt_omega.utils.pose_enc import encoding_to_camera
        model = getattr(self, "model", None) or self.teacher
        was_lora = hasattr(self, "net")
        with torch.no_grad():
            tokens_list, psi, pose_enc, depth, conf = self._full_forward(model, images_eval)
            ext, intr = encoding_to_camera(pose_enc.float(),
                                           (images_eval.shape[-2], images_eval.shape[-1]))
        ext4 = torch.eye(4, device=ext.device).unsqueeze(0).repeat(ext.shape[0], ext.shape[1], 1, 1)
        ext4[:, :, :3, :4] = ext
        return {"depth": depth[0].float().cpu(), "conf": conf[0].float().cpu(),
                "extr_w2c": ext4[0].float().cpu(), "intr": intr[0].float().cpu()}

    @property
    def patch_hw(self):
        return (self._ph, self._pw)
