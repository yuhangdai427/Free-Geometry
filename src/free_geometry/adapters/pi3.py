"""Pi3 (original pi^3, large) adapter — LoRA r32 on the 36-block trunk decoder.

Verified against /root/autodl-tmp/pi3/pi3/models/pi3.py (2026-09-17):
- trunk: even blocks frame-attn, odd blocks global; taps h34||h35 -> (B*N, 5+P, 2048)
- 5 register tokens, NO camera token; camera branch reads PATCH tokens
- readout (plan section 4.3): point_decoder.blocks[0].norm1(point_decoder.projects(h))
  — recomputed exactly (same frozen modules, one extra Linear pass), registers stripped
- outputs: local_points (depth = [...,2]), conf logits (sigmoid), camera_poses c2w
  (center = translation column), NO intrinsics -> eval uses GT-K (labeled)
- forward mirrors Pi3.forward: bf16 autocast trunk, fp32 heads
"""
import sys
from typing import Dict, List

import torch

PI3_ROOT = "/root/autodl-tmp/pi3"
PI3_WEIGHTS = "/root/autodl-tmp/models/pi3"
PI3_RES = 504
PI3_PATCH = 14

sys.path.insert(0, PI3_ROOT)

from .base import BaseAdapter  # noqa: E402


def _lora_targets() -> List[str]:
    targets = []
    for i in range(36):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            targets.append(f"decoder.{i}.{sub}")
    return targets


class Pi3Adapter(BaseAdapter):
    model_key = "pi3"
    uses_lora = True

    def __init__(self, lora_rank: int = 32, lora_alpha: float = 32.0):
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self._ph = self._pw = None

    def _fresh_model(self):
        from pi3.models.pi3 import Pi3
        m = Pi3(pos_type="rope100", decoder_size="large")
        from safetensors.torch import load_file
        m.load_state_dict(load_file(f"{PI3_WEIGHTS}/model.safetensors"))
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
        self.net.eval()
        self.model = self.net.base_model.model
        self.model.to(device)
        count = 0
        for m in self.model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                for a in m.lora_A.values():
                    nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                    count += 1
                for b in m.lora_B.values():
                    nn.init.zeros_(b.weight)
        assert count == 144, f"expected 144 LoRA modules, got {count}"
        return self.net

    def student_params(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def student_label(self):
        return f"lora_r{self.lora_rank}_36blk"

    # ---------------- forward (mirrors Pi3.forward) ----------------
    def _full_forward(self, model, imgs, want_readouts):
        B, N, _, H, W = imgs.shape
        self._ph, self._pw = H // PI3_PATCH, W // PI3_PATCH
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=imgs.is_cuda):
            x = (imgs - model.image_mean) / model.image_std
            x = x.reshape(B * N, *x.shape[2:])
            hidden = model.encoder(x, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            hidden, pos = model.decode(hidden, N, H, W)  # (B*N, 5+P, 2048)

            point_hidden = model.point_decoder(hidden, xpos=pos)
            conf_hidden = model.conf_decoder(hidden, xpos=pos)
            camera_hidden = model.camera_decoder(hidden, xpos=pos)

            readouts = {}
            if want_readouts:
                # exact recompute of projects -> blocks[0].norm1 (frozen modules)
                proj = model.point_decoder.projects(hidden)
                phi = model.point_decoder.blocks[0].norm1(proj)
                phi_patch = phi[:, model.patch_start_idx:]      # (B*N, P, C)
                readouts["point_proj_ln"] = phi_patch.reshape(
                    B, N, phi_patch.shape[1], -1).float()

        with torch.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = model.point_head([point_hidden[:, model.patch_start_idx:]], (H, W)) \
                .reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = model.conf_head([conf_hidden[:, model.patch_start_idx:]], (H, W)) \
                .reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            camera_poses = model.camera_head(
                camera_hidden[:, model.patch_start_idx:], self._ph, self._pw) \
                .reshape(B, N, 4, 4)

        depth = local_points[..., 2]                     # [B,N,H,W]
        conf_sig = torch.sigmoid(conf[..., 0])           # [B,N,H,W]
        centers = camera_poses[:, :, :3, 3]              # c2w translation
        return {"readouts": readouts, "depth": depth, "conf": conf_sig,
                "centers": centers, "camera_poses": camera_poses,
                "local_points": local_points}

    @staticmethod
    def _estimate_k(local_points: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Estimate per-view K at MODEL resolution from Pi3's own local pointmap.
        Pinhole, zero skew: u = fx*(X/Z) + cx, v = fy*(Y/Z) + cy — linear LSQ
        per view on high-confidence pixels, with one residual-trimming pass.
        Scale-invariant (X/Z cancels any global scale)."""
        B, N, H, W, _ = local_points.shape
        dev = local_points.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32),
                                indexing="ij")
        u = xs.reshape(-1) + 0.5
        v = ys.reshape(-1) + 0.5
        K = torch.eye(3).repeat(B * N, 1, 1)
        for idx in range(B * N):
            lp = local_points.reshape(B * N, H * W, 3)[idx]
            cf = conf.reshape(B * N, H * W)[idx]
            z = lp[:, 2]
            ok = (z > 1e-6) & (cf > torch.quantile(cf, 0.5))
            xz, yz = lp[:, 0] / z, lp[:, 1] / z
            for _ in range(2):  # robust: refit on inliers
                A1 = torch.stack([xz[ok], torch.ones_like(xz[ok])], -1)
                fx, cx = torch.linalg.lstsq(A1, u[ok]).solution
                A2 = torch.stack([yz[ok], torch.ones_like(yz[ok])], -1)
                fy, cy = torch.linalg.lstsq(A2, v[ok]).solution
                r = ((u - (fx * xz + cx)).abs() + (v - (fy * yz + cy)).abs())
                ok = ok & (r < max(4.0, float(r[ok].median()) * 4))
            K[idx, 0, 0], K[idx, 0, 2] = fx, cx
            K[idx, 1, 1], K[idx, 1, 2] = fy, cy
        return K.reshape(B, N, 3, 3)

    def forward_teacher(self, images_N, shared_slots):
        with torch.no_grad():
            o = self._full_forward(self.teacher, images_N, want_readouts=True)
        return {
            "readouts": {k: v[:, shared_slots].detach() for k, v in o["readouts"].items()},
            "depth": o["depth"][0, shared_slots].detach(),
            "conf": o["conf"][0, shared_slots].detach(),
            "centers": o["centers"][0, shared_slots].detach(),
        }

    def forward_student(self, images_M):
        o = self._full_forward(self.model, images_M, want_readouts=True)
        return {"readouts": o["readouts"], "depth": o["depth"][0], "conf": o["conf"][0],
                "centers": o["centers"][0], "camera_poses": o["camera_poses"].detach()}

    def export_eval(self, images_eval):
        with torch.no_grad():
            o = self._full_forward(getattr(self, "model", None) or self.teacher,
                                   images_eval, want_readouts=False)
        c2w = o["camera_poses"][0]                        # [N,4,4]
        w2c = torch.inverse(c2w.float()).cpu()
        # K from Pi3's OWN pointmap at model resolution (2026-09-17 fix: the
        # previous GT-K was stored at native resolution and got re-scaled by
        # _prep_unposed -> 12x focal error on eth3d -> F1 collapse)
        k = self._estimate_k(o["local_points"].float(), o["conf"].float())[0].cpu()
        return {"depth": o["depth"][0].float().cpu(), "conf": o["conf"][0].float().cpu(),
                "extr_w2c": w2c, "intr": k}

    @property
    def patch_hw(self):
        return (self._ph, self._pw)
