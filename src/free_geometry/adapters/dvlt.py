"""DVLT (Deja View Looping Transformer) adapter — LoRA r32 on shared recurrent
block (2026-09-17 user decision). Supports two variants:
  - "shared":     single LoRA across all K loop iterations (PEFT, default)
  - "two_stage":  early loops use one LoRA, late loops use another (RRT-inspired,
                   manual implementation — PEFT cannot switch adapters mid-forward)

Key verified facts (2026-09-17, /root/autodl-tmp/dvlt):
- eval-mode forward() routes to @torch.no_grad forward_inference; train mode
  switches to RANDOM-K sampling + drop_path 0.1 -> we NEVER call .train()
- recurrence is weight-shared: recurrent_blocks[0].frame_attn / .global_attn
- DecoderHead returns (out, features) with features PRE-norm; the post-norm
  patch tokens are exactly self.norm(features)[:, patch_start_idx:] (384-d)
- camera centers (uniform conf, matching use_depth_conf_for_pose=False):
  bilinear patch-grid downsample (align_corners=True) of ray ORIGINS, mean
- eval export uses the OFFICIAL rays_to_pose (RANSAC; no_grad is fine there)
"""
import math
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

DVLT_ROOT = "/root/autodl-tmp/dvlt"
DVLT_WEIGHTS = "/root/autodl-tmp/models/dvlt"
DVLT_RES = 504
DVLT_PATCH = 14

sys.path.insert(0, DVLT_ROOT + "/src")

from .base import BaseAdapter  # noqa: E402


class TwoStageLoRA(nn.Module):
    """Wraps a frozen nn.Linear with two LoRA paths (early/late loops).
    current_step is set externally by the adapter's solve loop before each
    loop iteration. Zero-init B -> initial forward == frozen baseline."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, split_step: int):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.split_step = split_step
        self.current_step = 0
        # Early LoRA (loops < split_step)
        self.A_early = nn.Parameter(torch.empty(rank, base.in_features))
        self.B_early = nn.Parameter(torch.zeros(base.out_features, rank))
        # Late LoRA (loops >= split_step)
        self.A_late = nn.Parameter(torch.empty(rank, base.in_features))
        self.B_late = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A_early, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_late, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        scale = self.alpha / self.rank
        if self.current_step < self.split_step:
            delta = F.linear(F.linear(x, self.A_early), self.B_early) * scale
        else:
            delta = F.linear(F.linear(x, self.A_late), self.B_late) * scale
        return out + delta


class DVLTAdapter(BaseAdapter):
    model_key = "dvlt"
    uses_lora = True

    def __init__(self, lora_rank: int = 32, lora_alpha: float = 32.0,
                 lora_variant: str = "shared", split_step: int = 6):
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self.lora_variant = lora_variant  # "shared" | "two_stage"
        self.split_step = split_step      # loop index where late LoRA takes over
        self._ph = self._pw = None
        self._ts_modules: List[TwoStageLoRA] = []

    def _fresh_model(self):
        from dvlt.model.dvlt.model import DVLTModel
        try:
            m = DVLTModel.from_pretrained(DVLT_WEIGHTS)  # reads config.json
        except Exception:
            m = DVLTModel(img_size=504, patch_size=14, embed_dim=768,
                          num_steps=16, min_steps=8, load_patch_embed_weights=False,
                          recurrence_mode="gated", time_conditioning="interval",
                          k_sampling="linspace", inference_steps=12,
                          decoder_head_type="linear", depth_head_type="conv",
                          decode_chunk_size=128, drop_path=0.1)
            from safetensors.torch import load_file
            m.load_state_dict(load_file(f"{DVLT_WEIGHTS}/model.safetensors"))
        m.eval()
        return m

    def load(self, device="cuda"):
        self.teacher = self._fresh_model().to(device)
        for p in self.teacher.parameters():
            p.requires_grad = False

    def reset_student(self, device="cuda"):
        base = self._fresh_model()

        if self.lora_variant == "two_stage":
            # Manual two-stage LoRA (RRT-inspired, no PEFT — PEFT can't switch
            # adapters mid-forward between loop iterations)
            self.model = base.to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._ts_modules = []
            block = self.model.recurrent_blocks[0]
            for attn_name in ("frame_attn", "global_attn"):
                attn = getattr(block, attn_name)
                for sub_path in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    parts = sub_path.split(".")
                    obj = attn
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    old = getattr(obj, parts[-1])
                    ts = TwoStageLoRA(old, self.lora_rank, self.lora_alpha,
                                      self.split_step).to(device)
                    setattr(obj, parts[-1], ts)
                    self._ts_modules.append(ts)
            assert len(self._ts_modules) == 8
            self.net = None  # no PEFT wrapper in this variant
            return self.model

        # else: shared LoRA via PEFT (default)
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        targets = []
        for attn in ("frame_attn", "global_attn"):
            for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                targets.append(f"recurrent_blocks.0.{attn}.{sub}")
        cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                         lora_dropout=0.0, target_modules=targets,
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
        assert count == 8, f"expected 8 LoRA modules, got {count}"
        return self.net

    def student_params(self):
        if self.lora_variant == "two_stage" and self._ts_modules:
            params = []
            for m in self._ts_modules:
                params.extend([m.A_early, m.B_early, m.A_late, m.B_late])
            return params
        if getattr(self, "net", None) is not None:
            return [p for p in self.net.parameters() if p.requires_grad]
        m = getattr(self, "model", None) or getattr(self, "teacher", None)
        return list(m.parameters())

    def student_label(self):
        n = sum(p.numel() for p in self.student_params())
        tag = "ts" if self.lora_variant == "two_stage" else "sh"
        return f"lora_{tag}_r{self.lora_rank}_{n / 1e6:.1f}M"

    # ---------------- forward ----------------
    def _forward_adapt(self, model, images, want_readouts):
        """Mirror forward_inference (fixed-K solve, deterministic) WITH grad.
        Model must be in eval() mode (guaranteed by callers)."""
        from dvlt.model.dvlt.model import _slice_expand_flatten, activate_head
        B, S, _, H, W = images.shape
        self._ph, self._pw = H // DVLT_PATCH, W // DVLT_PATCH
        z0 = model._encode_images(images)
        register_token = model.register_token.expand(B, S, -1, -1) \
            .reshape(B * S, model.num_register_tokens, -1)
        camera_token = _slice_expand_flatten(model.camera_token, B, S)
        x = torch.cat([camera_token, register_token, z0], dim=1)
        rope_pos = model._get_rope_positions(B * S, H, W, images.device)
        # Use staged solve (with step counters) if the model has TwoStageLoRA
        if self._ts_modules and hasattr(model.recurrent_blocks[0].frame_attn.attn.qkv, "A_early"):
            features = self._solve_staged(model, x, rope_pos, B, S)
        else:
            features = model._solve_inference(x, rope_pos, B, S)

        BS = B * S
        chunk = model.decode_chunk_size or BS
        psi = model.patch_start_idx
        ray_outs, depth_outs, ray_feats, depth_feats = [], [], [], []
        for start in range(0, BS, chunk):
            f = features[start:start + chunk]
            p = rope_pos[start:start + chunk]
            kw = dict(H=H, W=W, patch_start_idx=psi, pos=p)
            ro, rf = model.ray_decoder(f, **kw)
            do, df = model.depth_decoder(f, **kw)
            ray_outs.append(ro)
            depth_outs.append(do)
            ray_feats.append(rf)
            depth_feats.append(df)
        ray_out = torch.cat(ray_outs, 0)
        depth_out = torch.cat(depth_outs, 0)

        pred_rays, _ = activate_head(ray_out, activation="identity", conf_activation=None)
        pred_depth, depth_conf = activate_head(
            depth_out, activation="exp_clamped", conf_activation="exp_plus_one")

        readouts = {}
        if want_readouts:
            rn = model.ray_decoder.norm(torch.cat(ray_feats, 0))[:, psi:]     # (BS,P,384)
            dn = model.depth_decoder.norm(torch.cat(depth_feats, 0))[:, psi:]
            readouts["ray_ln"] = rn.reshape(B, S, *rn.shape[1:]).float()
            readouts["depth_ln"] = dn.reshape(B, S, *dn.shape[1:]).float()

        rays = pred_rays.view(B, S, H, W, 6)
        depth = pred_depth.view(B, S, H, W, 1)[..., 0]
        conf = depth_conf.view(B, S, H, W)
        centers = self._ray_centers(rays)  # differentiable, mirrors rays_to_pose
        return {"readouts": readouts, "depth": depth, "conf": conf,
                "centers": centers, "rays": rays}

    def _solve_staged(self, model, x, rope_pos, B, S):
        """Mirror of _solve_inference_linspace_k that sets the loop step counter
        on all TwoStageLoRA modules so early/late LoRA paths activate correctly."""
        K = model.inference_steps
        ts = torch.linspace(0.0, 1.0, K).tolist()
        for i in range(K):
            for m in self._ts_modules:
                m.current_step = i
            t_now = ts[i]
            t_next = ts[i + 1] if i + 1 < K else 1.0
            x = model._interval_step(x, t_now, t_next, rope_pos, B, S)
        return x

    def _ray_centers(self, rays):
        """Uniform-weight camera centers = mean of bilinear-downsampled ray
        origins on the patch grid (align_corners=True, exactly as
        rays_to_pose + _camray_to_caminfo with uniform confidence)."""
        B, S, H, W, _ = rays.shape
        ph, pw = H // DVLT_PATCH, W // DVLT_PATCH
        origins = rays[..., 3:6]  # last 3 = origins (verified in rays.py docstring)
        o = F.interpolate(origins.reshape(B * S, H, W, 3).permute(0, 3, 1, 2),
                          size=(ph, pw), mode="bilinear", align_corners=True)
        o = o.permute(0, 2, 3, 1).reshape(B, S, ph * pw, 3)
        return o.mean(dim=2)  # [B,S,3]

    def forward_teacher(self, images_N, shared_slots):
        with torch.no_grad():
            o = self._forward_adapt(self.teacher, images_N, want_readouts=True)
        return {
            "readouts": {k: v[:, shared_slots].detach() for k, v in o["readouts"].items()},
            "depth": o["depth"][0, shared_slots].detach(),
            "conf": o["conf"][0, shared_slots].detach(),
            "centers": o["centers"][0, shared_slots].detach(),
        }

    def forward_student(self, images_M):
        o = self._forward_adapt(self.model, images_M, want_readouts=True)
        return {"readouts": o["readouts"], "depth": o["depth"][0], "conf": o["conf"][0],
                "centers": o["centers"][0], "rays": o["rays"].detach()}

    def export_eval(self, images_eval):
        from dvlt.common.rays import rays_to_pose
        model = getattr(self, "model", None) or self.teacher
        with torch.no_grad():
            o = self._forward_adapt(model, images_eval, want_readouts=False)
            rays = o["rays"].float()
            ones = torch.ones_like(o["depth"])
            c2w, intr = rays_to_pose(rays, ones, images_eval.shape[-2],
                                     images_eval.shape[-1], patch_size=DVLT_PATCH)
        w2c = torch.inverse(c2w[0].float()).cpu()
        return {"depth": o["depth"][0].float().cpu(), "conf": o["conf"][0].float().cpu(),
                "extr_w2c": w2c, "intr": intr[0].float().cpu()}

    @property
    def patch_hw(self):
        return (self._ph, self._pw)
