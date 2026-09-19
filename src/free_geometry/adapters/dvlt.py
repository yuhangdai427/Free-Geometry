"""DVLT: shared recurrent LoRA, differentiable ray pose for all five losses."""

import torch

from .base import BaseAdapter

DVLT_PATCH = 14


class DVLTAdapter(BaseAdapter):
    model_key = "dvlt"

    def _fresh_model(self):
        from dvlt.model.dvlt.model import DVLTModel

        return DVLTModel.from_pretrained(self.weight_path).eval()

    def load(self, device="cuda"):
        self.teacher = self._fresh_model().to(device)
        self.teacher.requires_grad_(False)

    def reset_student(self, device="cuda"):
        from peft import LoraConfig, get_peft_model

        targets = [
            f"recurrent_blocks.0.{a}.{m}"
            for a in ("frame_attn", "global_attn")
            for m in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
        ]
        self.net = (
            get_peft_model(
                self._fresh_model(),
                LoraConfig(
                    r=self.config.rank,
                    lora_alpha=self.config.alpha,
                    lora_dropout=self.config.dropout,
                    target_modules=targets,
                    bias="none",
                ),
            )
            .to(device)
            .eval()
        )
        self.model = self.net.base_model.model

    def _forward_adapt(self, model, images, want_readouts):
        """Mirror forward_inference (fixed-K solve, deterministic) WITH grad.
        Model must be in eval() mode (guaranteed by callers)."""
        from dvlt.model.dvlt.model import _slice_expand_flatten, activate_head

        B, S, _, H, W = images.shape
        self._ph, self._pw = H // DVLT_PATCH, W // DVLT_PATCH
        z0 = model._encode_images(images)
        register_token = model.register_token.expand(B, S, -1, -1).reshape(
            B * S, model.num_register_tokens, -1
        )
        camera_token = _slice_expand_flatten(model.camera_token, B, S)
        x = torch.cat([camera_token, register_token, z0], dim=1)
        rope_pos = model._get_rope_positions(B * S, H, W, images.device)
        features = model._solve_inference(x, rope_pos, B, S)

        BS = B * S
        chunk = model.decode_chunk_size or BS
        psi = model.patch_start_idx
        ray_outs, depth_outs, ray_feats, depth_feats = [], [], [], []
        for start in range(0, BS, chunk):
            f = features[start : start + chunk]
            p = rope_pos[start : start + chunk]
            kw = {"H": H, "W": W, "patch_start_idx": psi, "pos": p}
            ro, rf = model.ray_decoder(f, **kw)
            do, df = model.depth_decoder(f, **kw)
            ray_outs.append(ro)
            depth_outs.append(do)
            ray_feats.append(rf)
            depth_feats.append(df)
        ray_out = torch.cat(ray_outs, 0)
        depth_out = torch.cat(depth_outs, 0)

        pred_rays, _ = activate_head(
            ray_out, activation="identity", conf_activation=None
        )
        pred_depth, depth_conf = activate_head(
            depth_out, activation="exp_clamped", conf_activation="exp_plus_one"
        )

        readouts = {}
        if want_readouts:
            rn = model.ray_decoder.norm(torch.cat(ray_feats, 0))[:, psi:]  # (BS,P,384)
            dn = model.depth_decoder.norm(torch.cat(depth_feats, 0))[:, psi:]
            readouts["ray_ln"] = rn.reshape(B, S, *rn.shape[1:]).float()
            readouts["depth_ln"] = dn.reshape(B, S, *dn.shape[1:]).float()

        rays = pred_rays.view(B, S, H, W, 6)
        depth = pred_depth.view(B, S, H, W, 1)[..., 0]
        conf = depth_conf.view(B, S, H, W)
        from ..geometry import centers_from_w2c
        from ..ray_pose import rays_to_pose_differentiable

        ext, intr = rays_to_pose_differentiable(rays, H, W, DVLT_PATCH)
        centers = centers_from_w2c(ext)
        return {
            "readouts": readouts,
            "depth": depth,
            "conf": conf,
            "centers": centers,
            "rays": rays,
            "ext_w2c": ext,
            "intr": intr,
        }

    def forward_teacher(self, images_N, shared_slots):
        with torch.no_grad():
            o = self._forward_adapt(self.teacher, images_N, want_readouts=True)
        return {
            "readouts": {
                k: v[:, shared_slots].detach() for k, v in o["readouts"].items()
            },
            "depth": o["depth"][0, shared_slots].detach(),
            "conf": o["conf"][0, shared_slots].detach(),
            "centers": o["centers"][0, shared_slots].detach(),
            "ext_w2c": o["ext_w2c"][0, shared_slots].detach(),
        }

    def forward_student(self, images_M):
        o = self._forward_adapt(self.model, images_M, want_readouts=True)
        return {
            "readouts": o["readouts"],
            "depth": o["depth"][0],
            "conf": o["conf"][0],
            "centers": o["centers"][0],
            "ext_w2c": o["ext_w2c"][0],
        }

    def export_eval(self, images_eval):
        from dvlt.common.rays import rays_to_pose

        model = getattr(self, "model", None) or self.teacher
        with torch.no_grad():
            o = self._forward_adapt(model, images_eval, want_readouts=False)
            rays = o["rays"].float()
            ones = torch.ones_like(o["depth"])
            c2w, intr = rays_to_pose(
                rays,
                ones,
                images_eval.shape[-2],
                images_eval.shape[-1],
                patch_size=DVLT_PATCH,
            )
        w2c = torch.inverse(c2w[0].float()).cpu()
        from ..geometry import rotation_angle

        angle = rotation_angle(o["ext_w2c"][0, :, :3, :3].cpu(), w2c[:, :3, :3])
        self.pose_diagnostics = {
            "fit_vs_official_rotation_deg_mean": float(angle.mean() * 180 / torch.pi),
            "fit_vs_official_rotation_deg_max": float(angle.max() * 180 / torch.pi),
            "training_solver": "homogeneous_dlt_irls_qr",
            "evaluation_solver": "official_ransac",
        }
        return {
            "depth": o["depth"][0].float().cpu(),
            "conf": o["conf"][0].float().cpu(),
            "extr_w2c": w2c,
            "intr": intr[0].float().cpu(),
        }

    @property
    def patch_hw(self):
        return (self._ph, self._pw)
