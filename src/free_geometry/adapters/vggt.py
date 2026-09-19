"""VGGT: all aggregator blocks, native frozen DPT and camera heads."""

import torch

from ..geometry import centers_from_w2c
from .base import BaseAdapter


class VGGTAdapter(BaseAdapter):
    model_key = "vggt"
    taps = (4, 11, 17, 23)

    def _fresh_model(self):
        from vggt.models.vggt import VGGT

        return VGGT.from_pretrained(self.weight_path).eval()

    def load(self, device="cuda"):
        self.teacher = self._fresh_model().to(device)
        self.teacher.requires_grad_(False)

    def reset_student(self, device="cuda"):
        from peft import LoraConfig, get_peft_model

        targets = [
            f"aggregator.{kind}.{i}.{module}"
            for kind in ("frame_blocks", "global_blocks")
            for i in range(24)
            for module in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
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

    def _forward(self, model, images):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=images.is_cuda and getattr(self, "amp_enabled", True),
        ):
            features, psi = model.aggregator(images)
        self._patch_hw = (images.shape[-2] // 14, images.shape[-1] // 14)
        with torch.autocast("cuda", enabled=False):
            features = [x.float() if x is not None else None for x in features]
            pose = model.camera_head(features)[-1]
            depth, conf = model.depth_head(features, images=images, patch_start_idx=psi)
            ext, intr = pose_encoding_to_extri_intri(pose.float(), images.shape[-2:])
            readouts = {
                str(i): model.depth_head.norm(features[i][:, :, psi:])
                for i in self.taps
            }
        return {
            "readouts": readouts,
            "depth": depth[0, ..., 0],
            "conf": conf[0],
            "ext_w2c": ext[0],
            "centers": centers_from_w2c(ext[0]),
            "intr": intr[0],
        }

    def forward_teacher(self, images, shared_slots):
        out = self._forward(self.teacher, images)
        return {
            k: (
                {n: x[:, shared_slots] for n, x in v.items()}
                if k == "readouts"
                else v[shared_slots]
            )
            for k, v in out.items()
        }

    def forward_student(self, images):
        return self._forward(self.model, images)

    def export_eval(self, images):
        out = self._forward(getattr(self, "model", self.teacher), images)
        return {
            k: out[v].detach().cpu()
            for k, v in [
                ("depth", "depth"),
                ("conf", "conf"),
                ("extr_w2c", "ext_w2c"),
                ("intr", "intr"),
            ]
        }

    @property
    def patch_hw(self):
        return self._patch_hw
