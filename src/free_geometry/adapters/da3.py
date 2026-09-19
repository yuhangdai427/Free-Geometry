"""DA3 giant: native DPT taps, differentiable camera decoder, MV-block LoRA."""

import torch

from ..geometry import centers_from_w2c
from .base import BaseAdapter


class DA3Adapter(BaseAdapter):
    model_key = "da3"
    mask_fill = (0.0, 0.0, 0.0)
    taps = (19, 27, 33, 39)

    def load(self, device="cuda"):
        from depth_anything_3.api import DepthAnything3

        self.teacher = (
            DepthAnything3.from_pretrained(self.weight_path).to(device).eval()
        )
        self.teacher.requires_grad_(False)

    def reset_student(self, device="cuda"):
        from depth_anything_3.test_time_adaption.models import StudentModel

        self.net = (
            StudentModel(
                model_name=self.weight_path,
                output_layers=list(self.taps),
                embed_dim=1536,
                lora_rank=self.config.rank,
                lora_alpha=self.config.alpha,
                lora_dropout=self.config.dropout,
                train_camera_token=True,
                lora_layers=list(range(13, 40)),
                ref_view_strategy="first",
                patch_swiglu_mlp_for_lora=True,
            )
            .to(device)
            .eval()
        )
        self.model = self.net.da3

    def prepare_images(self, paths):
        import numpy as np
        from PIL import Image

        from depth_anything_3.utils.io.input_processor import InputProcessor

        images, _, transforms = InputProcessor()(
            list(paths),
            None,
            np.broadcast_to(np.eye(3), (len(paths), 3, 3)).copy(),
            self.resolution,
            "upper_bound_resize",
            sequential=True,
        )
        valid = torch.ones(len(paths), *images.shape[-2:], dtype=torch.bool)
        metadata = []
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                w, h = image.size
            metadata.append(
                {
                    "path": path,
                    "processor": "DA3 upper_bound_resize; scene-fixed common center crop",
                    "original_hw": [h, w],
                    "output_hw": list(images.shape[-2:]),
                    "image_transform": transforms[i].tolist(),
                }
            )
        return images, valid, metadata

    def _forward(self, model, images):
        vit = model.model.backbone.pretrained
        if hasattr(vit, "base_model"):
            vit = vit.base_model.model
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=images.is_cuda and getattr(self, "amp_enabled", True),
        ):
            feats, _ = vit.get_intermediate_layers(
                images,
                list(self.taps),
                export_feat_layers=[],
                cam_token=None,
                ref_view_strategy="first",
            )
        h, w = images.shape[-2:]
        self._patch_hw = (h // 14, w // 14)
        with torch.autocast("cuda", enabled=False):
            feats = [(f.float(), c.float()) for f, c in feats]
            prediction = model.model.forward_head_only(
                feats, H=h, W=w, process_camera=True, process_sky=False
            )
            readouts = {
                str(layer): model.model.head.norm(feats[i][0])
                for i, layer in enumerate(self.taps)
            }
        ext = prediction["extrinsics"][0].float()
        return {
            "readouts": readouts,
            "depth": prediction["depth"][0].float(),
            "conf": prediction["depth_conf"][0].float(),
            "ext_w2c": ext,
            "centers": centers_from_w2c(ext),
            "intr": prediction["intrinsics"][0],
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
