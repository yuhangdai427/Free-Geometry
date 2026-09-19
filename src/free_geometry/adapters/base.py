"""The only boundary allowed to know model internals."""

from pathlib import Path

import torch

from ..types import ModelOutput


class BaseAdapter:
    model_key = "base"
    uses_lora = True
    patch_size = 14
    default_resolution = 504
    mask_fill = (0.485, 0.456, 0.406)

    def configure(self, config):
        self.config = config
        self.weight_path = config.weights
        self.lora_rank, self.lora_alpha = config.rank, config.alpha
        self.resolution = config.resolution or self.default_resolution
        if not config.weights:
            raise ValueError(
                "model.weights is required (local checkpoint or supported Hub ID)"
            )
        if config.source:
            import sys

            source = str(Path(config.source).resolve())
            if not Path(source).is_dir():
                raise FileNotFoundError(source)
            sys.path.insert(0, source)
            if (Path(source) / "src").is_dir():
                sys.path.insert(0, str(Path(source) / "src"))
        return self

    def resolve_weights(self):
        """Resolve Hub IDs once so Teacher and Student load identical immutable files."""
        source = self.config.weights
        if source and not Path(source).exists():
            if self.model_key in ("omega", "pi3"):
                raise FileNotFoundError(source)
            from huggingface_hub import snapshot_download

            source = snapshot_download(
                repo_id=source,
                allow_patterns=[
                    "*.json",
                    "*.yaml",
                    "*.safetensors",
                    "*.bin",
                    "*.pt",
                    "*.pth",
                ],
            )
        self.weight_path = source
        return source

    def prepare_images(self, paths):
        """Scene-fixed resize/padding, never recomputed per teacher context."""
        import numpy as np
        from PIL import Image

        images, metadata = [], []
        p = self.patch_size
        for path in paths:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = self.resolution / max(h, w)
                nh, nw = (
                    max(p, round(h * scale / p) * p),
                    max(p, round(w * scale / p) * p),
                )
                array = (
                    np.asarray(
                        im.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32
                    ).copy()
                    / 255
                )
            images.append(torch.from_numpy(array).permute(2, 0, 1))
            metadata.append(
                {
                    "path": str(path),
                    "original_hw": [h, w],
                    "resized_hw": [nh, nw],
                    "image_transform": [[nw / w, 0, 0], [0, nh / h, 0], [0, 0, 1]],
                    "padding": "bottom_right",
                }
            )
        h, w = max(x.shape[-2] for x in images), max(x.shape[-1] for x in images)
        output = (
            torch.tensor(self.mask_fill)
            .view(1, 3, 1, 1)
            .expand(len(images), 3, h, w)
            .clone()
        )
        valid = torch.zeros(len(images), h, w, dtype=torch.bool)
        for i, image in enumerate(images):
            nh, nw = image.shape[-2:]
            output[i, :, :nh, :nw] = image
            valid[i, :nh, :nw] = True
        return output, valid, metadata

    def predict(self, images, frame_ids, valid, teacher=False, slots=None):
        if teacher:
            slots = list(range(len(frame_ids))) if slots is None else list(slots)
            with torch.no_grad():
                out = self.forward_teacher(images, slots)
            ids = tuple(frame_ids[i] for i in slots)
            support = valid[slots]
        else:
            out = self.forward_student(images)
            ids, support = tuple(frame_ids), valid
        result = ModelOutput(
            ids,
            out["readouts"],
            out["depth"],
            out["conf"],
            out["ext_w2c"],
            out["centers"],
            support,
            self.patch_hw,
        ).check()
        return result.to(images.device, detach=True) if teacher else result

    def trainable_model(self):
        return self.net

    def student_params(self):
        return [p for p in self.trainable_model().parameters() if p.requires_grad]

    def trainable_state(self):
        return {
            n: p.detach().cpu().clone()
            for n, p in self.trainable_model().named_parameters()
            if p.requires_grad
        }

    def load_trainable_state(self, state):
        params = {
            n: p
            for n, p in self.trainable_model().named_parameters()
            if p.requires_grad
        }
        if params.keys() != state.keys():
            raise ValueError("checkpoint trainable parameter names do not match model")
        with torch.no_grad():
            for name, p in params.items():
                if p.shape != state[name].shape:
                    raise ValueError(f"checkpoint shape mismatch: {name}")
                p.copy_(state[name].to(p.device, p.dtype))

    def describe(self):
        params = {
            n: p.numel()
            for n, p in self.trainable_model().named_parameters()
            if p.requires_grad
        }
        if not params:
            raise ValueError("adapter has no trainable parameters")
        return {
            "model": self.model_key,
            "weights": self.config.weights,
            "trainable_parameters": params,
            "total_trainable": sum(params.values()),
            "resolution": self.resolution,
            "patch_size": self.patch_size,
        }
