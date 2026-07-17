# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import VGGT


class Test3RPromptAggregator(Aggregator):
    """VGGT Aggregator with Test3R-style deep visual prompts.

    Mirrors Test3R's encoder-only prompt design by mapping:
      * Test3R's per-image encoder self-attn ↔ VGGT frame-attention.
      * Test3R's cross-view decoder            ↔ VGGT global-attention.

    So prompts are injected ONLY at each frame-attention block (per-image,
    just like Test3R), and stripped back out before every global-attention
    block (cross-frame, which Test3R's decoder never receives prompts in).
    Prompts are deep VPT: one learnable tensor per aggregator block-pair.

    The prompt parameter is an nn.Parameter and is not "consumed" by the
    forward pass; its attention-updated outputs during a frame-attn block
    are what get stripped before global-attn. The next block-pair re-injects
    a fresh learnable prompt_{pair_i+1}. Gradient still flows back through
    the patch tokens that were influenced by the prompt during frame-attn.

    All heads see prompt-free tokens (matching Test3R's post-encoder strip).
    """

    def __init__(self, *args, prompt_size: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = self.camera_token.shape[-1]
        self.prompt_size = int(prompt_size)
        self.use_ttt_prompt = self.prompt_size > 0
        if self.use_ttt_prompt:
            # One prompt per block-pair (deep VPT).
            self.ttt_prompt = nn.Parameter(
                torch.zeros(self.aa_block_num, 1, 2, self.prompt_size, self.embed_dim)
            )
        else:
            self.register_parameter("ttt_prompt", None)

    @torch.no_grad()
    def reset_ttt_prompt(self) -> None:
        if self.ttt_prompt is not None:
            self.ttt_prompt.zero_()

    @property
    def prompt_patch_start_idx(self) -> int:
        return self.patch_start_idx

    def _inject_prompt(
        self,
        tokens_real: torch.Tensor,  # (B, S, P_real, C)
        pair_i: int,
        B: int,
        S: int,
        ps: int,
        psz: int,
    ) -> torch.Tensor:
        """Splice ttt_prompt[pair_i] between special tokens and patches."""
        prompt_i = slice_expand_and_flatten(self.ttt_prompt[pair_i], B, S).view(B, S, psz, -1)
        return torch.cat(
            [tokens_real[:, :, :ps, :], prompt_i, tokens_real[:, :, ps:, :]],
            dim=2,
        )

    def forward(
        self,
        images: torch.Tensor,
        output_layers: Optional[List[int]] = None,
        disable_prompt: bool = False,
    ) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        use_prompt = (
            self.use_ttt_prompt
            and self.ttt_prompt is not None
            and not disable_prompt
        )
        psz = self.prompt_size
        ps = self.patch_start_idx  # start index of patch tokens (prompt-free layout)

        # Real-only tokens: this is the state that flows through global-attn
        # and is handed to the heads at the end. Shape (B, S, P_real, C).
        tokens_real = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, C = tokens_real.shape[-1], tokens_real.shape[-1]
        P_real = tokens_real.shape[1]
        tokens_real = tokens_real.view(B, S, P_real, -1)

        # Positions
        pos_patches = None
        if self.rope is not None:
            pos_patches = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )
            pos_patches = pos_patches + 1

        # Frame-attn positions (prompt included): zeros for [camera, register, prompt]
        # Global-attn positions (prompt stripped): zeros for [camera, register]
        pos_global = None
        pos_frame_with_prompt = None
        if pos_patches is not None:
            pos_special = torch.zeros(
                B * S, ps, 2, device=images.device, dtype=pos_patches.dtype
            )
            pos_global = torch.cat([pos_special, pos_patches], dim=1)  # (B*S, ps + Pp, 2)
            if use_prompt:
                pos_prompt = torch.zeros(
                    B * S, psz, 2, device=images.device, dtype=pos_patches.dtype
                )
                pos_frame_with_prompt = torch.cat(
                    [pos_special, pos_prompt, pos_patches], dim=1
                )  # (B*S, ps + psz + Pp, 2)

        frame_idx = 0
        global_idx = 0
        layer_idx = 0
        output_layer_set = set(output_layers) if output_layers is not None else None
        output_list = []

        for pair_i in range(self.aa_block_num):
            # ---------- Frame attention (prompt active) ----------
            if use_prompt:
                tokens_with_prompt = self._inject_prompt(
                    tokens_real, pair_i, B, S, ps, psz
                )
                P_with = tokens_with_prompt.shape[2]
                tokens_bs = tokens_with_prompt.reshape(B * S, P_with, tokens_real.shape[-1])
                pos_for_frame = pos_frame_with_prompt
            else:
                P_with = P_real
                tokens_bs = tokens_real.reshape(B * S, P_with, tokens_real.shape[-1])
                pos_for_frame = pos_global

            frame_intermediates = []
            for _ in range(self.aa_block_size):
                if self.training:
                    tokens_bs = checkpoint(
                        self.frame_blocks[frame_idx],
                        tokens_bs,
                        pos_for_frame,
                        use_reentrant=self.use_reentrant,
                    )
                else:
                    tokens_bs = self.frame_blocks[frame_idx](tokens_bs, pos=pos_for_frame)
                frame_idx += 1
                frame_intermediates.append(tokens_bs.view(B, S, P_with, -1))

            # ---------- Strip prompt before global attn ----------
            tokens_bsp = tokens_bs.view(B, S, P_with, -1)
            if use_prompt:
                tokens_real = torch.cat(
                    [tokens_bsp[:, :, :ps, :], tokens_bsp[:, :, ps + psz :, :]],
                    dim=2,
                )
            else:
                tokens_real = tokens_bsp

            # ---------- Global attention (prompt-free) ----------
            tokens_global = tokens_real.reshape(B, S * P_real, -1)
            # Global-attn expects pos shaped (B, S*P_real, 2); our pos_global
            # is (B*S, P_real, 2). Reshape (matches base Aggregator behaviour).
            if pos_global is not None:
                pos_global_flat = pos_global.view(B, S, P_real, 2).view(B, S * P_real, 2)
            else:
                pos_global_flat = None
            global_intermediates = []
            for _ in range(self.aa_block_size):
                if self.training:
                    tokens_global = checkpoint(
                        self.global_blocks[global_idx],
                        tokens_global,
                        pos_global_flat,
                        use_reentrant=self.use_reentrant,
                    )
                else:
                    tokens_global = self.global_blocks[global_idx](tokens_global, pos=pos_global_flat)
                global_idx += 1
                global_intermediates.append(tokens_global.view(B, S, P_real, -1))

            tokens_real = tokens_global.view(B, S, P_real, -1)

            # ---------- Record intermediates (prompt-free) for heads ----------
            for i in range(len(frame_intermediates)):
                # Strip prompt from the frame intermediate so it matches the
                # (B, S, P_real, C) shape expected by the heads.
                frame_inter = frame_intermediates[i]
                if use_prompt:
                    frame_inter = torch.cat(
                        [frame_inter[:, :, :ps, :], frame_inter[:, :, ps + psz :, :]],
                        dim=2,
                    )
                concat_inter = torch.cat(
                    [frame_inter, global_intermediates[i]], dim=-1
                )
                if output_layer_set is None or layer_idx in output_layer_set:
                    output_list.append(concat_inter)
                layer_idx += 1

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx


class VGGTTest3R(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        prompt_size: int = 32,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        enable_track: bool = True,
    ):
        super().__init__()
        self.aggregator = Test3RPromptAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            prompt_size=prompt_size,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def reset_ttt_prompt(self) -> None:
        self.aggregator.reset_ttt_prompt()

    def set_prompt_trainable_only(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        if self.aggregator.ttt_prompt is None:
            raise RuntimeError("VGGTTest3R was created with prompt_size=0")
        self.aggregator.ttt_prompt.requires_grad_(True)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Single aggregator pass. Prompts influence real tokens only inside
        # each frame-attention block (Test3R-style encoder-only prompting);
        # they are stripped before global-attn and before heads. All heads
        # therefore see the plain VGGT [camera, register, patch] layout.
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images
        return predictions


def load_vggt_test3r(
    model_name: str = "facebook/vggt-1b",
    prompt_size: int = 32,
    device: Union[str, torch.device] = "cuda",
    **kwargs,
) -> VGGTTest3R:
    base = VGGT.from_pretrained(model_name)
    model = VGGTTest3R(prompt_size=prompt_size, **kwargs)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    allowed_missing = {"aggregator.ttt_prompt"}
    extra_missing = set(missing) - allowed_missing
    if extra_missing or unexpected:
        raise RuntimeError(
            "Failed to load VGGT weights into VGGTTest3R: "
            f"missing={sorted(extra_missing)}, unexpected={sorted(unexpected)}"
        )
    del base
    model.to(device)
    model.eval()
    return model
