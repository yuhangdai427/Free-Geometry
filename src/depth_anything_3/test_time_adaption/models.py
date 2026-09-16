"""
Teacher and student model wrappers for DA3 Free-Geometry.

This module provides model wrappers that:
- Extract intermediate features at specified layers
- Separate global features from concatenated output (when cat_token=True)
- Handle camera token extraction (from global features only, NOT local)
- Use HuggingFace PEFT for LoRA adaptation

Key classes:
- TeacherModel: Frozen DA3-Giant for 8-view processing
- StudentModel: DA3-Giant + PEFT LoRA for 4-view processing
- FreeGeometryOutput: Container for extracted features

Note on camera tokens:
    The backbone returns tokens at position 0 as [local_cls, camera_token] concatenated.
    We extract only the second half (camera_token) since the first half (local_cls) contains
    no camera information - it's just a regular cls token from per-view processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3

# HuggingFace PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer


def _swiglu_unfused_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Unfused SwiGLU forward that calls self.w12(x) / self.w3(x) so LoRA hooks fire."""
    x12 = self.w12(x)
    x1, x2 = x12.chunk(2, dim=-1)
    hidden = F.silu(x1) * x2
    return self.w3(hidden)


def patch_swiglu_for_lora(model: nn.Module, lora_layers: List[int]) -> None:
    """Replace SwiGLUFFNFused.forward with unfused version on LoRA layers so MLP LoRA gets gradients."""
    from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFNFused
    # Access blocks through PEFT wrapper (PEFT: model.base_model.model -> original backbone).
    if hasattr(model, "base_model"):
        base = model.base_model
        base = base.model if hasattr(base, "model") else base
        blocks = base.blocks
    else:
        blocks = model.blocks
    patched = 0
    for idx in lora_layers:
        mlp = blocks[idx].mlp
        if isinstance(mlp, SwiGLUFFNFused):
            import types
            mlp.forward = types.MethodType(_swiglu_unfused_forward, mlp)
            patched += 1
    print(f"Patched {patched} SwiGLUFFNFused layers to unfused forward for LoRA training")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class FreeGeometryOutput:
    """
    Container for Free-Geometry outputs.

    Attributes:
        layer_features: Full features per layer [B, S, P, 3072] (local + global)
        camera_tokens: Camera tokens per layer [B, S, embed_dim] - extracted from global
                       features only (second half of concatenated output at position 0).
                       This is the actual camera token injected at alt_start layer.
        camera_tokens_full: Full pos0 token per layer [B, S, 2*embed_dim] - the complete
                            [local_x[:,:,0], x[:,:,0]] that goes to camera decoder.
        global_features: Global features only per layer [B, S, P, embed_dim]
        local_features: Local features only per layer [B, S, P, embed_dim]
    """
    layer_features: Dict[int, torch.Tensor]
    camera_tokens: Dict[int, torch.Tensor]
    camera_tokens_full: Dict[int, torch.Tensor]  # Full 3072-dim token for camera decoder
    global_features: Dict[int, torch.Tensor]
    local_features: Dict[int, torch.Tensor]


class TeacherModel(nn.Module):
    """
    Teacher model for Free-Geometry.

    Wraps a frozen DA3-Giant model and extracts intermediate features
    at specified output layers.

    Args:
        model_name: HuggingFace model name or local path
        output_layers: Layer indices to extract features from
        embed_dim: Embedding dimension (1536 for Giant)
        ref_view_strategy: Reference view selection strategy ('first', 'saddle_balanced', etc.)
    """

    # Default output layers for DA3-Giant (layers 33 and 39 for Free-Geometry)
    DEFAULT_OUTPUT_LAYERS = [33, 39]

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-GIANT-1.1",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1536,
        ref_view_strategy: str = "first",
    ):
        super().__init__()
        self.output_layers = output_layers or self.DEFAULT_OUTPUT_LAYERS
        self.embed_dim = embed_dim
        self.ref_view_strategy = ref_view_strategy

        # Load pretrained model
        print(f"Loading teacher model: {model_name}")
        self.da3 = DepthAnything3.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.da3.parameters():
            param.requires_grad = False

        self.eval()

        # Print parameter count
        total, trainable = count_parameters(self.da3)
        print(f"Teacher parameters: {total:,} total, {trainable:,} trainable")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> FreeGeometryOutput:
        """
        Forward pass with feature extraction.

        Args:
            images: [B, S, 3, H, W] input images (S=8 for teacher)

        Returns:
            FreeGeometryOutput with layer features and camera tokens
        """
        B, S, C, H, W = images.shape

        # Get backbone
        backbone = self.da3.model.backbone

        # Forward through backbone with feature extraction
        # feats is a tuple of (features, camera_token) per output layer
        # Use configurable ref_view_strategy
        feats, aux_feats = backbone(
            images,
            cam_token=None,  # Use learned camera tokens
            export_feat_layers=self.output_layers,
            ref_view_strategy=self.ref_view_strategy,
        )

        # Process outputs — map feats by backbone.out_layers, not self.output_layers
        backbone_out_layers = backbone.out_layers
        feats_by_layer = {layer_idx: ft for ft, layer_idx in zip(feats, backbone_out_layers)}

        layer_features = {}
        camera_tokens = {}
        camera_tokens_full = {}
        global_features = {}
        local_features = {}

        for layer_idx in self.output_layers:
            feat_tuple = feats_by_layer[layer_idx]
            features = feat_tuple[0]  # [B, S, P, 3072]
            cam_token_raw = feat_tuple[1]  # [B, S, 3072]
            cam_token = cam_token_raw[..., self.embed_dim:]  # [B, S, 1536]

            layer_features[layer_idx] = features
            camera_tokens[layer_idx] = cam_token
            camera_tokens_full[layer_idx] = cam_token_raw
            global_features[layer_idx] = features[..., self.embed_dim:]
            local_features[layer_idx] = features[..., :self.embed_dim]

        return FreeGeometryOutput(
            layer_features=layer_features,
            camera_tokens=camera_tokens,
            camera_tokens_full=camera_tokens_full,
            global_features=global_features,
            local_features=local_features,
        )

    @torch.no_grad()
    def forward_with_preds(self, images: torch.Tensor) -> tuple[FreeGeometryOutput, Dict[str, torch.Tensor]]:
        """Forward that also returns DA3 head predictions (depth/camera/etc.)."""
        # Backbone-only pass to get Free-Geometry features and to feed head-only.
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images,
            extrinsics=None,
            intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )

        # Extract Free-Geometry features from the backbone outputs.
        layer_features = {}
        camera_tokens = {}
        camera_tokens_full = {}
        global_features = {}
        local_features = {}

        for feat_tuple, layer_idx in zip(feats, self.output_layers):
            features = feat_tuple[0]  # [B, S, P, 3072]
            cam_token_raw = feat_tuple[1]  # [B, S, 3072] (local_cls + camera_token)
            cam_token = cam_token_raw[..., self.embed_dim:]

            local_feat = features[..., : self.embed_dim]
            global_feat = features[..., self.embed_dim :]

            layer_features[layer_idx] = features
            camera_tokens[layer_idx] = cam_token
            camera_tokens_full[layer_idx] = cam_token_raw
            global_features[layer_idx] = global_feat
            local_features[layer_idx] = local_feat

        free_geometry_out = FreeGeometryOutput(
            layer_features=layer_features,
            camera_tokens=camera_tokens,
            camera_tokens_full=camera_tokens_full,
            global_features=global_features,
            local_features=local_features,
        )

        # Head-only predictions (depth/depth_conf/extrinsics/intrinsics/sky/...).
        preds = self.da3.model.forward_head_only(feats, H=H, W=W, process_camera=True, process_sky=True)
        return free_geometry_out, preds

    @torch.no_grad()
    def forward_features_only(self, images: torch.Tensor) -> FreeGeometryOutput:
        """Forward through backbone only, skipping decoder heads."""
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images,
            extrinsics=None,
            intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )

        layer_features = {}
        camera_tokens = {}
        camera_tokens_full = {}
        global_features = {}
        local_features = {}

        for feat_tuple, layer_idx in zip(feats, self.output_layers):
            features = feat_tuple[0]
            cam_token_raw = feat_tuple[1]
            cam_token = cam_token_raw[..., self.embed_dim:]

            local_feat = features[..., : self.embed_dim]
            global_feat = features[..., self.embed_dim :]

            layer_features[layer_idx] = features
            camera_tokens[layer_idx] = cam_token
            camera_tokens_full[layer_idx] = cam_token_raw
            global_features[layer_idx] = global_feat
            local_features[layer_idx] = local_feat

        return FreeGeometryOutput(
            layer_features=layer_features,
            camera_tokens=camera_tokens,
            camera_tokens_full=camera_tokens_full,
            global_features=global_features,
            local_features=local_features,
        )


class StudentModel(nn.Module):
    """
    Student model for Free-Geometry using HuggingFace PEFT.

    Wraps DA3-Giant with PEFT LoRA adapters on attention layers.
    Only LoRA parameters and camera tokens are trainable.

    Args:
        model_name: HuggingFace model name or local path
        output_layers: Layer indices to extract features from
        embed_dim: Embedding dimension (1536 for Giant)
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout probability
        train_camera_token: Whether to make camera token trainable
        lora_layers: Which layers to apply LoRA to (default: 13-39)
        ref_view_strategy: Reference view selection strategy ('first', 'saddle_balanced', etc.)
    """

    DEFAULT_OUTPUT_LAYERS = [33, 39]

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-GIANT-1.1",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1536,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        train_camera_token: bool = True,
        lora_layers: Optional[List[int]] = None,
        ref_view_strategy: str = "first",
        patch_swiglu_mlp_for_lora: bool = True,
    ):
        super().__init__()
        self.output_layers = output_layers or self.DEFAULT_OUTPUT_LAYERS
        self.embed_dim = embed_dim
        self.lora_rank = lora_rank
        self.train_camera_token = train_camera_token
        self.ref_view_strategy = ref_view_strategy
        self.patch_swiglu_mlp_for_lora = patch_swiglu_mlp_for_lora

        # Default: apply LoRA to layers 13-39 (after alt_start where cross-view attention happens)
        self.lora_layers = lora_layers or list(range(13, 40))

        # Load pretrained model
        print(f"Loading student model: {model_name}")
        self.da3 = DepthAnything3.from_pretrained(model_name)

        # Freeze ALL parameters first
        for param in self.da3.parameters():
            param.requires_grad = False

        # Apply PEFT LoRA to backbone (this will unfreeze LoRA params)
        print(f"Applying PEFT LoRA (rank={lora_rank}, alpha={lora_alpha}) to layers {self.lora_layers[0]}-{self.lora_layers[-1]}")
        self._apply_peft_lora(lora_rank, lora_alpha, lora_dropout)

        # Make camera token trainable if requested
        if train_camera_token:
            _bb = self.da3.model.backbone.pretrained
            _pre = getattr(_bb, "base_model", _bb)  # unwrap PeftModel -> LoraModel
            _pre = getattr(_pre, "model", _pre)     # unwrap LoraModel -> DinoV2
            if hasattr(_pre, 'camera_token'):
                _pre.camera_token.requires_grad = True
                print("Camera token is trainable")
            else:
                print("WARNING: camera_token not found on backbone (unfreeze FAILED)")

        # Print parameter count
        total, trainable = count_parameters(self.da3)
        print(f"Student parameters: {total:,} total, {trainable:,} trainable")

    def _apply_peft_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        """Apply PEFT LoRA to the backbone transformer blocks."""
        backbone = self.da3.model.backbone.pretrained

        # Note: All parameters are already frozen in __init__
        # PEFT will add trainable LoRA parameters

        # Build target modules list for specific layers
        # Format: blocks.{layer_idx}.attn.qkv, blocks.{layer_idx}.attn.proj
        #         blocks.{layer_idx}.mlp.w12, blocks.{layer_idx}.mlp.w3 (for SwiGLU)
        target_modules = []
        for layer_idx in self.lora_layers:
            # Attention layers
            target_modules.append(f"blocks.{layer_idx}.attn.qkv")
            target_modules.append(f"blocks.{layer_idx}.attn.proj")
            # MLP layers (SwiGLU: w12 and w3)
            target_modules.append(f"blocks.{layer_idx}.mlp.w12")
            target_modules.append(f"blocks.{layer_idx}.mlp.w3")

        # Create LoRA config
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )

        # Apply PEFT to backbone
        self.da3.model.backbone.pretrained = get_peft_model(backbone, lora_config)

        # Patch SwiGLUFFNFused to use unfused forward so MLP LoRA actually gets gradients
        # Keep this optional so inference can stay fully fused after merging LoRA.
        if self.patch_swiglu_mlp_for_lora:
            patch_swiglu_for_lora(self.da3.model.backbone.pretrained, self.lora_layers)

        # Count LoRA parameters
        lora_params = sum(
            p.numel() for n, p in self.da3.model.backbone.pretrained.named_parameters()
            if p.requires_grad and 'lora' in n.lower()
        )
        print(f"Total PEFT LoRA parameters: {lora_params:,}")

    def forward(
        self,
        images: torch.Tensor,
    ) -> FreeGeometryOutput:
        """
        Forward pass with feature extraction.

        Args:
            images: [B, S, 3, H, W] input images (S=4 for student)

        Returns:
            FreeGeometryOutput with layer features and camera tokens
        """
        B, S, C, H, W = images.shape

        # Get backbone
        backbone = self.da3.model.backbone

        # Forward through backbone with feature extraction
        # Use configurable ref_view_strategy
        feats, aux_feats = backbone(
            images,
            cam_token=None,  # Use learned camera tokens
            export_feat_layers=self.output_layers,
            ref_view_strategy=self.ref_view_strategy,
        )

        # Process outputs
        # backbone.out_layers (e.g. [19,27,33,39]) determines which layers
        # produce entries in feats. Map them correctly to self.output_layers.
        backbone_out_layers = backbone.out_layers
        tta_out = self._extract_tta_output(feats, backbone_out_layers)

        return tta_out

    def _extract_tta_output(self, feats, backbone_out_layers):
        """Map backbone feats (indexed by backbone.out_layers) to self.output_layers."""
        feats_by_layer = {layer_idx: ft for ft, layer_idx in zip(feats, backbone_out_layers)}

        layer_features = {}
        camera_tokens = {}
        camera_tokens_full = {}
        global_features = {}
        local_features = {}

        for layer_idx in self.output_layers:
            feat_tuple = feats_by_layer[layer_idx]
            features = feat_tuple[0]  # [B, S, P, 3072]
            cam_token_raw = feat_tuple[1]  # [B, S, 3072]
            cam_token = cam_token_raw[..., self.embed_dim:]  # [B, S, 1536]

            layer_features[layer_idx] = features
            camera_tokens[layer_idx] = cam_token
            camera_tokens_full[layer_idx] = cam_token_raw
            global_features[layer_idx] = features[..., self.embed_dim:]
            local_features[layer_idx] = features[..., :self.embed_dim]

        return FreeGeometryOutput(
            layer_features=layer_features,
            camera_tokens=camera_tokens,
            camera_tokens_full=camera_tokens_full,
            global_features=global_features,
            local_features=local_features,
        )

    def forward_with_preds(self, images: torch.Tensor) -> tuple[FreeGeometryOutput, Dict[str, torch.Tensor]]:
        """Forward that also returns DA3 head predictions (depth/camera/etc.)."""
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images,
            extrinsics=None,
            intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )

        backbone_out_layers = self.da3.model.backbone.out_layers
        tta_out = self._extract_tta_output(feats, backbone_out_layers)

        preds = self.da3.model.forward_head_only(feats, H=H, W=W, process_camera=True, process_sky=True)
        return tta_out, preds

    def forward_features_only(self, images: torch.Tensor) -> FreeGeometryOutput:
        """Forward through backbone only, skipping decoder heads.
        Uses same code path as forward_with_preds() to preserve gradient flow."""
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images,
            extrinsics=None,
            intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )

        backbone_out_layers = self.da3.model.backbone.out_layers
        return self._extract_tta_output(feats, backbone_out_layers)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA + camera tokens)."""
        return [p for p in self.da3.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.da3.parameters() if p.requires_grad)

    def save_lora_weights(self, path: str) -> None:
        """Save PEFT LoRA weights and camera token to file."""
        # Get the PEFT model
        peft_model = self.da3.model.backbone.pretrained

        # Save PEFT adapter
        peft_model.save_pretrained(path.replace('.pt', '_peft'))

        # Also save camera token if trainable
        state_dict = {}
        backbone = peft_model
        if hasattr(backbone, 'camera_token') and backbone.camera_token.requires_grad:
            state_dict['camera_token'] = backbone.camera_token.data.clone()
        elif hasattr(backbone, 'base_model'):
            base = backbone.base_model.model if hasattr(backbone.base_model, 'model') else backbone.base_model
            if hasattr(base, 'camera_token') and base.camera_token.requires_grad:
                state_dict['camera_token'] = base.camera_token.data.clone()

        if state_dict:
            torch.save(state_dict, path)
            print(f"Saved camera token to {path}")

        print(f"Saved PEFT LoRA weights to {path.replace('.pt', '_peft')}")

    def load_lora_weights(self, path: str) -> None:
        """Load PEFT LoRA weights and camera token from file."""
        peft_path = path.replace('.pt', '_peft')

        # Load PEFT adapter
        peft_model = self.da3.model.backbone.pretrained
        if hasattr(peft_model, 'load_adapter'):
            peft_model.load_adapter(peft_path, adapter_name="default")
            # IMPORTANT: Activate the loaded adapter
            if hasattr(peft_model, 'set_adapter'):
                peft_model.set_adapter("default")
            print(f"Loaded and activated PEFT LoRA weights from {peft_path}")

        # Load camera token if present
        import os
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            if 'camera_token' in state_dict:
                backbone = peft_model
                if hasattr(backbone, 'camera_token'):
                    backbone.camera_token.data.copy_(state_dict['camera_token'])
                elif hasattr(backbone, 'base_model'):
                    base = backbone.base_model.model if hasattr(backbone.base_model, 'model') else backbone.base_model
                    if hasattr(base, 'camera_token'):
                        base.camera_token.data.copy_(state_dict['camera_token'])
                print("Loaded camera token")


class DA3StudentFinetune(nn.Module):
    """
    Student model that directly unfreezes layers 13-39 (no LoRA).

    Same interface as StudentModel but with full fine-tuning of the
    specified transformer blocks instead of low-rank adapters.
    More parameters to train, so use a smaller learning rate.
    """

    DEFAULT_OUTPUT_LAYERS = [33, 39]

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-GIANT-1.1",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1536,
        train_camera_token: bool = True,
        finetune_layers: Optional[List[int]] = None,
        ref_view_strategy: str = "first",
    ):
        super().__init__()
        self.output_layers = output_layers or self.DEFAULT_OUTPUT_LAYERS
        self.embed_dim = embed_dim
        self.train_camera_token = train_camera_token
        self.ref_view_strategy = ref_view_strategy
        self.finetune_layers = finetune_layers or list(range(13, 40))

        # Load pretrained model
        print(f"Loading student model (finetune): {model_name}")
        self.da3 = DepthAnything3.from_pretrained(model_name)

        # Freeze ALL parameters first
        for param in self.da3.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        backbone = self.da3.model.backbone.pretrained
        unfrozen = 0
        for layer_idx in self.finetune_layers:
            block = backbone.blocks[layer_idx]
            for param in block.parameters():
                param.requires_grad = True
                unfrozen += param.numel()
        print(f"Unfroze layers {self.finetune_layers[0]}-{self.finetune_layers[-1]}: {unfrozen:,} parameters")

        # Make camera token trainable if requested
        if train_camera_token:
            if hasattr(backbone, 'camera_token'):
                backbone.camera_token.requires_grad = True
                print("Camera token is trainable")

        # Print parameter count
        total, trainable = count_parameters(self.da3)
        print(f"Student parameters: {total:,} total, {trainable:,} trainable")

    def forward(self, images: torch.Tensor) -> FreeGeometryOutput:
        B, S, C, H, W = images.shape
        backbone = self.da3.model.backbone
        feats, aux_feats = backbone(
            images,
            cam_token=None,
            export_feat_layers=self.output_layers,
            ref_view_strategy=self.ref_view_strategy,
        )
        backbone_out_layers = backbone.out_layers
        return self._extract_tta_output(feats, backbone_out_layers)

    def _extract_tta_output(self, feats, backbone_out_layers):
        feats_by_layer = {layer_idx: ft for ft, layer_idx in zip(feats, backbone_out_layers)}
        layer_features = {}
        camera_tokens = {}
        camera_tokens_full = {}
        global_features = {}
        local_features = {}

        for layer_idx in self.output_layers:
            feat_tuple = feats_by_layer[layer_idx]
            features = feat_tuple[0]
            cam_token_raw = feat_tuple[1]
            cam_token = cam_token_raw[..., self.embed_dim:]

            layer_features[layer_idx] = features
            camera_tokens[layer_idx] = cam_token
            camera_tokens_full[layer_idx] = cam_token_raw
            global_features[layer_idx] = features[..., self.embed_dim:]
            local_features[layer_idx] = features[..., :self.embed_dim]

        return FreeGeometryOutput(
            layer_features=layer_features,
            camera_tokens=camera_tokens,
            camera_tokens_full=camera_tokens_full,
            global_features=global_features,
            local_features=local_features,
        )

    def forward_with_preds(self, images: torch.Tensor) -> tuple[FreeGeometryOutput, Dict[str, torch.Tensor]]:
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images, extrinsics=None, intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )
        backbone_out_layers = self.da3.model.backbone.out_layers
        tta_out = self._extract_tta_output(feats, backbone_out_layers)
        preds = self.da3.model.forward_head_only(feats, H=H, W=W, process_camera=True, process_sky=True)
        return tta_out, preds

    def forward_features_only(self, images: torch.Tensor) -> FreeGeometryOutput:
        feats, _, H, W = self.da3.model.forward_backbone_only(
            images, extrinsics=None, intrinsics=None,
            ref_view_strategy=self.ref_view_strategy,
        )
        backbone_out_layers = self.da3.model.backbone.out_layers
        return self._extract_tta_output(feats, backbone_out_layers)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.da3.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.da3.parameters() if p.requires_grad)

    def save_finetune_weights(self, path: str) -> None:
        """Save only the finetuned layer weights and camera token."""
        backbone = self.da3.model.backbone.pretrained
        state_dict = {}
        for layer_idx in self.finetune_layers:
            block = backbone.blocks[layer_idx]
            for name, param in block.named_parameters():
                state_dict[f"blocks.{layer_idx}.{name}"] = param.data.clone()
        if hasattr(backbone, 'camera_token') and backbone.camera_token.requires_grad:
            state_dict['camera_token'] = backbone.camera_token.data.clone()
        torch.save(state_dict, path)
        print(f"Saved finetune weights ({len(state_dict)} tensors) to {path}")

    def load_finetune_weights(self, path: str) -> None:
        """Load finetuned layer weights and camera token."""
        state_dict = torch.load(path, map_location='cpu')
        backbone = self.da3.model.backbone.pretrained
        loaded = 0
        for key, value in state_dict.items():
            if key == 'camera_token':
                if hasattr(backbone, 'camera_token'):
                    backbone.camera_token.data.copy_(value)
                    loaded += 1
            else:
                # key format: blocks.{idx}.{rest}
                parts = key.split('.', 2)
                layer_idx = int(parts[1])
                param_name = parts[2]
                block = backbone.blocks[layer_idx]
                param = dict(block.named_parameters())[param_name]
                param.data.copy_(value)
                loaded += 1
        print(f"Loaded {loaded} finetune tensors from {path}")


def create_teacher_student_pair(
    model_name: str = "depth-anything/DA3-GIANT-1.1",
    output_layers: Optional[List[int]] = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    device: str = "cuda",
) -> Tuple[TeacherModel, StudentModel]:
    """
    Create a teacher-student pair for Free-Geometry.

    Args:
        model_name: HuggingFace model name
        output_layers: Layer indices to extract features from
        lora_rank: LoRA rank for student
        lora_alpha: LoRA alpha for student
        device: Device to load models on

    Returns:
        Tuple of (teacher, student) models
    """
    teacher = TeacherModel(
        model_name=model_name,
        output_layers=output_layers,
    ).to(device)

    student = StudentModel(
        model_name=model_name,
        output_layers=output_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)

    return teacher, student
