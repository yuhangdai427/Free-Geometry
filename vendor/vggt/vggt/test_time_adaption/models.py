"""
Teacher and student model wrappers for VGGT Free-Geometry.

This module provides model wrappers that:
- Extract intermediate features at specified layers from VGGT's Aggregator
- Separate frame features from global features (concatenated output)
- Handle camera token extraction
- Use HuggingFace PEFT for LoRA adaptation

Key classes:
- VGGTTeacherModel: Frozen VGGT for 8-view processing
- VGGTStudentModel: VGGT + PEFT LoRA for 4-view processing
- VGGTFreeGeometryOutput: Container for extracted features
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT

# HuggingFace PEFT imports
from peft import LoraConfig, get_peft_model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class VGGTFreeGeometryOutput:
    """
    Container for VGGT Free-Geometry outputs.

    Attributes:
        layer_features: Full features per layer [B, S, P, 2048] (frame + global)
        frame_features: Frame-only features per layer [B, S, P, 1024]
        global_features: Global-only features per layer [B, S, P, 1024]
        camera_tokens: Camera tokens per layer [B, S, 2048] - from position 0
    """
    layer_features: Dict[int, torch.Tensor]
    frame_features: Dict[int, torch.Tensor]
    global_features: Dict[int, torch.Tensor]
    camera_tokens: Dict[int, torch.Tensor]
    patch_start_idx: int


class VGGTTeacherModel(nn.Module):
    """
    Teacher model for VGGT Free-Geometry.

    Wraps a frozen VGGT model and extracts intermediate features
    at specified output layers from the Aggregator's output_list.

    Args:
        model_name: HuggingFace model name or local path
        output_layers: Layer indices to extract features from (0-23)
        embed_dim: Embedding dimension (1024 for ViT-Large)
    """

    DEFAULT_OUTPUT_LAYERS = [19, 23]

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.output_layers = output_layers or self.DEFAULT_OUTPUT_LAYERS
        self.embed_dim = embed_dim

        # Load pretrained VGGT model
        print(f"Loading teacher model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.vggt.parameters():
            param.requires_grad = False

        self.eval()

        # Print parameter count
        total, trainable = count_parameters(self.vggt)
        print(f"Teacher parameters: {total:,} total, {trainable:,} trainable")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> VGGTFreeGeometryOutput:
        """
        Forward pass with feature extraction.

        Args:
            images: [B, S, 3, H, W] input images (S=8 for teacher), range [0, 1]

        Returns:
            VGGTFreeGeometryOutput with layer features and camera tokens
        """
        B, S, C, H, W = images.shape

        # Get only required layers from aggregator to reduce activation memory.
        requested_layers = sorted(set(self.output_layers))
        output_list, patch_start_idx = self.vggt.aggregator(images, output_layers=requested_layers)
        if len(output_list) != len(requested_layers):
            raise RuntimeError(
                f"Aggregator returned {len(output_list)} layers, expected {len(requested_layers)}"
            )
        layer_to_features = {layer: feat for layer, feat in zip(requested_layers, output_list)}

        # Process outputs at specified layers
        layer_features = {}
        frame_features = {}
        global_features = {}
        camera_tokens = {}

        for layer_idx in self.output_layers:
            if layer_idx not in layer_to_features:
                raise ValueError(f"Requested layer {layer_idx} was not returned by aggregator")
            features = layer_to_features[layer_idx]  # [B, S, P, 2048]

            # Extract camera token from position 0
            cam_token = features[:, :, 0, :]  # [B, S, 2048]

            # Split into frame (first 1024) and global (second 1024)
            frame_feat = features[..., :self.embed_dim]  # [B, S, P, 1024]
            global_feat = features[..., self.embed_dim:]  # [B, S, P, 1024]

            layer_features[layer_idx] = features
            frame_features[layer_idx] = frame_feat
            global_features[layer_idx] = global_feat
            camera_tokens[layer_idx] = cam_token

        return VGGTFreeGeometryOutput(
            layer_features=layer_features,
            frame_features=frame_features,
            global_features=global_features,
            camera_tokens=camera_tokens,
            patch_start_idx=patch_start_idx,
        )

    def _extract_tta_output(
        self,
        all_output_list: List[torch.Tensor],
    ) -> VGGTFreeGeometryOutput:
        """Extract Free-Geometry features from a full (all-layer) output list."""
        layer_features = {}
        frame_features = {}
        global_features = {}
        camera_tokens = {}

        for layer_idx in self.output_layers:
            features = all_output_list[layer_idx]
            cam_token = features[:, :, 0, :]
            frame_feat = features[..., :self.embed_dim]
            global_feat = features[..., self.embed_dim:]

            layer_features[layer_idx] = features
            frame_features[layer_idx] = frame_feat
            global_features[layer_idx] = global_feat
            camera_tokens[layer_idx] = cam_token

        return VGGTFreeGeometryOutput(
            layer_features=layer_features,
            frame_features=frame_features,
            global_features=global_features,
            camera_tokens=camera_tokens,
            patch_start_idx=self.vggt.aggregator.patch_start_idx,
        )

    @torch.no_grad()
    def forward_with_preds(
        self,
        images: torch.Tensor,
    ) -> Tuple[VGGTFreeGeometryOutput, Dict[str, torch.Tensor]]:
        """Forward that returns both Free-Geometry features and head predictions (depth, camera)."""
        # Run full aggregator (all 24 layers) so heads can index freely
        all_output_list, patch_start_idx = self.vggt.aggregator(images)

        free_geometry_out = self._extract_tta_output(all_output_list)

        # Run heads
        predictions: Dict[str, torch.Tensor] = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.vggt.camera_head is not None:
                pose_enc_list = self.vggt.camera_head(all_output_list)
                predictions["pose_enc"] = pose_enc_list[-1]
            if self.vggt.depth_head is not None:
                depth, depth_conf = self.vggt.depth_head(
                    all_output_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if self.vggt.point_head is not None:
                world_points, world_points_conf = self.vggt.point_head(
                    all_output_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = world_points
                predictions["world_points_conf"] = world_points_conf

        return free_geometry_out, predictions


class VGGTStudentModel(nn.Module):
    """
    Student model for VGGT Free-Geometry using HuggingFace PEFT.

    Wraps VGGT with PEFT LoRA adapters on attention and MLP layers.
    Only LoRA parameters and optionally camera tokens are trainable.

    Args:
        model_name: HuggingFace model name or local path
        output_layers: Layer indices to extract features from
        embed_dim: Embedding dimension (1024 for ViT-Large)
        lora_rank: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout probability
        train_camera_token: Whether to make camera token trainable
        lora_layers: Which layers to apply LoRA to (default: 12-23)
    """

    DEFAULT_OUTPUT_LAYERS = [19, 23]

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1024,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        train_camera_token: bool = True,
        lora_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.output_layers = output_layers or self.DEFAULT_OUTPUT_LAYERS
        self.embed_dim = embed_dim
        self.lora_rank = lora_rank
        self.train_camera_token = train_camera_token

        # Default: apply LoRA to layers 12-23 (second half)
        self.lora_layers = lora_layers or list(range(12, 24))

        # Load pretrained model
        print(f"Loading student model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)

        # Freeze ALL parameters first
        for param in self.vggt.parameters():
            param.requires_grad = False

        # Apply PEFT LoRA to aggregator
        print(f"Applying PEFT LoRA (rank={lora_rank}, alpha={lora_alpha}) to layers {self.lora_layers[0]}-{self.lora_layers[-1]}")
        self._apply_peft_lora(lora_rank, lora_alpha, lora_dropout)

        # Make camera token trainable if requested
        if train_camera_token:
            if hasattr(self.vggt.aggregator, 'camera_token'):
                self.vggt.aggregator.camera_token.requires_grad = True
                print("Camera token is trainable")

        # Print parameter count
        total, trainable = count_parameters(self.vggt)
        print(f"Student parameters: {total:,} total, {trainable:,} trainable")

    def _apply_peft_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        """Apply PEFT LoRA to the aggregator's frame_blocks and global_blocks."""
        # Build target modules list for specific layers
        # VGGT Block structure: attn.qkv, attn.proj, mlp.fc1, mlp.fc2
        target_modules = []
        for layer_idx in self.lora_layers:
            # Frame blocks
            target_modules.extend([
                f"aggregator.frame_blocks.{layer_idx}.attn.qkv",
                f"aggregator.frame_blocks.{layer_idx}.attn.proj",
                f"aggregator.frame_blocks.{layer_idx}.mlp.fc1",
                f"aggregator.frame_blocks.{layer_idx}.mlp.fc2",
            ])
            # Global blocks
            target_modules.extend([
                f"aggregator.global_blocks.{layer_idx}.attn.qkv",
                f"aggregator.global_blocks.{layer_idx}.attn.proj",
                f"aggregator.global_blocks.{layer_idx}.mlp.fc1",
                f"aggregator.global_blocks.{layer_idx}.mlp.fc2",
            ])

        # Create LoRA config
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )

        # Apply PEFT to the entire vggt model
        self.vggt = get_peft_model(self.vggt, lora_config)

        # Count LoRA parameters
        lora_params = sum(
            p.numel() for n, p in self.vggt.named_parameters()
            if p.requires_grad and 'lora' in n.lower()
        )
        print(f"Total PEFT LoRA parameters: {lora_params:,}")

    def forward(
        self,
        images: torch.Tensor,
    ) -> VGGTFreeGeometryOutput:
        """
        Forward pass with feature extraction.

        Args:
            images: [B, S, 3, H, W] input images (S=4 for student), range [0, 1]

        Returns:
            VGGTFreeGeometryOutput with layer features and camera tokens
        """
        B, S, C, H, W = images.shape

        aggregator = self._get_aggregator()

        # Get only required layers from aggregator to reduce activation memory.
        requested_layers = sorted(set(self.output_layers))
        output_list, patch_start_idx = aggregator(images, output_layers=requested_layers)
        if len(output_list) != len(requested_layers):
            raise RuntimeError(
                f"Aggregator returned {len(output_list)} layers, expected {len(requested_layers)}"
            )
        layer_to_features = {layer: feat for layer, feat in zip(requested_layers, output_list)}

        # Process outputs at specified layers
        layer_features = {}
        frame_features = {}
        global_features = {}
        camera_tokens = {}

        for layer_idx in self.output_layers:
            if layer_idx not in layer_to_features:
                raise ValueError(f"Requested layer {layer_idx} was not returned by aggregator")
            features = layer_to_features[layer_idx]  # [B, S, P, 2048]

            # Extract camera token from position 0
            cam_token = features[:, :, 0, :]  # [B, S, 2048]

            # Split into frame (first 1024) and global (second 1024)
            frame_feat = features[..., :self.embed_dim]  # [B, S, P, 1024]
            global_feat = features[..., self.embed_dim:]  # [B, S, P, 1024]

            layer_features[layer_idx] = features
            frame_features[layer_idx] = frame_feat
            global_features[layer_idx] = global_feat
            camera_tokens[layer_idx] = cam_token

        return VGGTFreeGeometryOutput(
            layer_features=layer_features,
            frame_features=frame_features,
            global_features=global_features,
            camera_tokens=camera_tokens,
            patch_start_idx=patch_start_idx,
        )

    def _get_aggregator(self):
        """Access aggregator through PEFT wrapper."""
        if hasattr(self.vggt, 'base_model'):
            return self.vggt.base_model.model.aggregator
        return self.vggt.aggregator

    def _get_vggt_model(self):
        """Access underlying VGGT model through PEFT wrapper."""
        if hasattr(self.vggt, 'base_model'):
            return self.vggt.base_model.model
        return self.vggt

    def _extract_tta_output(
        self,
        all_output_list: List[torch.Tensor],
    ) -> VGGTFreeGeometryOutput:
        """Extract Free-Geometry features from a full (all-layer) output list."""
        layer_features = {}
        frame_features = {}
        global_features = {}
        camera_tokens = {}

        for layer_idx in self.output_layers:
            features = all_output_list[layer_idx]
            cam_token = features[:, :, 0, :]
            frame_feat = features[..., :self.embed_dim]
            global_feat = features[..., self.embed_dim:]

            layer_features[layer_idx] = features
            frame_features[layer_idx] = frame_feat
            global_features[layer_idx] = global_feat
            camera_tokens[layer_idx] = cam_token

        return VGGTFreeGeometryOutput(
            layer_features=layer_features,
            frame_features=frame_features,
            global_features=global_features,
            camera_tokens=camera_tokens,
            patch_start_idx=self._get_aggregator().patch_start_idx,
        )

    def forward_with_preds(
        self,
        images: torch.Tensor,
    ) -> Tuple[VGGTFreeGeometryOutput, Dict[str, torch.Tensor]]:
        """Forward that returns both Free-Geometry features and head predictions (depth, camera, points)."""
        aggregator = self._get_aggregator()
        vggt_model = self._get_vggt_model()

        # Run full aggregator (all 24 layers) so heads can index freely
        all_output_list, patch_start_idx = aggregator(images)

        free_geometry_out = self._extract_tta_output(all_output_list)

        # Run heads
        predictions: Dict[str, torch.Tensor] = {}
        with torch.cuda.amp.autocast(enabled=False):
            if vggt_model.camera_head is not None:
                pose_enc_list = vggt_model.camera_head(all_output_list)
                predictions["pose_enc"] = pose_enc_list[-1]
            if vggt_model.depth_head is not None:
                depth, depth_conf = vggt_model.depth_head(
                    all_output_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if vggt_model.point_head is not None:
                pts3d, pts3d_conf = vggt_model.point_head(
                    all_output_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return free_geometry_out, predictions

    def forward_features_only(
        self,
        images: torch.Tensor,
    ) -> VGGTFreeGeometryOutput:
        """Forward through aggregator only, skipping heads. Same as forward()."""
        return self.forward(images)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA + camera tokens)."""
        return [p for p in self.vggt.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.vggt.parameters() if p.requires_grad)

    def save_lora_weights(self, path: str) -> None:
        """Save PEFT LoRA weights and camera token to file."""
        import os

        # Save PEFT adapter
        peft_path = path.replace('.pt', '_peft')
        self.vggt.save_pretrained(peft_path)

        # Also save camera token if trainable
        state_dict = {}
        aggregator = self._get_aggregator()

        if hasattr(aggregator, 'camera_token') and aggregator.camera_token.requires_grad:
            state_dict['camera_token'] = aggregator.camera_token.data.clone()

        if state_dict:
            torch.save(state_dict, path)
            print(f"Saved camera token to {path}")

        print(f"Saved PEFT LoRA weights to {peft_path}")

    def load_lora_weights(self, path: str) -> None:
        """Load PEFT LoRA weights and camera token from file."""
        import os
        from peft import PeftModel

        peft_path = path.replace('.pt', '_peft')

        # Load PEFT adapter
        if os.path.exists(peft_path):
            if hasattr(self.vggt, 'load_adapter'):
                self.vggt.load_adapter(peft_path, adapter_name="default")
                if hasattr(self.vggt, 'set_adapter'):
                    self.vggt.set_adapter("default")
                print(f"Loaded and activated PEFT LoRA weights from {peft_path}")

        # Load camera token if present
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            if 'camera_token' in state_dict:
                aggregator = self._get_aggregator()

                if hasattr(aggregator, 'camera_token'):
                    aggregator.camera_token.data.copy_(state_dict['camera_token'])
                    print("Loaded camera token")


def create_teacher_student_pair(
    model_name: str = "facebook/vggt-1b",
    output_layers: Optional[List[int]] = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    device: str = "cuda",
) -> Tuple[VGGTTeacherModel, VGGTStudentModel]:
    """
    Create a teacher-student pair for VGGT Free-Geometry.

    Args:
        model_name: HuggingFace model name
        output_layers: Layer indices to extract features from
        lora_rank: LoRA rank for student
        lora_alpha: LoRA alpha for student
        device: Device to load models on

    Returns:
        Tuple of (teacher, student) models
    """
    teacher = VGGTTeacherModel(
        model_name=model_name,
        output_layers=output_layers,
    ).to(device)

    student = VGGTStudentModel(
        model_name=model_name,
        output_layers=output_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)

    return teacher, student
