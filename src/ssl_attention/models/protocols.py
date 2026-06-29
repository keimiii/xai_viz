"""Core protocols and data structures for vision model outputs."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from PIL import Image
from torch import Tensor


@dataclass
class ModelOutput:
    """Standardized output from vision backbone models.

    This structure normalizes outputs across different SSL architectures,
    making downstream attention analysis model-agnostic.

    Attributes:
        cls_token: CLS token embedding, shape (B, D) where D is embed_dim.
        patch_tokens: Patch token embeddings, shape (B, N, D) where N is num_patches.
            Note: This excludes CLS token and any register tokens.
            For CNN models (like ResNet), this may be None.
        attention_weights: Per-layer attention weights.
            For ViTs: List of L tensors, each with shape (B, H, seq_len, seq_len)
            where H is num_heads and seq_len includes CLS + registers + patches.
            For CNNs: List of L tensors, each with shape (B, H, W) - Grad-CAM heatmaps.
        hidden_states: Per-layer hidden states (optional).
            List of L tensors, each with shape (B, seq_len, D).
            Only populated when output_hidden_states=True in forward().
            Index i contains the output after transformer layer i (0-indexed).
            The embedding layer output is not included.
    """

    cls_token: Tensor
    patch_tokens: Tensor | None
    attention_weights: list[Tensor]
    hidden_states: list[Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate tensor shapes are compatible."""
        batch_cls = self.cls_token.shape[0]

        # Validate patch_tokens if present (ViT models)
        if self.patch_tokens is not None:
            batch_patch = self.patch_tokens.shape[0]
            if batch_cls != batch_patch:
                raise ValueError(
                    f"Batch size mismatch: cls_token has {batch_cls}, "
                    f"patch_tokens has {batch_patch}"
                )

        if self.attention_weights:
            batch_attn = self.attention_weights[0].shape[0]
            if batch_attn != batch_cls:
                raise ValueError(
                    f"Batch size mismatch: attention has {batch_attn}, "
                    f"cls_token has {batch_cls}"
                )

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.cls_token.shape[0]

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.cls_token.shape[1]

    @property
    def num_patches(self) -> int | None:
        """Return the number of patches (excluding CLS and registers).

        Returns None for CNN models that don't have patch tokens.
        """
        return self.patch_tokens.shape[1] if self.patch_tokens is not None else None

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers with attention."""
        return len(self.attention_weights)


@runtime_checkable
class VisionBackbone(Protocol):
    """Protocol defining the interface for vision backbone models.

    All SSL model wrappers must implement this protocol to ensure
    consistent behavior across DINOv2, DINOv3, MAE, CLIP, and SigLIP.

    The protocol is runtime_checkable, allowing isinstance() checks.
    """

    # Model metadata
    model_name: str
    """Identifier for the model (e.g., 'dinov2', 'clip')."""

    patch_size: int
    """Size of each image patch in pixels (typically 14 or 16)."""

    embed_dim: int
    """Dimension of token embeddings."""

    num_layers: int
    """Number of transformer layers."""

    num_heads: int
    """Number of attention heads per layer."""

    num_registers: int
    """Number of register tokens (0 for models without registers)."""

    device: torch.device
    """Device the model is loaded on."""

    def forward(self, images: Tensor) -> ModelOutput:
        """Process preprocessed images through the model.

        Args:
            images: Batch of preprocessed images, shape (B, C, H, W).
                Should be the output of preprocess().

        Returns:
            ModelOutput with cls_token, patch_tokens, and attention_weights.
        """
        ...

    def preprocess(self, images: list[Image.Image]) -> Tensor:
        """Preprocess PIL images for model input.

        Applies model-specific normalization and resizing.

        Args:
            images: List of PIL Images to preprocess.

        Returns:
            Tensor of shape (B, C, H, W) ready for forward().
        """
        ...


# Type alias for attention weight tensors
AttentionWeights = list[Tensor]
"""List of attention tensors, one per layer, each (B, H, seq, seq)."""
