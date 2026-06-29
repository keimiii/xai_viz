"""CLS token attention extraction and visualization.

This module provides the primary attention extraction method: extracting
how much the CLS token attends to each image patch. This gives a spatial
attention map showing which parts of the image the model finds important.

Key functions:
- `extract_cls_attention()`: Get CLS-to-patch attention from attention weights
- `fuse_heads()`: Combine attention across multiple heads
- `attention_to_heatmap()`: Upsample attention to image size
"""

import math
from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from ssl_attention.config import DEFAULT_IMAGE_SIZE, EPSILON, INTERPOLATION_MODE


class HeadFusion(Enum):
    """Strategy for combining attention across heads.

    - MEAN: Average attention across all heads (democratic, most common)
    - MAX: Maximum attention from any head (if any head attends, it counts)
    - MIN: Minimum attention across heads (only where all heads agree)
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"


def fuse_heads(
    attention: Tensor,
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
) -> Tensor:
    """Fuse attention across heads.

    Args:
        attention: Attention tensor of shape (B, H, seq, seq) or (B, H, N).
        fusion: Strategy for combining heads.
        head_indices: If provided, only use these heads. If None, use all.

    Returns:
        Fused attention of shape (B, seq, seq) or (B, N).

    Example:
        >>> attn = torch.randn(2, 12, 197, 197)  # B=2, H=12
        >>> fused = fuse_heads(attn, HeadFusion.MEAN)
        >>> fused.shape
        torch.Size([2, 197, 197])
    """
    # Select specific heads if requested
    if head_indices is not None:
        attention = attention[:, head_indices, ...]

    # Fuse across head dimension (dim=1)
    if fusion == HeadFusion.MEAN:
        return attention.mean(dim=1)
    elif fusion == HeadFusion.MAX:
        return attention.max(dim=1).values
    elif fusion == HeadFusion.MIN:
        return attention.min(dim=1).values
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion}")


def extract_cls_attention(
    attention_weights: list[Tensor],
    layer: int = -1,
    num_registers: int = 0,
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
) -> Tensor:
    """Extract CLS token's attention to image patches.

    The CLS token aggregates information from patches. Its attention weights
    show which patches it attends to, providing a spatial importance map.

    Args:
        attention_weights: List of per-layer attention tensors.
            Each tensor has shape (B, H, seq, seq) where:
            - B = batch size
            - H = number of heads
            - seq = sequence length (CLS + registers + patches)
        layer: Which layer to extract from. Default -1 (last layer).
        num_registers: Number of register tokens to skip (e.g., 4 for DINOv2).
        fusion: How to combine attention across heads.
        head_indices: Specific heads to use. None = all heads.

    Returns:
        CLS-to-patch attention of shape (B, N) where N is num_patches.
        Values sum to approximately 1 (they're attention weights).

    Example:
        >>> # For DINOv2: seq = 1 + 4 + 256 = 261
        >>> attns = [torch.randn(1, 12, 261, 261) for _ in range(12)]
        >>> cls_attn = extract_cls_attention(attns, layer=-1, num_registers=4)
        >>> cls_attn.shape
        torch.Size([1, 256])
    """
    # Get attention from specified layer
    attn = attention_weights[layer]  # (B, H, seq, seq)

    # Fuse heads first
    attn_fused = fuse_heads(attn, fusion=fusion, head_indices=head_indices)  # (B, seq, seq)

    # Extract CLS row (position 0 attends to all positions)
    cls_to_all = attn_fused[:, 0, :]  # (B, seq)

    # Remove CLS token and registers, keep only patches
    # Sequence: [CLS, reg1, ..., regN, patch1, ..., patchM]
    patch_start = 1 + num_registers
    cls_to_patches = cls_to_all[:, patch_start:]  # (B, num_patches)

    return cls_to_patches


def extract_cls_attention_all_layers(
    attention_weights: list[Tensor],
    num_registers: int = 0,
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
) -> Tensor:
    """Extract CLS attention from all layers.

    Args:
        attention_weights: List of per-layer attention tensors.
        num_registers: Number of register tokens to skip.
        fusion: How to combine attention across heads.
        head_indices: Specific heads to use. None = all heads.

    Returns:
        CLS attention of shape (B, L, N) where L is num_layers, N is num_patches.
    """
    all_layers = []
    for layer_idx in range(len(attention_weights)):
        layer_attn = extract_cls_attention(
            attention_weights,
            layer=layer_idx,
            num_registers=num_registers,
            fusion=fusion,
            head_indices=head_indices,
        )
        all_layers.append(layer_attn)

    return torch.stack(all_layers, dim=1)  # (B, L, N)


def attention_to_heatmap(
    attention: Tensor,
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
) -> Tensor:
    """Upsample patch attention to full image size.

    Reshapes 1D patch attention to 2D spatial grid, then upsamples
    to match the original image size for overlay visualization.

    Args:
        attention: CLS-to-patch attention of shape (B, N) or (N,).
        image_size: Target image size (assumes square).
        normalize: If True, normalize to [0, 1] range.

    Returns:
        Heatmap of shape (B, H, W) or (H, W).

    Example:
        >>> attn = torch.randn(1, 256)  # 16x16 patches
        >>> heatmap = attention_to_heatmap(attn, image_size=224)
        >>> heatmap.shape
        torch.Size([1, 224, 224])
    """
    # Handle unbatched input
    squeeze = attention.dim() == 1
    if squeeze:
        attention = attention.unsqueeze(0)

    batch_size = attention.shape[0]
    num_patches = attention.shape[1]
    patches_per_side = int(math.sqrt(num_patches))

    # Validate square patch grid
    if patches_per_side * patches_per_side != num_patches:
        raise ValueError(
            f"attention_to_heatmap requires square patch grid, got {num_patches} patches "
            f"(sqrt={math.sqrt(num_patches):.2f}). Check image preprocessing - "
            f"ensure fixed 224x224 resolution for consistent 14x14 or 16x16 patch grids."
        )

    # Reshape to 2D grid
    attn_2d = attention.view(batch_size, patches_per_side, patches_per_side)

    # Upsample to image size
    attn_upsampled = F.interpolate(
        attn_2d.unsqueeze(1),  # Add channel dim: (B, 1, H, W)
        size=(image_size, image_size),
        mode=INTERPOLATION_MODE,
        align_corners=False,
    ).squeeze(1)  # Remove channel dim: (B, H, W)

    # Normalize to [0, 1]
    if normalize:
        # Per-sample normalization
        flat = attn_upsampled.view(batch_size, -1)
        min_val = flat.min(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        max_val = flat.max(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        attn_upsampled = (attn_upsampled - min_val) / (max_val - min_val + EPSILON)

    if squeeze:
        attn_upsampled = attn_upsampled.squeeze(0)

    return attn_upsampled


def get_per_head_attention(
    attention_weights: list[Tensor],
    layer: int = -1,
    num_registers: int = 0,
) -> Tensor:
    """Get CLS attention for each head separately.

    Useful for analyzing what different heads specialize in.

    Args:
        attention_weights: List of per-layer attention tensors.
        layer: Which layer to extract from.
        num_registers: Number of register tokens to skip.

    Returns:
        Attention of shape (B, H, N) where H is num_heads, N is num_patches.
    """
    attn = attention_weights[layer]  # (B, H, seq, seq)

    # Extract CLS row for each head
    cls_to_all = attn[:, :, 0, :]  # (B, H, seq)

    # Remove CLS and registers
    patch_start = 1 + num_registers
    cls_to_patches = cls_to_all[:, :, patch_start:]  # (B, H, num_patches)

    return cls_to_patches


def extract_mean_attention(
    attention_weights: list[Tensor],
    layer: int = -1,
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
) -> Tensor:
    """Extract mean attention across all patches.

    For models without a CLS token (like SigLIP), this provides an
    alternative attention map by averaging how much each patch attends
    to other patches.

    Args:
        attention_weights: List of per-layer attention tensors.
            Each tensor has shape (B, H, seq, seq).
        layer: Which layer to extract from. Default -1 (last layer).
        fusion: How to combine attention across heads.
        head_indices: Specific heads to use. None = all heads.

    Returns:
        Mean attention of shape (B, N) where N is num_patches.
        Higher values indicate patches that receive more attention overall.

    Example:
        >>> # For SigLIP: seq = 196 (all patches, no CLS)
        >>> attns = [torch.randn(1, 12, 196, 196) for _ in range(12)]
        >>> mean_attn = extract_mean_attention(attns, layer=-1)
        >>> mean_attn.shape
        torch.Size([1, 196])
    """
    # Get attention from specified layer
    attn = attention_weights[layer]  # (B, H, seq, seq)

    # Fuse heads
    attn_fused = fuse_heads(attn, fusion=fusion, head_indices=head_indices)  # (B, seq, seq)

    # Mean across rows: how much does each position get attended to?
    # This gives us a "saliency" score for each position
    mean_received = attn_fused.mean(dim=1)  # (B, seq)

    return mean_received
