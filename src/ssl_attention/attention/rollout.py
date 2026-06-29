"""Attention rollout for aggregating attention across layers.

Attention rollout computes the effective attention by multiplying
attention matrices across layers. This captures indirect attention
paths: if layer 1 attends A→B and layer 2 attends B→C, rollout
shows that A effectively attends to C.

Reference:
    Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
    https://arxiv.org/abs/2005.00928
"""

import torch
from torch import Tensor

from ssl_attention.attention.cls_attention import HeadFusion, fuse_heads
from ssl_attention.config import EPSILON


def attention_rollout(
    attention_weights: list[Tensor],
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
    discard_ratio: float = 0.0,
    start_layer: int = 0,
    end_layer: int | None = None,
) -> Tensor:
    """Compute attention rollout across transformer layers.

    Rollout recursively multiplies attention matrices:
        R_0 = I (identity)
        Ã_i = normalize(A_i)
        R_i = normalize(Ã_i + I) @ R_{i-1}

    Adding identity (I) accounts for the residual connection.
    Normalization ensures rows remain valid probability distributions.

    Args:
        attention_weights: List of per-layer attention tensors.
            Each tensor has shape (B, H, seq, seq).
        fusion: How to combine attention across heads.
        head_indices: Specific heads to use. None = all heads.
        discard_ratio: Fraction of lowest attention values to zero out.
            Can help reduce noise from low-confidence attention.
        start_layer: First layer to include (0-indexed).
        end_layer: Last layer to include (exclusive). None = all layers.

    Returns:
        Rolled-out attention of shape (B, seq, seq).
        R[i, j] represents how much position i attends to position j
        through all paths across layers.

    Example:
        >>> attns = [torch.randn(1, 12, 197, 197) for _ in range(12)]
        >>> rollout = attention_rollout(attns)
        >>> rollout.shape
        torch.Size([1, 197, 197])
    """
    if end_layer is None:
        end_layer = len(attention_weights)

    # Get batch size and sequence length
    batch_size, _, seq_len, _ = attention_weights[0].shape

    # Initialize rollout as identity matrix
    rollout = torch.eye(seq_len, device=attention_weights[0].device)
    rollout = rollout.unsqueeze(0).expand(batch_size, -1, -1).clone()  # (B, seq, seq)

    for layer_idx in range(start_layer, end_layer):
        # Fuse heads for this layer
        attn = fuse_heads(
            attention_weights[layer_idx],
            fusion=fusion,
            head_indices=head_indices,
        )  # (B, seq, seq)

        # Optionally discard low attention values
        if discard_ratio > 0:
            attn = _discard_low_attention(attn, discard_ratio)

        # Re-normalize after potential discarding
        attn = attn / (attn.sum(dim=-1, keepdim=True) + EPSILON)

        # Add residual connection (identity) and re-normalize
        # R_i = normalize(normalize(A_i) + I) @ R_{i-1}
        identity = torch.eye(seq_len, device=attn.device).unsqueeze(0)
        attn_with_residual = attn + identity

        # Re-normalize
        attn_with_residual = attn_with_residual / (
            attn_with_residual.sum(dim=-1, keepdim=True) + EPSILON
        )

        # Matrix multiply
        rollout = torch.bmm(attn_with_residual, rollout)

    return rollout


def _discard_low_attention(attention: Tensor, discard_ratio: float) -> Tensor:
    """Zero out the lowest attention values.

    Args:
        attention: Attention tensor of shape (B, seq, seq).
        discard_ratio: Fraction of values to zero out (per row).

    Returns:
        Attention with low values zeroed out.
    """
    batch_size, seq_len, _ = attention.shape

    # Flatten to find threshold
    flat = attention.view(batch_size, -1)
    num_to_discard = int(flat.shape[1] * discard_ratio)

    if num_to_discard == 0:
        return attention

    # Find threshold value
    threshold, _ = flat.kthvalue(num_to_discard, dim=1, keepdim=True)
    threshold = threshold.view(batch_size, 1, 1)

    # Zero out values below threshold
    attention = attention.clone()
    attention[attention < threshold] = 0

    return attention


def extract_cls_rollout(
    attention_weights: list[Tensor],
    num_registers: int = 0,
    fusion: HeadFusion = HeadFusion.MEAN,
    head_indices: list[int] | None = None,
    discard_ratio: float = 0.0,
    start_layer: int = 0,
    end_layer: int | None = None,
) -> Tensor:
    """Extract CLS token attention using rollout.

    Combines attention_rollout with CLS extraction, similar to
    extract_cls_attention but using rolled-out attention.

    Args:
        attention_weights: List of per-layer attention tensors.
        num_registers: Number of register tokens to skip.
        fusion: How to combine attention across heads.
        head_indices: Specific heads to use. None = all heads.
        discard_ratio: Fraction of lowest attention values to zero out.
        start_layer: First layer to include.
        end_layer: Last layer to include (exclusive). None = all.

    Returns:
        CLS-to-patch attention of shape (B, N) where N is num_patches.

    Example:
        >>> attns = [torch.randn(1, 12, 261, 261) for _ in range(12)]
        >>> cls_attn = extract_cls_rollout(attns, num_registers=4)
        >>> cls_attn.shape  # (1, 256)
    """
    rollout = attention_rollout(
        attention_weights,
        fusion=fusion,
        head_indices=head_indices,
        discard_ratio=discard_ratio,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    # Extract CLS row
    cls_to_all = rollout[:, 0, :]  # (B, seq)

    # Remove CLS and registers
    patch_start = 1 + num_registers
    cls_to_patches = cls_to_all[:, patch_start:]  # (B, num_patches)

    return cls_to_patches


def compare_rollout_depths(
    attention_weights: list[Tensor],
    num_registers: int = 0,
    depths: list[int] | None = None,
) -> dict[str, Tensor]:
    """Compare rollout at different layer depths.

    Useful for analyzing how attention evolves through the network.

    Args:
        attention_weights: List of per-layer attention tensors.
        num_registers: Number of register tokens to skip.
        depths: List of layer indices to compute rollout up to.
            Default: [3, 6, 9, 12] (for 12-layer models).

    Returns:
        Dictionary mapping depth name to CLS attention tensor.
    """
    num_layers = len(attention_weights)
    if depths is None:
        depths = [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]

    results = {}
    for depth in depths:
        if depth > num_layers:
            continue
        attn = extract_cls_rollout(
            attention_weights,
            num_registers=num_registers,
            end_layer=depth,
        )
        results[f"layer_{depth}"] = attn

    return results
