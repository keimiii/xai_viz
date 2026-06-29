"""Attention extraction methods.

This module provides methods to extract and analyze attention patterns
from vision transformer models:

- CLS Attention: Direct attention from CLS token to patches
- Attention Rollout: Accumulated attention across layers
- Grad-CAM: See models/resnet50.py for CNN-based Grad-CAM baseline

Usage:
    >>> from ssl_attention.attention import extract_cls_attention, HeadFusion
    >>> attn = extract_cls_attention(
    ...     output.attention_weights,
    ...     num_registers=model.num_registers,
    ...     fusion=HeadFusion.MEAN,
    ... )
"""

from ssl_attention.attention.cls_attention import (
    HeadFusion,
    attention_to_heatmap,
    extract_cls_attention,
    extract_cls_attention_all_layers,
    extract_mean_attention,
    fuse_heads,
    get_per_head_attention,
)
from ssl_attention.attention.rollout import (
    attention_rollout,
    compare_rollout_depths,
    extract_cls_rollout,
)

__all__ = [
    # CLS Attention
    "HeadFusion",
    "fuse_heads",
    "extract_cls_attention",
    "extract_cls_attention_all_layers",
    "extract_mean_attention",
    "attention_to_heatmap",
    "get_per_head_attention",
    # Rollout
    "attention_rollout",
    "extract_cls_rollout",
    "compare_rollout_depths",
]
