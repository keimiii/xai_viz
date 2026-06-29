"""Vision backbone model wrappers.

This module provides unified wrappers for SSL vision models:
- DINOv2: Self-distillation with registers
- DINOv3: RoPE-based DINO with larger training data
- MAE: Masked autoencoder
- CLIP: Contrastive language-image pretraining
- SigLIP: Sigmoid loss for language-image pretraining

All models implement the VisionBackbone protocol and return
standardized ModelOutput with cls_token, patch_tokens, and
attention_weights.

Usage:
    >>> from ssl_attention.models import get_model
    >>> model = get_model("dinov2")
    >>> output = model(model.preprocess([image]))
"""

from ssl_attention.models.protocols import ModelOutput, VisionBackbone
from ssl_attention.models.registry import (
    clear_cache,
    create_model,
    get_model,
    list_models,
    model_info,
)

__all__ = [
    # Protocols
    "ModelOutput",
    "VisionBackbone",
    # Registry
    "get_model",
    "create_model",
    "list_models",
    "model_info",
    "clear_cache",
]
