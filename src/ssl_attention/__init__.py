"""SSL Attention Analysis Library.

Compare self-supervised learning model attention patterns against
expert annotations on the WikiChurches dataset.

Key modules:
- models: Vision backbone wrappers (DINOv2, DINOv3, MAE, CLIP, SigLIP)
- attention: Attention extraction methods (CLS attention, rollout, Grad-CAM)
- utils: Device management and utilities
- config: Centralized configuration for all models and defaults
"""

__version__ = "0.1.0"

# Re-export commonly used items for convenience
from ssl_attention.config import MODELS, ModelConfig
from ssl_attention.models.protocols import ModelOutput, VisionBackbone
from ssl_attention.models.registry import (
    clear_cache,
    create_model,
    get_model,
    list_models,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "MODELS",
    "ModelConfig",
    # Protocols
    "ModelOutput",
    "VisionBackbone",
    # Registry
    "get_model",
    "create_model",
    "list_models",
    "clear_cache",
]
