"""Model registry for lazy loading and caching.

The registry provides:
- `get_model(name)`: Cached loading (LRU cache, max 2 models in memory)
- `create_model(name)`: Fresh instance every time
- `list_models()`: Available model names
- `clear_cache()`: Free memory by clearing cached models
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from ssl_attention.config import CACHE_MAX_MODELS, MODEL_ALIASES
from ssl_attention.utils.device import clear_memory

if TYPE_CHECKING:
    from ssl_attention.models.base import BaseVisionModel

# Model name -> (module path, class name)
# Using strings for lazy imports to avoid loading all models at startup
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "dinov2": ("ssl_attention.models.dinov2", "DINOv2"),
    "dinov3": ("ssl_attention.models.dinov3", "DINOv3"),
    "mae": ("ssl_attention.models.mae", "MAE"),
    "clip": ("ssl_attention.models.clip_model", "CLIP"),
    "siglip": ("ssl_attention.models.siglip", "SigLIP"),
    "siglip2": ("ssl_attention.models.siglip2", "SigLIP2"),
    "resnet50": ("ssl_attention.models.resnet50", "ResNet50"),
}

# Use aliases from central config
_ALIASES = MODEL_ALIASES


def _resolve_name(name: str) -> str:
    """Resolve model name or alias to canonical name."""
    name_lower = name.lower()
    return _ALIASES.get(name_lower, name_lower)


def _import_model_class(name: str) -> type["BaseVisionModel"]:
    """Dynamically import a model class.

    Args:
        name: Canonical model name.

    Returns:
        The model class.

    Raises:
        ValueError: If model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    module_path, class_name = _MODEL_REGISTRY[name]

    # Dynamic import
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def create_model(
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "BaseVisionModel":
    """Create a fresh model instance.

    Unlike get_model(), this always creates a new instance.
    Use this when you need multiple copies or want to control lifetime.

    Args:
        name: Model name (e.g., 'dinov2', 'clip') or alias.
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Fresh model instance.

    Example:
        >>> model = create_model("dinov2")
        >>> model2 = create_model("dinov2")  # Different instance
        >>> model is model2
        False
    """
    canonical = _resolve_name(name)
    model_class = _import_model_class(canonical)
    return model_class(device=device, dtype=dtype)


# Cache key includes device and dtype for proper caching
@lru_cache(maxsize=CACHE_MAX_MODELS)
def _cached_model(
    name: str,
    device_str: str,
    dtype_str: str,
) -> "BaseVisionModel":
    """Internal cached model loader.

    Uses string representations of device/dtype for hashability.
    """
    device = torch.device(device_str)
    # Parse dtype from string
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    return create_model(name, device=device, dtype=dtype)


def get_model(
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "BaseVisionModel":
    """Get a cached model instance.

    Uses LRU cache with maxsize=2, so at most 2 models are kept in memory.
    This is important for GPU memory management.

    Args:
        name: Model name (e.g., 'dinov2', 'clip') or alias.
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Cached model instance.

    Example:
        >>> model = get_model("dinov2")
        >>> model2 = get_model("dinov2")  # Same instance
        >>> model is model2
        True
    """
    from ssl_attention.utils.device import get_device, get_dtype_for_device

    # Resolve defaults for caching
    actual_device = device or get_device()
    actual_dtype = dtype or get_dtype_for_device(actual_device)

    return _cached_model(
        _resolve_name(name),
        str(actual_device),
        str(actual_dtype),
    )


def list_models() -> list[str]:
    """List all available model names.

    Returns:
        Sorted list of canonical model names.
    """
    return sorted(_MODEL_REGISTRY.keys())


def list_aliases() -> dict[str, str]:
    """List all model aliases.

    Returns:
        Dictionary mapping alias -> canonical name.
    """
    return dict(_ALIASES)


def clear_cache() -> None:
    """Clear the model cache and free memory.

    Call this when switching between many models or when memory is tight.
    """
    _cached_model.cache_clear()
    clear_memory()


def model_info(name: str) -> dict[str, str | int]:
    """Get metadata about a model without loading it.

    Args:
        name: Model name or alias.

    Returns:
        Dictionary with model metadata.
    """
    canonical = _resolve_name(name)
    model_class = _import_model_class(canonical)

    return {
        "name": model_class.model_name,
        "model_id": model_class.model_id,
        "patch_size": model_class.patch_size,
        "embed_dim": model_class.embed_dim,
        "num_layers": model_class.num_layers,
        "num_heads": model_class.num_heads,
        "num_registers": model_class.num_registers,
    }
