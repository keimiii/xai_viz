"""Shared HTTP validation helpers for API endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from app.backend.config import (
    AVAILABLE_MODELS,
    DEFAULT_METHOD,
    MODEL_METHODS,
    VALID_FINETUNE_STRATEGIES,
    AttentionMethod,
    get_model_num_layers,
    resolve_model_name,
    split_model_name,
)
from ssl_attention.config import FINETUNE_MODELS

_FINETUNED_SUFFIX = "_finetuned"
RankingMode = Literal["default_method", "best_available"]


def split_model_variant(model: str) -> tuple[str, bool, str | None]:
    """Split and resolve a model identifier into base model + finetuned flag + strategy.

    Accepts identifiers like:
    - "dinov2" -> ("dinov2", False, None)
    - "dinov2_finetuned" -> ("dinov2", True, None)
    - "dinov2_finetuned_lora" -> ("dinov2", True, "lora")
    """
    base, is_finetuned, strategy = split_model_name(model)
    return resolve_model_name(base), is_finetuned, strategy


def resolve_default_method(model: str) -> str:
    """Resolve the default attention method for a model.

    Args:
        model: Model name (may be alias like 'siglip2').

    Returns:
        Default method string (e.g., 'cls', 'mean', 'gradcam').
    """
    base_model, _, _ = split_model_variant(model)
    method: str = DEFAULT_METHOD.get(base_model, AttentionMethod.CLS).value
    return method


def validate_method(model: str, method: str | None) -> str:
    """Validate and resolve attention method for a model.

    Args:
        model: Model name (may be alias like 'siglip2').
        method: Requested method, or None for default.

    Returns:
        Valid method string.

    Raises:
        HTTPException: If method not available for model.
    """
    base_model, _, _ = split_model_variant(model)
    available = MODEL_METHODS.get(base_model, [])

    if method is None:
        default: AttentionMethod = DEFAULT_METHOD.get(base_model, AttentionMethod.CLS)
        return str(default.value)

    # Validate requested method
    try:
        method_enum = AttentionMethod(method)
    except ValueError:
        valid_methods = [m.value for m in AttentionMethod]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method: '{method}'. Valid methods: {valid_methods}",
        ) from None

    if method_enum not in available:
        available_str = [m.value for m in available]
        raise HTTPException(
            status_code=400,
            detail=f"Method '{method}' not available for '{model}'. Available: {available_str}",
        )

    return method


def validate_attention_method(method: str | None) -> str | None:
    """Validate an attention method string without binding it to a model."""
    if method is None:
        return None

    try:
        return AttentionMethod(method).value
    except ValueError:
        valid_methods = [m.value for m in AttentionMethod]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method: '{method}'. Valid methods: {valid_methods}",
        ) from None


def resolve_ranking_mode_request(method: str | None, ranking_mode: RankingMode | None) -> RankingMode | None:
    """Resolve the requested ranking mode, enforcing mutual exclusivity with method."""
    if method is not None and ranking_mode is not None:
        raise HTTPException(
            status_code=400,
            detail="Query parameters 'method' and 'ranking_mode' cannot be combined.",
        )

    if method is not None:
        return None

    return ranking_mode or "default_method"


def model_supports_method(model: str, method: str) -> bool:
    """Return whether the requested model supports the given attention method."""
    base_model, _, _ = split_model_variant(model)
    available = MODEL_METHODS.get(base_model, [])
    return AttentionMethod(method) in available


def validate_model(model: str) -> str:
    """Validate model name and return resolved canonical name.

    Args:
        model: Model name (may be alias like 'siglip2').

    Returns:
        Resolved canonical model name (e.g., 'siglip' for 'siglip2').

    Raises:
        HTTPException: If model is not available.
    """
    base_model, is_finetuned, strategy = split_model_variant(model)

    if base_model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )

    if is_finetuned and base_model not in FINETUNE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Fine-tuned variant not supported for model: {base_model}. "
                f"Fine-tunable models: {sorted(FINETUNE_MODELS)}"
            ),
        )

    if is_finetuned and strategy is not None and strategy not in VALID_FINETUNE_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid fine-tuning strategy '{strategy}'. "
                f"Available: {sorted(VALID_FINETUNE_STRATEGIES)}"
            ),
        )

    if is_finetuned and strategy:
        return f"{base_model}_finetuned_{strategy}"
    return f"{base_model}{_FINETUNED_SUFFIX}" if is_finetuned else base_model


def validate_layer_for_model(layer: int, model: str) -> str:
    """Validate layer is within bounds and return layer key.

    Args:
        layer: Layer index (0-based).
        model: Model name (may be alias).

    Returns:
        Layer key string (e.g., 'layer5').

    Raises:
        HTTPException: If layer is out of bounds for the model.
    """
    base_model, _, _ = split_model_variant(model)
    num_layers = get_model_num_layers(base_model)
    if not 0 <= layer < num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer: {layer}. Model '{model}' has {num_layers} layers (0-{num_layers - 1}).",
        )
    return f"layer{layer}"
