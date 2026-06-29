"""Backend configuration."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (app/ directory is at project_root/app/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# SSL Attention source
SSL_ATTENTION_SRC = PROJECT_ROOT / "src"

# Dataset paths
DATASET_PATH = Path(os.environ.get("SSL_DATASET_PATH", PROJECT_ROOT / "dataset"))
IMAGES_PATH = DATASET_PATH / "images"
ANNOTATIONS_PATH = DATASET_PATH / "building_parts.json"

# Cache paths (pre-computed data)
CACHE_PATH = Path(os.environ.get("SSL_CACHE_PATH", PROJECT_ROOT / "outputs" / "cache"))
ATTENTION_CACHE_PATH = CACHE_PATH / "attention_viz.h5"
FEATURE_CACHE_PATH = CACHE_PATH / "features.h5"
HEATMAPS_PATH = CACHE_PATH / "heatmaps"
METRICS_DB_PATH = CACHE_PATH / "metrics.db"
METRICS_SUMMARY_PATH = CACHE_PATH / "metrics_summary.json"
LEGACY_Q2_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "q2_metrics_analysis.json"

# Available models (must match ssl_attention.config.MODELS)
AVAILABLE_MODELS = ["dinov2", "dinov3", "mae", "clip", "siglip", "siglip2", "resnet50"]

# Number of transformer layers
NUM_LAYERS = 12

# Style names
STYLE_NAMES = ["Romanesque", "Gothic", "Renaissance", "Baroque"]

# Debug mode — controls whether error responses include internal details
DEBUG = os.environ.get("SSL_DEBUG", "0").lower() in ("1", "true", "yes")

# API settings
API_PREFIX = "/api"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "http://localhost:5174",  # Vite fallback when 5173 is in use
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

# Image settings
THUMBNAIL_SIZE = (128, 128)
STANDARD_IMAGE_SIZE = (224, 224)

# Model name resolution and attention method configuration
# Re-exported for use by routers (F401 false positive - these are used externally)
from ssl_attention.config import DEFAULT_METHOD as DEFAULT_METHOD
from ssl_attention.config import FINETUNE_STRATEGIES as FINETUNE_STRATEGIES
from ssl_attention.config import MODEL_ALIASES
from ssl_attention.config import MODEL_METHODS as MODEL_METHODS
from ssl_attention.config import MODELS as MODELS
from ssl_attention.config import AttentionMethod as AttentionMethod
from ssl_attention.evaluation.fine_tuning_artifacts import resolve_active_artifact_path

_FINETUNED_SUFFIX = "_finetuned"
_FINETUNED_MARKER = "_finetuned_"
VALID_FINETUNE_STRATEGIES = {s.value for s in FINETUNE_STRATEGIES}
PER_HEAD_METHODS = {AttentionMethod.CLS.value, AttentionMethod.MEAN.value}
MODEL_NUM_HEADS = {
    model_name: (0 if model_name == "resnet50" else model_config.num_heads)
    for model_name, model_config in MODELS.items()
}


def get_current_q2_results_path() -> Path:
    """Resolve the current Q2 artifact path from the active experiment pointer."""
    return resolve_active_artifact_path("q2_metrics_path", LEGACY_Q2_RESULTS_PATH)


def split_model_name(model: str) -> tuple[str, bool, str | None]:
    """Split model identifier into base model, finetuned flag, and strategy."""
    if _FINETUNED_MARKER in model:
        base, strategy = model.split(_FINETUNED_MARKER, maxsplit=1)
        if strategy:
            return base, True, strategy
        return base, True, None
    if model.endswith(_FINETUNED_SUFFIX):
        base = model[: -len(_FINETUNED_SUFFIX)]
        return base, True, None
    return model, False, None


def resolve_model_name(model: str) -> str:
    """Resolve model alias to canonical name.

    Args:
        model: Model name (may be an alias like 'siglip2').

    Returns:
        Canonical model name (e.g., 'siglip').
    """
    base_model, is_finetuned, strategy = split_model_name(model)
    resolved_base = MODEL_ALIASES.get(base_model, base_model)
    if not is_finetuned:
        return resolved_base
    if strategy:
        return f"{resolved_base}{_FINETUNED_MARKER}{strategy}"
    return f"{resolved_base}{_FINETUNED_SUFFIX}"


# Reverse mapping: canonical DB name → frontend display name
# e.g., 'siglip' → 'siglip2' (only differs for aliased models)
_CANONICAL_TO_DISPLAY = {resolve_model_name(m): m for m in AVAILABLE_MODELS}


def display_model_name(canonical: str) -> str:
    """Map canonical DB model name back to frontend display name.

    Args:
        canonical: Canonical model name from DB (e.g., 'siglip').

    Returns:
        Display name matching AVAILABLE_MODELS (e.g., 'siglip2').
    """
    return _CANONICAL_TO_DISPLAY.get(canonical, canonical)


def get_model_num_layers(model: str) -> int:
    """Get number of layers for a model.

    Args:
        model: Canonical model name.

    Returns:
        Number of layers for the model.
    """
    resolved = resolve_model_name(model)
    resolved_base, _, _ = split_model_name(resolved)
    resolved = resolved_base
    if resolved in MODELS:
        return MODELS[resolved].num_layers
    return NUM_LAYERS  # Fallback to default
