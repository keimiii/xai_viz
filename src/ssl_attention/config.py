"""Centralized configuration for SSL attention models.

This module provides a single source of truth for all model configurations
and default parameters used throughout the library.

Usage:
    from ssl_attention.config import MODELS, CACHE_MAX_MODELS

    # Access model config
    dinov2_config = MODELS["dinov2"]
    print(dinov2_config.patch_size)  # 14

    # Access constants
    print(CACHE_MAX_MODELS)  # 2

    # Access data paths
    from ssl_attention.config import DATASET_PATH, STYLE_MAPPING
    print(DATASET_PATH)  # Path to WikiChurches dataset

    # Access attention methods
    from ssl_attention.config import AttentionMethod, MODEL_METHODS
    print(MODEL_METHODS["dinov2"])  # [AttentionMethod.CLS, AttentionMethod.ROLLOUT]
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# =============================================================================
# Attention Methods
# =============================================================================


class AttentionMethod(str, Enum):
    """Available attention visualization methods.

    Different models support different methods:
    - CLS: Direct CLS token attention to patches (ViTs with CLS token)
    - ROLLOUT: Accumulated attention across layers (ViTs)
    - GRADCAM: Gradient-weighted class activation mapping (CNNs)
    - MEAN: Mean attention across all tokens (ViTs without CLS)
    """

    CLS = "cls"
    ROLLOUT = "rollout"
    GRADCAM = "gradcam"
    MEAN = "mean"


class FineTuningStrategy(str, Enum):
    """Supported fine-tuning strategies for Q2 analysis."""

    LINEAR_PROBE = "linear_probe"
    LORA = "lora"
    FULL = "full"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a vision transformer model.

    Attributes:
        model_id: HuggingFace model identifier (e.g., "facebook/dinov2-with-registers-base").
        patch_size: Size of each image patch in pixels (14 or 16).
        embed_dim: Dimension of token embeddings.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        num_registers: Number of register tokens (0 if none).
        has_cls_token: Whether the model has a CLS token in the sequence.
            SigLIP uses mean pooling instead of CLS.
    """

    model_id: str
    patch_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_registers: int = 0
    has_cls_token: bool = True


# =============================================================================
# Model Configurations
# =============================================================================
# All model specifications in one place for easy comparison and modification.

MODELS: dict[str, ModelConfig] = {
    "dinov2": ModelConfig(
        model_id="facebook/dinov2-with-registers-base",
        patch_size=14,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=4,
        has_cls_token=True,
    ),
    "dinov3": ModelConfig(
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=4,
        has_cls_token=True,
    ),
    "mae": ModelConfig(
        model_id="facebook/vit-mae-base",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=True,
    ),
    "clip": ModelConfig(
        model_id="openai/clip-vit-base-patch16",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=True,
    ),
    "siglip": ModelConfig(
        model_id="google/siglip-base-patch16-224",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=False,  # SigLIP uses mean pooling, no CLS token
    ),
    "siglip2": ModelConfig(
        model_id="google/siglip2-base-patch16-224",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=False,  # SigLIP2 uses mean pooling, no CLS token
    ),
    "resnet50": ModelConfig(
        model_id="torchvision",  # Flag for torchvision loading (not HuggingFace)
        patch_size=32,  # 224 / 7 = 32 (7x7 final feature grid)
        embed_dim=2048,  # Final layer channels
        num_layers=4,  # 4 ResNet stages (layer1-4)
        num_heads=1,  # Dummy value (CNN has no attention heads)
        num_registers=0,
        has_cls_token=False,  # Uses global average pooling, not CLS
    ),
}


# =============================================================================
# Model Aliases
# =============================================================================
# Alternative names that map to canonical model names.

MODEL_ALIASES: dict[str, str] = {
    "dino": "dinov2",
    "dinov2-reg": "dinov2",
    "vit-mae": "mae",
    "openai-clip": "clip",
}


# =============================================================================
# Attention Method Availability
# =============================================================================
# Maps each model to its supported attention methods.
# - ViTs with CLS token: CLS (direct) + Rollout (accumulated)
# - ViTs without CLS (SigLIP): Mean attention only
# - CNNs (ResNet): Grad-CAM only

MODEL_METHODS: dict[str, list[AttentionMethod]] = {
    "dinov2": [AttentionMethod.CLS, AttentionMethod.ROLLOUT],
    "dinov3": [AttentionMethod.CLS, AttentionMethod.ROLLOUT],
    "mae": [AttentionMethod.CLS, AttentionMethod.ROLLOUT],
    "clip": [AttentionMethod.CLS, AttentionMethod.ROLLOUT],
    "siglip": [AttentionMethod.MEAN],
    "siglip2": [AttentionMethod.MEAN],
    "resnet50": [AttentionMethod.GRADCAM],
}


# Models that support style-task fine-tuning in this project.
# ResNet-50 uses a separate Grad-CAM pipeline and is excluded.
FINETUNE_MODELS: set[str] = {
    "dinov2",
    "dinov3",
    "mae",
    "clip",
    "siglip",
    "siglip2",
}

FINETUNE_STRATEGIES: tuple[FineTuningStrategy, ...] = (
    FineTuningStrategy.LINEAR_PROBE,
    FineTuningStrategy.LORA,
    FineTuningStrategy.FULL,
)

# Default method for each model (first in list for ViTs, only option for others)
DEFAULT_METHOD: dict[str, AttentionMethod] = {
    "dinov2": AttentionMethod.CLS,
    "dinov3": AttentionMethod.CLS,
    "mae": AttentionMethod.CLS,
    "clip": AttentionMethod.CLS,
    "siglip": AttentionMethod.MEAN,
    "siglip2": AttentionMethod.MEAN,
    "resnet50": AttentionMethod.GRADCAM,
}


# =============================================================================
# Cache Settings
# =============================================================================

# Maximum number of models to keep in memory via LRU cache.
# Set to 2 to avoid GPU memory exhaustion when switching models.
CACHE_MAX_MODELS: int = 2


# =============================================================================
# Attention Module Defaults
# =============================================================================

# Default image size for attention heatmap generation.
# Standard ViT input size is 224x224.
DEFAULT_IMAGE_SIZE: int = 224

# Small epsilon added before KL probability normalization so zero-heavy maps stay finite.
EPSILON: float = 1e-8

# Interpolation mode for upsampling attention maps to image size.
INTERPOLATION_MODE: str = "bilinear"


# =============================================================================
# Data Paths
# =============================================================================

# Root path to the WikiChurches dataset.
# Can be overridden via SSL_DATASET_PATH environment variable.
# Path: config.py -> ssl_attention -> src -> ssl_wikichurches -> dataset
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_DATASET_PATH = _PROJECT_ROOT / "dataset"
DATASET_PATH: Path = Path(os.environ.get("SSL_DATASET_PATH", _DEFAULT_DATASET_PATH))

# Derived paths
IMAGES_PATH: Path = DATASET_PATH / "images"
ANNOTATIONS_PATH: Path = DATASET_PATH / "building_parts.json"
CHURCHES_PATH: Path = DATASET_PATH / "churches.json"

# Cache directory for feature/attention storage
CACHE_PATH: Path = _PROJECT_ROOT / "outputs" / "cache"


# =============================================================================
# Style Classification
# =============================================================================

# Wikidata Q-IDs for the 4 main architectural styles in WikiChurches.
# These 4 styles represent the majority of annotated images (142 of 9,502).
# Note: Q46261=Romanesque, Q176483=Gothic (verified from style_names.txt)
STYLE_MAPPING: dict[str, int] = {
    "Q46261": 0,   # Romanesque (54 images in annotated subset)
    "Q176483": 1,  # Gothic (49 images in annotated subset)
    "Q236122": 2,  # Renaissance (22 images in annotated subset)
    "Q840829": 3,  # Baroque (17 images in annotated subset)
}

# Human-readable style names (indexed by STYLE_MAPPING values)
STYLE_NAMES: tuple[str, ...] = ("Romanesque", "Gothic", "Renaissance", "Baroque")

# Number of architectural styles for classification
NUM_STYLES: int = len(STYLE_MAPPING)
