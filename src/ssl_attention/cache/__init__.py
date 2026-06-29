"""Caching utilities for extracted features and attention maps.

This package provides HDF5-based storage for:
- Feature tokens (CLS + patch tokens) for linear probe evaluation
- Attention maps for visualization and IoU analysis

Using cached features avoids redundant model inference when running
multiple analyses on the same images.

Example:
    from ssl_attention.cache import FeatureCache, AttentionCache, CacheKey
    from ssl_attention.config import CACHE_PATH

    # Initialize caches
    feature_cache = FeatureCache(CACHE_PATH / "features.h5")
    attn_cache = AttentionCache(CACHE_PATH / "attention.h5")

    # Check if cached
    if not feature_cache.exists("dinov2", "layer11", "Q123_wd0.jpg"):
        # Run inference and store
        feature_cache.store("dinov2", "layer11", "Q123_wd0.jpg", cls, patches)

    # Load cached features
    cls, patches = feature_cache.load("dinov2", "layer11", "Q123_wd0.jpg")
"""

from ssl_attention.cache.manager import AttentionCache, CacheKey, FeatureCache

__all__ = [
    "CacheKey",
    "FeatureCache",
    "AttentionCache",
]
