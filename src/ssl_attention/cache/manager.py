"""Cache management for extracted features and attention maps.

This module provides HDF5-based caching to avoid redundant model inference.
Cached items are organized by model, layer, and image_id.

Example:
    from ssl_attention.cache import FeatureCache, AttentionCache
    from ssl_attention.config import CACHE_PATH

    # Cache extracted features
    feature_cache = FeatureCache(CACHE_PATH / "features.h5")
    feature_cache.store("dinov2", "layer11", "Q123_wd0.jpg", cls_token, patch_tokens)
    cls, patches = feature_cache.load("dinov2", "layer11", "Q123_wd0.jpg")

    # Cache attention maps
    attn_cache = AttentionCache(CACHE_PATH / "attention.h5")
    attn_cache.store("dinov2", "layer11", "Q123_wd0.jpg", attention_map)
    attn = attn_cache.load("dinov2", "layer11", "Q123_wd0.jpg")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class CacheKey:
    """Unique identifier for a cached item.

    Attributes:
        model: Model name (e.g., "dinov2", "clip").
        layer: Layer identifier (e.g., "layer11", "last").
        image_id: Image filename (e.g., "Q18785543_wd0.jpg").
        variant: Optional variant suffix (e.g., "augmented", "cropped").
    """

    model: str
    layer: str
    image_id: str
    variant: str = "default"

    def to_path(self) -> str:
        """Convert to HDF5 group path."""
        return f"{self.model}/{self.layer}/{self.variant}/{self.image_id}"


class FeatureCache:
    """HDF5 cache for CLS and patch token features.

    Stores features in float32 format with gzip compression.

    Attributes:
        path: Path to the HDF5 cache file.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize the feature cache.

        Args:
            path: Path to the HDF5 file (created if doesn't exist).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _get_group_path(self, key: CacheKey) -> str:
        """Get HDF5 group path for a cache key."""
        return key.to_path()

    def exists(
        self,
        model: str,
        layer: str,
        image_id: str,
        variant: str = "default",
    ) -> bool:
        """Check if features are cached for given key.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            variant: Optional variant suffix.

        Returns:
            True if both CLS and patch tokens are cached.
        """
        key = CacheKey(model, layer, image_id, variant)
        group_path = self._get_group_path(key)

        if not self.path.exists():
            return False

        with h5py.File(self.path, "r") as f:
            return (
                f"{group_path}/cls_token" in f
                and f"{group_path}/patch_tokens" in f
            )

    def store(
        self,
        model: str,
        layer: str,
        image_id: str,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        variant: str = "default",
    ) -> None:
        """Store CLS and patch token features.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            cls_token: CLS token tensor of shape (embed_dim,) or (1, embed_dim).
            patch_tokens: Patch tokens of shape (num_patches, embed_dim).
            variant: Optional variant suffix.
        """
        key = CacheKey(model, layer, image_id, variant)
        group_path = self._get_group_path(key)

        # Convert to numpy, squeeze any batch dimension
        cls_np = cls_token.squeeze().cpu().numpy().astype(np.float32)
        patches_np = patch_tokens.squeeze(0).cpu().numpy().astype(np.float32)

        with h5py.File(self.path, "a") as f:
            # Create group hierarchy if needed
            if group_path in f:
                del f[group_path]  # Replace existing

            grp = f.create_group(group_path)
            grp.create_dataset("cls_token", data=cls_np, compression="gzip")
            grp.create_dataset("patch_tokens", data=patches_np, compression="gzip")

    def load(
        self,
        model: str,
        layer: str,
        image_id: str,
        variant: str = "default",
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load cached CLS and patch token features.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            variant: Optional variant suffix.
            device: Device to load tensors to.

        Returns:
            Tuple of (cls_token, patch_tokens) tensors.

        Raises:
            KeyError: If the requested features are not cached.
        """
        key = CacheKey(model, layer, image_id, variant)
        group_path = self._get_group_path(key)

        with h5py.File(self.path, "r") as f:
            if group_path not in f:
                raise KeyError(f"Features not cached: {key}")

            grp = f[group_path]
            cls_np = grp["cls_token"][:]
            patches_np = grp["patch_tokens"][:]

        cls_token = torch.from_numpy(cls_np).to(device)
        patch_tokens = torch.from_numpy(patches_np).to(device)

        return cls_token, patch_tokens

    def list_cached(self, model: str | None = None) -> list[CacheKey]:
        """List all cached feature keys.

        Args:
            model: Optional model filter. If None, returns all.

        Returns:
            List of CacheKey objects for cached features.
        """
        if not self.path.exists():
            return []

        keys: list[CacheKey] = []

        with h5py.File(self.path, "r") as f:

            def visitor(name: str, obj: h5py.HLObject) -> None:
                if isinstance(obj, h5py.Group) and "cls_token" in obj:
                    parts = name.split("/")
                    if len(parts) >= 4:
                        m, layer, variant, image_id = parts[:4]
                        if model is None or m == model:
                            keys.append(CacheKey(m, layer, image_id, variant))

            f.visititems(visitor)

        return keys


class AttentionCache:
    """HDF5 cache for attention maps.

    Stores attention maps in float16 format for ~50% storage reduction
    with minimal precision loss (sufficient for visualization).

    Attributes:
        path: Path to the HDF5 cache file.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize the attention cache.

        Args:
            path: Path to the HDF5 file (created if doesn't exist).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _get_dataset_path(self, key: CacheKey) -> str:
        """Get HDF5 dataset path for a cache key."""
        return f"{key.to_path()}/attention"

    def exists(
        self,
        model: str,
        layer: str,
        image_id: str,
        variant: str = "default",
    ) -> bool:
        """Check if attention map is cached for given key.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            variant: Optional variant suffix.

        Returns:
            True if attention map is cached.
        """
        key = CacheKey(model, layer, image_id, variant)
        dataset_path = self._get_dataset_path(key)

        if not self.path.exists():
            return False

        with h5py.File(self.path, "r") as f:
            return dataset_path in f

    def store(
        self,
        model: str,
        layer: str,
        image_id: str,
        attention: torch.Tensor,
        variant: str = "default",
    ) -> None:
        """Store an attention map.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            attention: Attention tensor (any shape, typically (H, W) or (num_patches,)).
            variant: Optional variant suffix.
        """
        key = CacheKey(model, layer, image_id, variant)
        group_path = key.to_path()
        dataset_path = self._get_dataset_path(key)

        # Convert to float16 for storage efficiency
        attn_np = attention.cpu().numpy().astype(np.float16)

        with h5py.File(self.path, "a") as f:
            # Remove existing if present
            if dataset_path in f:
                del f[dataset_path]

            # Ensure group exists
            if group_path not in f:
                f.create_group(group_path)

            f.create_dataset(dataset_path, data=attn_np, compression="gzip")

    def load(
        self,
        model: str,
        layer: str,
        image_id: str,
        variant: str = "default",
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Load a cached attention map.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            variant: Optional variant suffix.
            device: Device to load tensor to.

        Returns:
            Attention tensor (converted back to float32).

        Raises:
            KeyError: If the requested attention map is not cached.
        """
        key = CacheKey(model, layer, image_id, variant)
        dataset_path = self._get_dataset_path(key)

        with h5py.File(self.path, "r") as f:
            if dataset_path not in f:
                raise KeyError(f"Attention not cached: {key}")

            attn_np = f[dataset_path][:].astype(np.float32)

        return torch.from_numpy(attn_np).to(device)

    def list_cached(self, model: str | None = None) -> list[CacheKey]:
        """List all cached attention keys.

        Args:
            model: Optional model filter. If None, returns all.

        Returns:
            List of CacheKey objects for cached attention maps.
        """
        if not self.path.exists():
            return []

        keys: list[CacheKey] = []

        with h5py.File(self.path, "r") as f:

            def visitor(name: str, obj: h5py.HLObject) -> None:
                if isinstance(obj, h5py.Dataset) and name.endswith("/attention"):
                    parts = name.split("/")
                    if len(parts) >= 4:
                        m, layer, variant, image_id = parts[:4]
                        if model is None or m == model:
                            keys.append(CacheKey(m, layer, image_id, variant))

            f.visititems(visitor)

        return keys

    def clear(self, model: str | None = None) -> int:
        """Clear cached attention maps.

        Args:
            model: Optional model filter. If None, clears all.

        Returns:
            Number of items cleared.
        """
        if not self.path.exists():
            return 0

        if model is None:
            # Clear entire file
            self.path.unlink()
            return -1  # Unknown count, file deleted

        # Clear specific model
        count = 0
        with h5py.File(self.path, "a") as f:
            if model in f:
                count = len(list(f[model].keys()))
                del f[model]

        return count
