"""Service for computing cosine similarity between bbox features and image patches."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import CACHE_PATH, resolve_model_name, split_model_name
from ssl_attention.cache import FeatureCache
from ssl_attention.config import MODELS

# Patch grid sizes per model (based on 224x224 input / patch_size)
# DINOv2: 14px patch_size -> 224/14 = 16x16 grid (256 patches)
# DINOv3, MAE, CLIP, SigLIP: 16px patch_size -> 224/16 = 14x14 grid (196 patches)
# ResNet-50: 7x7 final feature map (49 spatial positions)
# Note: Uses canonical model names (after alias resolution)
MODEL_PATCH_GRIDS: dict[str, tuple[int, int]] = {
    "dinov2": (16, 16),  # 256 patches
    "dinov3": (14, 14),  # 196 patches
    "mae": (14, 14),     # 196 patches
    "clip": (14, 14),    # 196 patches
    "siglip": (14, 14),  # 196 patches
    "siglip2": (14, 14),  # 196 patches
    "resnet50": (7, 7),  # 49 feature positions
}


class SimilarityService:
    """Service for computing cosine similarity between bboxes and patches."""

    _instance: SimilarityService | None = None
    _cache: FeatureCache | None = None

    def __new__(cls) -> SimilarityService:
        """Singleton pattern for cache access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cache = FeatureCache(CACHE_PATH / "features.h5")
        return cls._instance

    @property
    def cache(self) -> FeatureCache:
        """Get the feature cache instance."""
        if self._cache is None:
            self._cache = FeatureCache(CACHE_PATH / "features.h5")
        return self._cache

    def get_patch_grid(self, model: str) -> tuple[int, int]:
        """Get the patch grid dimensions for a model.

        Args:
            model: Model name (e.g., "dinov2").

        Returns:
            Tuple of (rows, cols) for the patch grid.
        """
        if model in MODEL_PATCH_GRIDS:
            return MODEL_PATCH_GRIDS[model]
        base_model, _, _ = split_model_name(model)
        if base_model in MODEL_PATCH_GRIDS:
            return MODEL_PATCH_GRIDS[base_model]

        # Fallback: compute from config
        if model in MODELS:
            patch_size = MODELS[model].patch_size
            grid_size = 224 // patch_size
            return (grid_size, grid_size)
        if base_model in MODELS:
            patch_size = MODELS[base_model].patch_size
            grid_size = 224 // patch_size
            return (grid_size, grid_size)

        # Default to 14x14
        return (14, 14)

    def bbox_to_patch_indices(
        self,
        left: float,
        top: float,
        width: float,
        height: float,
        patches_per_side: int,
    ) -> list[int]:
        """Convert normalized bbox coords to patch grid indices.

        Args:
            left: Left edge (0-1 normalized).
            top: Top edge (0-1 normalized).
            width: Width (0-1 normalized).
            height: Height (0-1 normalized).
            patches_per_side: Number of patches along each dimension.

        Returns:
            List of patch indices (flattened row-major order) that overlap with bbox.
        """
        # Convert normalized coords to patch grid coords
        col_start = int(left * patches_per_side)
        col_end = int((left + width) * patches_per_side)
        row_start = int(top * patches_per_side)
        row_end = int((top + height) * patches_per_side)

        # Clamp to valid range
        col_start = max(0, min(col_start, patches_per_side - 1))
        col_end = max(col_start + 1, min(col_end + 1, patches_per_side))
        row_start = max(0, min(row_start, patches_per_side - 1))
        row_end = max(row_start + 1, min(row_end + 1, patches_per_side))

        # Collect all patch indices in the bbox region (row-major order)
        indices = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                idx = row * patches_per_side + col
                indices.append(idx)

        return indices

    def compute_similarity(
        self,
        image_id: str,
        model: str,
        layer: int,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> dict:
        """Compute cosine similarity between bbox features and all patches.

        Args:
            image_id: Image filename.
            model: Model name.
            layer: Layer number (0-11).
            left: Bbox left edge (0-1 normalized).
            top: Bbox top edge (0-1 normalized).
            width: Bbox width (0-1 normalized).
            height: Bbox height (0-1 normalized).

        Returns:
            Dict with:
                - similarity: List of floats (one per patch)
                - patch_grid: [rows, cols]
                - min_similarity: float
                - max_similarity: float
                - bbox_patch_indices: List of patch indices in the bbox
        """
        layer_key = f"layer{layer}"

        # Resolve model alias to canonical name for cache lookup
        cache_model = resolve_model_name(model)

        # Load cached features
        try:
            _, patch_tokens = self.cache.load(cache_model, layer_key, image_id)
        except KeyError as e:
            raise ValueError(
                f"Features not cached for {model}/{layer_key}/{image_id}. "
                "Run generate_feature_cache.py first."
            ) from e

        # Get patch grid dimensions (use resolved model name for consistency)
        grid_rows, grid_cols = self.get_patch_grid(cache_model)
        total_patches = grid_rows * grid_cols

        # Verify patch count matches
        if patch_tokens.shape[0] != total_patches:
            # Some models might have different patch counts
            # Adjust grid dimensions based on actual token count
            actual_count = patch_tokens.shape[0]
            grid_size = int(math.sqrt(actual_count))
            if grid_size * grid_size == actual_count:
                grid_rows = grid_cols = grid_size
                total_patches = actual_count

        # Map bbox to patch indices
        bbox_indices = self.bbox_to_patch_indices(
            left, top, width, height, grid_cols  # Assuming square grid
        )

        # Handle edge case: empty bbox
        if not bbox_indices:
            # Return uniform similarity if no patches in bbox
            return {
                "similarity": [0.0] * total_patches,
                "patch_grid": [grid_rows, grid_cols],
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "bbox_patch_indices": [],
            }

        # Ensure indices are valid
        bbox_indices = [i for i in bbox_indices if 0 <= i < total_patches]

        if not bbox_indices:
            return {
                "similarity": [0.0] * total_patches,
                "patch_grid": [grid_rows, grid_cols],
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "bbox_patch_indices": [],
            }

        # Compute query vector: mean of patch features within bbox
        bbox_features = patch_tokens[bbox_indices]  # (num_bbox_patches, embed_dim)
        query_vector = bbox_features.mean(dim=0, keepdim=True)  # (1, embed_dim)

        # Normalize for cosine similarity
        query_normalized = F.normalize(query_vector, p=2, dim=1)
        patches_normalized = F.normalize(patch_tokens, p=2, dim=1)

        # Compute cosine similarity: (1, embed_dim) @ (embed_dim, num_patches) -> (1, num_patches)
        similarity = torch.mm(query_normalized, patches_normalized.t()).squeeze(0)

        # Convert to list
        similarity_list = similarity.tolist()

        return {
            "similarity": similarity_list,
            "patch_grid": [grid_rows, grid_cols],
            "min_similarity": min(similarity_list),
            "max_similarity": max(similarity_list),
            "bbox_patch_indices": bbox_indices,
        }

    def features_exist(self, model: str, layer: int, image_id: str) -> bool:
        """Check if features are cached for given parameters."""
        layer_key = f"layer{layer}"
        cache_model = resolve_model_name(model)
        result: bool = self.cache.exists(cache_model, layer_key, image_id)
        return result


# Global instance
similarity_service = SimilarityService()
