"""Service for loading and serving images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PILImage

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import (
    ANNOTATIONS_PATH,
    HEATMAPS_PATH,
    IMAGES_PATH,
    STANDARD_IMAGE_SIZE,
    STYLE_NAMES,
    THUMBNAIL_SIZE,
)

if TYPE_CHECKING:
    from ssl_attention.data.annotations import ImageAnnotation


class ImageService:
    """Service for loading images and annotations."""

    _instance: ImageService | None = None
    _annotations: dict[str, ImageAnnotation] | None = None
    _feature_types: list[str] | None = None

    def __new__(cls) -> ImageService:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_annotations(self) -> None:
        """Load annotations from JSON file."""
        if self._annotations is not None:
            return

        from ssl_attention.data.annotations import load_annotations_with_features

        self._annotations, features = load_annotations_with_features(ANNOTATIONS_PATH)
        self._feature_types = [f.name for f in features]

    @property
    def annotations(self) -> dict[str, ImageAnnotation]:
        """Get annotations dictionary."""
        self._load_annotations()
        return self._annotations  # type: ignore

    @property
    def feature_types(self) -> list[str]:
        """Get feature type names."""
        self._load_annotations()
        return self._feature_types  # type: ignore

    def list_image_ids(self) -> list[str]:
        """Get all annotated image IDs."""
        return list(self.annotations.keys())

    def get_annotation(self, image_id: str) -> ImageAnnotation | None:
        """Get annotation for an image."""
        return self.annotations.get(image_id)

    def get_style_names(self, style_qids: list[str]) -> list[str]:
        """Convert style QIDs to human-readable names."""
        from ssl_attention.config import STYLE_MAPPING

        names = []
        for qid in style_qids:
            if qid in STYLE_MAPPING:
                names.append(STYLE_NAMES[STYLE_MAPPING[qid]])
        return names

    def get_feature_name(self, label_index: int) -> str | None:
        """Get feature type name by index."""
        if 0 <= label_index < len(self.feature_types):
            return self.feature_types[label_index]
        return None

    def get_image_path(self, image_id: str) -> Path:
        """Get path to original image file."""
        return IMAGES_PATH / image_id

    def image_exists(self, image_id: str) -> bool:
        """Check if original image file exists."""
        return self.get_image_path(image_id).exists()

    def load_image(self, image_id: str, size: tuple[int, int] | None = None) -> PILImage.Image:
        """Load and optionally resize an image.

        Args:
            image_id: Image filename.
            size: Optional (width, height) to resize to.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If image doesn't exist.
        """
        path = self.get_image_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_id}")

        img = PILImage.open(path).convert("RGB")
        if size:
            img = img.resize(size, PILImage.Resampling.BILINEAR)
        return img

    def load_thumbnail(self, image_id: str) -> PILImage.Image:
        """Load image as thumbnail."""
        return self.load_image(image_id, size=THUMBNAIL_SIZE)

    def load_standard(self, image_id: str) -> PILImage.Image:
        """Load image at standard size (224x224)."""
        return self.load_image(image_id, size=STANDARD_IMAGE_SIZE)

    def get_heatmap_path(
        self,
        model: str,
        layer: str,
        image_id: str,
        method: str = "cls",
        variant: str = "overlay",
    ) -> Path:
        """Get path to pre-rendered heatmap PNG.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            method: Attention method ("cls", "rollout", "mean", "gradcam").
            variant: "heatmap", "overlay", or "overlay_bbox".

        Returns:
            Path to PNG file.
        """
        return HEATMAPS_PATH / model / layer / method / variant / f"{image_id}.png"

    def heatmap_exists(
        self,
        model: str,
        layer: str,
        image_id: str,
        method: str = "cls",
        variant: str = "overlay",
    ) -> bool:
        """Check if pre-rendered heatmap exists."""
        return self.get_heatmap_path(model, layer, image_id, method, variant).exists()

    def load_heatmap(
        self,
        model: str,
        layer: str,
        image_id: str,
        method: str = "cls",
        variant: str = "overlay",
    ) -> PILImage.Image:
        """Load pre-rendered heatmap image.

        Args:
            model: Model name.
            layer: Layer identifier.
            image_id: Image filename.
            method: Attention method ("cls", "rollout", "mean", "gradcam").
            variant: "heatmap", "overlay", or "overlay_bbox".

        Returns:
            PIL Image.

        Raises:
            FileNotFoundError: If heatmap not pre-rendered.
        """
        path = self.get_heatmap_path(model, layer, image_id, method, variant)
        if not path.exists():
            raise FileNotFoundError(
                f"Heatmap not found: {model}/{layer}/{method}/{variant}/{image_id}"
            )
        return PILImage.open(path)

    def get_original_with_bbox_path(self, image_id: str) -> Path:
        """Get path to original image with bounding boxes."""
        return HEATMAPS_PATH / "originals" / "bbox" / f"{image_id}.png"


# Global instance
image_service = ImageService()
