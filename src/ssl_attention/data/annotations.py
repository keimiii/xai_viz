"""Annotation data structures for WikiChurches building parts.

This module provides dataclasses for parsing and working with the
building_parts.json annotation file, which contains expert bounding box
annotations for 139 church images.

Example:
    from ssl_attention.data.annotations import load_annotations
    from ssl_attention.config import ANNOTATIONS_PATH

    annotations = load_annotations(ANNOTATIONS_PATH)
    sample = annotations["Q18785543_wd0.jpg"]
    print(f"Style: {sample.styles}")  # ['Q46261']
    print(f"Num bboxes: {sample.num_bboxes}")  # e.g., 5
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class FeatureType:
    """An architectural feature type from the WikiChurches ontology.

    The WikiChurches dataset defines 106 architectural feature types organized
    in a hierarchical structure (e.g., "Blind Arcade" has parents "Arcade" and "Blind").

    Attributes:
        index: Zero-based index in the meta array.
        name: Human-readable feature name (e.g., "Rose Window").
        parent_indices: Indices of parent feature types in the hierarchy.
    """

    index: int
    name: str
    parent_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundingBox:
    """A normalized bounding box annotation.

    Coordinates are normalized to [0, 1] relative to image dimensions.
    Negative coordinates (found in ~4 images) are clamped to 0.

    Attributes:
        left: Left edge as fraction of image width [0, 1].
        top: Top edge as fraction of image height [0, 1].
        width: Box width as fraction of image width.
        height: Box height as fraction of image height.
        label: Feature type index (0-105) from the ontology.
        group_label: Group label for this bbox (may differ from element label).
    """

    left: float
    top: float
    width: float
    height: float
    label: int
    group_label: int

    def __post_init__(self) -> None:
        """Clamp negative coordinates to 0."""
        if self.left < 0:
            object.__setattr__(self, "left", 0.0)
        if self.top < 0:
            object.__setattr__(self, "top", 0.0)

    @property
    def right(self) -> float:
        """Right edge as fraction of image width."""
        return min(self.left + self.width, 1.0)

    @property
    def bottom(self) -> float:
        """Bottom edge as fraction of image height."""
        return min(self.top + self.height, 1.0)

    def to_mask(self, height: int, width: int) -> torch.Tensor:
        """Convert bounding box to a binary mask.

        Args:
            height: Output mask height in pixels.
            width: Output mask width in pixels.

        Returns:
            Binary mask of shape (height, width) with 1s inside the box.
        """
        mask = torch.zeros(height, width, dtype=torch.bool)

        # Convert normalized coords to pixel coords
        y1 = int(self.top * height)
        y2 = int(self.bottom * height)
        x1 = int(self.left * width)
        x2 = int(self.right * width)

        # Ensure at least 1 pixel if box is very small
        y2 = max(y2, y1 + 1)
        x2 = max(x2, x1 + 1)

        mask[y1:y2, x1:x2] = True
        return mask


@dataclass(frozen=True)
class ImageAnnotation:
    """Complete annotation for a single image.

    Attributes:
        image_id: Image filename (e.g., "Q18785543_wd0.jpg").
        styles: Wikidata Q-IDs for architectural styles (e.g., ["Q46261"]).
        bboxes: Flattened list of all bounding boxes from all groups.
    """

    image_id: str
    styles: tuple[str, ...]
    bboxes: tuple[BoundingBox, ...] = field(default_factory=tuple)

    @property
    def num_bboxes(self) -> int:
        """Number of bounding box annotations."""
        return len(self.bboxes)

    def get_union_mask(self, height: int, width: int) -> torch.Tensor:
        """Create a binary mask covering all annotated regions.

        Args:
            height: Output mask height in pixels.
            width: Output mask width in pixels.

        Returns:
            Binary mask of shape (height, width) with 1s in any annotated region.
        """
        if not self.bboxes:
            return torch.zeros(height, width, dtype=torch.bool)

        union_mask = torch.zeros(height, width, dtype=torch.bool)
        for bbox in self.bboxes:
            union_mask |= bbox.to_mask(height, width)
        return union_mask


def load_feature_types(meta: Sequence[dict]) -> tuple[FeatureType, ...]:
    """Parse the meta array into FeatureType objects.

    Args:
        meta: The "meta" array from building_parts.json containing
            feature type definitions with "name" and "parents" fields.

    Returns:
        Tuple of 106 FeatureType objects indexed by their position.
    """
    return tuple(
        FeatureType(
            index=i,
            name=entry["name"],
            parent_indices=tuple(entry.get("parents", [])),
        )
        for i, entry in enumerate(meta)
    )


def load_annotations(json_path: Path | str) -> dict[str, ImageAnnotation]:
    """Load and parse the building_parts.json annotation file.

    Args:
        json_path: Path to the building_parts.json file.

    Returns:
        Dictionary mapping image_id to ImageAnnotation.
        Contains 139 annotated images with 631 total bounding boxes.

    Raises:
        FileNotFoundError: If the annotation file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    annotations: dict[str, ImageAnnotation] = {}

    for image_id, ann in data["annotations"].items():
        # Flatten all bbox_groups into a single list of BoundingBox
        bboxes: list[BoundingBox] = []

        for group in ann.get("bbox_groups", []):
            group_label = group["group_label"]

            for elem in group.get("elements", []):
                bboxes.append(
                    BoundingBox(
                        left=elem["left"],
                        top=elem["top"],
                        width=elem["width"],
                        height=elem["height"],
                        label=elem["label"],
                        group_label=group_label,
                    )
                )

        annotations[image_id] = ImageAnnotation(
            image_id=image_id,
            styles=tuple(ann.get("styles", [])),
            bboxes=tuple(bboxes),
        )

    return annotations


def load_annotations_with_features(
    json_path: Path | str,
) -> tuple[dict[str, ImageAnnotation], tuple[FeatureType, ...]]:
    """Load annotations along with feature type definitions.

    Convenience function that returns both annotations and the feature
    type ontology in a single call.

    Args:
        json_path: Path to the building_parts.json file.

    Returns:
        Tuple of (annotations dict, feature_types tuple).
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    feature_types = load_feature_types(data["meta"])

    # Parse annotations directly from already-loaded data to avoid reading file twice
    annotations: dict[str, ImageAnnotation] = {}
    for image_id, ann in data["annotations"].items():
        bboxes: list[BoundingBox] = []
        for group in ann.get("bbox_groups", []):
            group_label = group["group_label"]
            for elem in group.get("elements", []):
                bboxes.append(
                    BoundingBox(
                        left=elem["left"],
                        top=elem["top"],
                        width=elem["width"],
                        height=elem["height"],
                        label=elem["label"],
                        group_label=group_label,
                    )
                )
        annotations[image_id] = ImageAnnotation(
            image_id=image_id,
            styles=tuple(ann.get("styles", [])),
            bboxes=tuple(bboxes),
        )

    return annotations, feature_types
