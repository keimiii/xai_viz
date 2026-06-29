"""WikiChurches dataset classes for SSL attention analysis.

Provides two dataset classes:
- AnnotatedSubset: 139 images with expert bounding box annotations (for IoU evaluation)
- FullDataset: 9,502 images with style labels (for linear probe/fine-tuning)

Example:
    from ssl_attention.data import AnnotatedSubset, FullDataset
    from ssl_attention.config import DATASET_PATH

    # For attention-bbox IoU analysis
    annotated = AnnotatedSubset(DATASET_PATH)
    sample = annotated[0]
    print(sample["image_id"])  # e.g., "Q18785543_wd0.jpg"
    print(sample["annotation"].num_bboxes)  # e.g., 5

    # For style classification
    full = FullDataset(DATASET_PATH)
    sample = full[0]
    print(sample["style_label"])  # e.g., 0 for Romanesque
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from PIL import Image
from torch.utils.data import Dataset

from ssl_attention.config import STYLE_MAPPING
from ssl_attention.data.annotations import ImageAnnotation, load_annotations

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class FullSampleMetadata(TypedDict):
    """Metadata-only payload for a FullDataset sample."""

    image_id: str
    style_label: int | None
    wikidata_id: str


class AnnotatedSubset(Dataset):
    """Dataset of 139 images with expert bounding box annotations.

    Use this dataset for evaluating attention-bbox IoU (whether model
    attention aligns with human-annotated architectural features).

    Each sample is a dict containing:
        - image_id: str - Filename like "Q18785543_wd0.jpg"
        - image: PIL.Image - RGB image (lazy loaded)
        - annotation: ImageAnnotation - Bboxes and style info
        - style_label: int | None - Style class (0-3) or None if not in STYLE_MAPPING

    Args:
        dataset_path: Path to the WikiChurches dataset root.
        transform: Optional transform to apply to images.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.transform = transform

        # Load annotations
        annotations_path = self.dataset_path / "building_parts.json"
        self._annotations = load_annotations(annotations_path)

        # Sort for deterministic ordering
        self._image_ids = sorted(self._annotations.keys())

    def __len__(self) -> int:
        return len(self._image_ids)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples in the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self._image_ids[idx]
        annotation = self._annotations[image_id]

        # Lazy load image
        image_path = self.images_path / image_id
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Map style to label (first style if multiple, None if unknown)
        style_label = None
        if annotation.styles:
            style_label = STYLE_MAPPING.get(annotation.styles[0])

        return {
            "image_id": image_id,
            "image": image,
            "annotation": annotation,
            "style_label": style_label,
        }

    @property
    def image_ids(self) -> list[str]:
        """Sorted list of image IDs in this dataset."""
        return list(self._image_ids)

    @property
    def annotations(self) -> dict[str, ImageAnnotation]:
        """Access the full annotations dictionary."""
        return self._annotations


class FullDataset(Dataset):
    """Dataset of all 9,502 WikiChurches images with style labels.

    Use this dataset for:
    - Linear probe evaluation (frozen features → style classification)
    - Fine-tuning experiments
    - Large-scale attention analysis

    Style labels are derived from Wikidata Q-IDs via churches.json lookup.
    Images without a style in STYLE_MAPPING get style_label=None.

    Each sample is a dict containing:
        - image_id: str - Filename like "Q1000218_wd0.jpg"
        - image: PIL.Image - RGB image (lazy loaded)
        - style_label: int | None - Style class (0-3) or None if not in STYLE_MAPPING
        - wikidata_id: str - Wikidata Q-ID (e.g., "Q1000218")

    Args:
        dataset_path: Path to the WikiChurches dataset root.
        transform: Optional transform to apply to images.
        filter_labeled: If True, only include images with style_label != None.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        transform: Callable[[Image.Image], Any] | None = None,
        filter_labeled: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.transform = transform

        # Load churches.json for Q-ID → style mapping
        churches_path = self.dataset_path / "churches.json"
        with open(churches_path, encoding="utf-8") as f:
            self._churches = json.load(f)

        # Build Q-ID → style_label lookup
        self._qid_to_style: dict[str, int | None] = {}
        for qid, info in self._churches.items():
            styles = info.get("styles", [])
            if styles:
                # Use first style if multiple
                self._qid_to_style[qid] = STYLE_MAPPING.get(styles[0])
            else:
                self._qid_to_style[qid] = None

        # Get all image files
        all_images = sorted(self.images_path.glob("Q*_wd0.jpg"))

        # Optionally filter to only labeled images
        if filter_labeled:
            self._image_paths = [
                p for p in all_images
                if self._get_style_for_path(p) is not None
            ]
        else:
            self._image_paths = all_images

    def _extract_qid(self, image_id: str) -> str:
        """Extract Wikidata Q-ID from image filename.

        Example: "Q18785543_wd0.jpg" → "Q18785543"
        """
        return image_id.split("_")[0]

    def _get_style_for_path(self, image_path: Path) -> int | None:
        """Get style label for an image path."""
        qid = self._extract_qid(image_path.name)
        return self._qid_to_style.get(qid)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples in the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def _resolve_sample(self, idx: int) -> tuple[Path, FullSampleMetadata]:
        """Resolve image path plus metadata from one index lookup."""
        image_path = self._image_paths[idx]
        image_id = image_path.name
        wikidata_id = self._extract_qid(image_id)
        style_label = self._qid_to_style.get(wikidata_id)

        metadata: FullSampleMetadata = {
            "image_id": image_id,
            "style_label": style_label,
            "wikidata_id": wikidata_id,
        }
        return image_path, metadata

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, metadata = self._resolve_sample(idx)
        image_id = metadata["image_id"]
        wikidata_id = metadata["wikidata_id"]
        style_label = metadata["style_label"]

        # Lazy load image
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image_id": image_id,
            "image": image,
            "style_label": style_label,
            "wikidata_id": wikidata_id,
        }

    def get_metadata(self, idx: int) -> FullSampleMetadata:
        """Return sample metadata without loading image bytes.

        This is used by split/label bookkeeping code paths to avoid expensive
        image decoding when only identifiers and labels are needed.
        """
        _, metadata = self._resolve_sample(idx)
        return metadata


def collate_annotated(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate function for AnnotatedSubset.

    Keeps images as a list (not stacked) since model.preprocess()
    handles batching. This allows variable-size images.

    Returns:
        Dict with:
            - image_ids: list[str]
            - images: list[PIL.Image]
            - annotations: list[ImageAnnotation]
            - style_labels: list[int | None]
    """
    return {
        "image_ids": [s["image_id"] for s in batch],
        "images": [s["image"] for s in batch],
        "annotations": [s["annotation"] for s in batch],
        "style_labels": [s["style_label"] for s in batch],
    }


def collate_classification(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate function for style classification batching.

    Similar to collate_annotated but without annotations field.
    Suitable for FullDataset.

    Returns:
        Dict with:
            - image_ids: list[str]
            - images: list[PIL.Image]
            - style_labels: list[int | None]
            - wikidata_ids: list[str] (if present in samples)
    """
    result = {
        "image_ids": [s["image_id"] for s in batch],
        "images": [s["image"] for s in batch],
        "style_labels": [s["style_label"] for s in batch],
    }

    # Include wikidata_ids if present (FullDataset has them)
    if "wikidata_id" in batch[0]:
        result["wikidata_ids"] = [s["wikidata_id"] for s in batch]

    return result
