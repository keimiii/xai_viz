"""Tests for AnnotatedSubset.image_ids property.

Verifies that metadata-only iteration (image_ids + annotations) is
consistent with __getitem__ and safe from external mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from ssl_attention.config import STYLE_MAPPING
from ssl_attention.data.wikichurches import AnnotatedSubset, FullDataset


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> AnnotatedSubset:
    """Create a minimal AnnotatedSubset with 3 dummy images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create dummy images (1x1 white pixel)
    ids = ["Q333_wd0.jpg", "Q111_wd0.jpg", "Q222_wd0.jpg"]
    for img_id in ids:
        Image.new("RGB", (1, 1), "white").save(images_dir / img_id)

    # Minimal building_parts.json matching load_annotations schema:
    # top-level "annotations" key, bbox_groups with elements
    annotations = {
        "annotations": {
            "Q333_wd0.jpg": {
                "styles": ["Q46261"],
                "bbox_groups": [
                    {
                        "group_label": 0,
                        "elements": [
                            {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4, "label": 0},
                        ],
                    },
                ],
            },
            "Q111_wd0.jpg": {
                "styles": ["Q46261"],
                "bbox_groups": [
                    {
                        "group_label": 0,
                        "elements": [
                            {"left": 0.0, "top": 0.0, "width": 0.5, "height": 0.5, "label": 0},
                        ],
                    },
                ],
            },
            "Q222_wd0.jpg": {
                "styles": [],
                "bbox_groups": [],
            },
        },
    }
    with open(tmp_path / "building_parts.json", "w") as f:
        json.dump(annotations, f)

    return AnnotatedSubset(tmp_path)


@pytest.fixture()
def tiny_full_dataset(tmp_path: Path) -> FullDataset:
    """Create a minimal FullDataset with labeled and unlabeled samples."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_ids = [
        "Q100_wd0.jpg",
        "Q101_wd0.jpg",
        "Q200_wd0.jpg",
        "Q999_wd0.jpg",
    ]
    for image_id in image_ids:
        Image.new("RGB", (1, 1), "white").save(images_dir / image_id)

    style_qids = list(STYLE_MAPPING.keys())
    assert len(style_qids) >= 2
    churches = {
        "Q100": {"styles": [style_qids[0]]},
        "Q101": {"styles": [style_qids[0]]},
        "Q200": {"styles": [style_qids[1]]},
        "Q999": {"styles": []},
    }
    with open(tmp_path / "churches.json", "w", encoding="utf-8") as f:
        json.dump(churches, f)

    return FullDataset(tmp_path)


class TestImageIdsProperty:
    """Tests for the image_ids property on AnnotatedSubset."""

    def test_image_ids_matches_getitem_order(self, tiny_dataset: AnnotatedSubset) -> None:
        """image_ids[i] equals dataset[i]['image_id'] for all i."""
        for i, image_id in enumerate(tiny_dataset.image_ids):
            assert image_id == tiny_dataset[i]["image_id"]

    def test_image_ids_returns_copy(self, tiny_dataset: AnnotatedSubset) -> None:
        """Mutating the returned list does not affect the dataset."""
        ids = tiny_dataset.image_ids
        ids.clear()

        assert len(tiny_dataset.image_ids) == 3

    def test_image_ids_are_sorted(self, tiny_dataset: AnnotatedSubset) -> None:
        """image_ids are in deterministic sorted order."""
        ids = tiny_dataset.image_ids
        assert ids == sorted(ids)

    def test_annotations_keys_match_image_ids(self, tiny_dataset: AnnotatedSubset) -> None:
        """The set of image_ids equals the set of annotation keys."""
        assert set(tiny_dataset.image_ids) == set(tiny_dataset.annotations.keys())

    def test_length_matches(self, tiny_dataset: AnnotatedSubset) -> None:
        """len(image_ids) equals len(dataset)."""
        assert len(tiny_dataset.image_ids) == len(tiny_dataset)


class TestFullDatasetMetadata:
    """Tests for FullDataset metadata-only access."""

    def test_get_metadata_does_not_open_images(self, tiny_full_dataset: FullDataset) -> None:
        """Metadata lookup should not decode image bytes."""
        with patch("ssl_attention.data.wikichurches.Image.open") as mock_open:
            metadata = tiny_full_dataset.get_metadata(0)

        mock_open.assert_not_called()
        assert metadata == {
            "image_id": "Q100_wd0.jpg",
            "style_label": tiny_full_dataset[0]["style_label"],
            "wikidata_id": "Q100",
        }
