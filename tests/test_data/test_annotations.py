"""Tests for bounding box annotations.

Priority 5: These tests verify annotation handling is correct since
annotations are ground truth. Coordinate clamping and mask generation
must be correct.
"""

from __future__ import annotations

import pytest
import torch

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


class TestBoundingBoxClamping:
    """Test coordinate clamping in BoundingBox."""

    def test_negative_left_clamped_to_zero(self):
        """Negative left coordinate is clamped to 0.0."""
        bbox = BoundingBox(left=-0.05, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        assert bbox.left == 0.0

    def test_negative_top_clamped_to_zero(self):
        """Negative top coordinate is clamped to 0.0."""
        bbox = BoundingBox(left=0.1, top=-0.1, width=0.3, height=0.4, label=0, group_label=0)

        assert bbox.top == 0.0

    def test_both_negative_clamped(self):
        """Both negative coordinates are clamped to 0.0."""
        bbox = BoundingBox(left=-0.5, top=-0.3, width=0.3, height=0.4, label=0, group_label=0)

        assert bbox.left == 0.0
        assert bbox.top == 0.0


class TestBoundingBoxProperties:
    """Test computed properties of BoundingBox."""

    def test_right_property(self):
        """Right edge is left + width, clamped to 1.0."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        assert bbox.right == 0.4  # 0.1 + 0.3

    def test_right_property_clamped(self):
        """Right edge is clamped to 1.0 if exceeds."""
        bbox = BoundingBox(left=0.8, top=0.2, width=0.5, height=0.4, label=0, group_label=0)

        assert bbox.right == 1.0  # min(0.8 + 0.5, 1.0)

    def test_bottom_property(self):
        """Bottom edge is top + height, clamped to 1.0."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        assert abs(bbox.bottom - 0.6) < 1e-10  # 0.2 + 0.4, with floating point tolerance

    def test_bottom_property_clamped(self):
        """Bottom edge is clamped to 1.0 if exceeds."""
        bbox = BoundingBox(left=0.1, top=0.8, width=0.3, height=0.5, label=0, group_label=0)

        assert bbox.bottom == 1.0  # min(0.8 + 0.5, 1.0)


class TestBoundingBoxToMask:
    """Test mask generation from BoundingBox."""

    def test_mask_dimensions(self):
        """Mask has correct output dimensions."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        mask = bbox.to_mask(224, 224)

        assert mask.shape == (224, 224)

    def test_mask_dtype(self):
        """Mask has boolean dtype."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        mask = bbox.to_mask(224, 224)

        assert mask.dtype == torch.bool

    def test_mask_area_approximates_bbox_area(self):
        """Mask area approximates width * height * total_pixels."""
        bbox = BoundingBox(left=0.0, top=0.0, width=0.5, height=0.5, label=0, group_label=0)

        mask = bbox.to_mask(100, 100)
        actual_area = mask.sum().item()

        # Expected: 0.5 * 0.5 * 10000 = 2500
        expected_area = 0.5 * 0.5 * 100 * 100
        assert abs(actual_area - expected_area) < 10  # Allow small rounding error

    def test_mask_covers_correct_region(self):
        """Mask is True inside bbox and False outside."""
        bbox = BoundingBox(left=0.0, top=0.0, width=0.5, height=0.5, label=0, group_label=0)

        mask = bbox.to_mask(100, 100)

        # Inside region should be True
        assert mask[0:50, 0:50].all()

        # Outside region should be False
        assert not mask[50:100, :].any()
        assert not mask[:, 50:100].any()

    def test_small_bbox_has_at_least_one_pixel(self):
        """Very small bbox produces at least 1 pixel in mask."""
        bbox = BoundingBox(left=0.5, top=0.5, width=0.001, height=0.001, label=0, group_label=0)

        mask = bbox.to_mask(100, 100)

        assert mask.sum() >= 1

    def test_full_image_bbox(self):
        """Full image bbox (0, 0, 1, 1) covers all pixels."""
        bbox = BoundingBox(left=0.0, top=0.0, width=1.0, height=1.0, label=0, group_label=0)

        mask = bbox.to_mask(100, 100)

        assert mask.all()

    def test_non_square_output(self):
        """Mask works with non-square dimensions."""
        bbox = BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=0, group_label=0)

        mask = bbox.to_mask(100, 200)

        assert mask.shape == (100, 200)


class TestImageAnnotation:
    """Test ImageAnnotation dataclass."""

    def test_num_bboxes_property(self):
        """num_bboxes returns correct count."""
        bbox1 = BoundingBox(left=0.1, top=0.1, width=0.2, height=0.2, label=0, group_label=0)
        bbox2 = BoundingBox(left=0.5, top=0.5, width=0.2, height=0.2, label=1, group_label=0)

        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(bbox1, bbox2),
        )

        assert annotation.num_bboxes == 2

    def test_empty_annotation(self):
        """Empty annotation has zero bboxes."""
        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(),
        )

        assert annotation.num_bboxes == 0


class TestGetUnionMask:
    """Test union mask generation from ImageAnnotation."""

    def test_union_of_disjoint_bboxes(self):
        """Union of disjoint bboxes ORs both regions."""
        bbox1 = BoundingBox(left=0.0, top=0.0, width=0.3, height=0.3, label=0, group_label=0)
        bbox2 = BoundingBox(left=0.7, top=0.7, width=0.3, height=0.3, label=1, group_label=0)

        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(bbox1, bbox2),
        )

        mask = annotation.get_union_mask(100, 100)

        # Both regions should be True
        assert mask[0:30, 0:30].all()  # First bbox
        assert mask[70:100, 70:100].all()  # Second bbox

        # Middle should be False
        assert not mask[35:65, 35:65].any()

    def test_union_of_overlapping_bboxes(self):
        """Union of overlapping bboxes covers combined area."""
        bbox1 = BoundingBox(left=0.0, top=0.0, width=0.6, height=0.6, label=0, group_label=0)
        bbox2 = BoundingBox(left=0.4, top=0.4, width=0.6, height=0.6, label=1, group_label=0)

        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(bbox1, bbox2),
        )

        mask = annotation.get_union_mask(100, 100)

        # All of first bbox
        assert mask[0:60, 0:60].all()
        # All of second bbox
        assert mask[40:100, 40:100].all()

    def test_empty_annotation_returns_empty_mask(self):
        """Empty annotation returns all-False mask."""
        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(),
        )

        mask = annotation.get_union_mask(100, 100)

        assert mask.shape == (100, 100)
        assert not mask.any()

    def test_single_bbox_annotation(self):
        """Single bbox annotation returns that bbox's mask."""
        bbox = BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=0, group_label=0)

        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(bbox,),
        )

        union_mask = annotation.get_union_mask(100, 100)
        single_mask = bbox.to_mask(100, 100)

        torch.testing.assert_close(union_mask, single_mask)

    def test_union_mask_dimensions(self):
        """Union mask has correct dimensions."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(bbox,),
        )

        mask = annotation.get_union_mask(224, 224)

        assert mask.shape == (224, 224)


class TestBoundingBoxFrozen:
    """Test that BoundingBox is immutable (frozen dataclass)."""

    def test_cannot_modify_left(self):
        """Cannot modify left coordinate after creation."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        with pytest.raises(AttributeError):
            bbox.left = 0.5  # type: ignore

    def test_cannot_modify_label(self):
        """Cannot modify label after creation."""
        bbox = BoundingBox(left=0.1, top=0.2, width=0.3, height=0.4, label=0, group_label=0)

        with pytest.raises(AttributeError):
            bbox.label = 1  # type: ignore


class TestImageAnnotationFrozen:
    """Test that ImageAnnotation is immutable (frozen dataclass)."""

    def test_cannot_modify_image_id(self):
        """Cannot modify image_id after creation."""
        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(),
        )

        with pytest.raises(AttributeError):
            annotation.image_id = "other.jpg"  # type: ignore

    def test_cannot_modify_bboxes(self):
        """Cannot modify bboxes tuple after creation."""
        annotation = ImageAnnotation(
            image_id="test.jpg",
            styles=("Q46261",),
            bboxes=(),
        )

        with pytest.raises(AttributeError):
            annotation.bboxes = ()  # type: ignore
