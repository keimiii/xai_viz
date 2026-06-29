"""Tests for Pointing Game metric with tolerance support.

Tests verify that:
- tolerance=0 preserves strict containment behavior
- tolerance>0 dilates bbox masks, allowing near-misses to count as hits
- Tolerance propagates correctly to batch and per-feature functions
"""

from __future__ import annotations

import torch

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.metrics.pointing_game import (
    compute_pointing_accuracy,
    compute_top_k_accuracy,
    pointing_game_by_feature,
    pointing_game_hit,
    top_k_pointing_accuracy,
)


def _make_annotation(
    left: float, top: float, width: float, height: float, label: int = 0
) -> ImageAnnotation:
    """Create a single-bbox ImageAnnotation for testing."""
    bbox = BoundingBox(
        left=left, top=top, width=width, height=height, label=label, group_label=0
    )
    return ImageAnnotation(image_id="test.jpg", styles=(), bboxes=(bbox,))


class TestPointingGameHitTolerance:
    """Test tolerance parameter in pointing_game_hit()."""

    def test_strict_hit(self):
        """Max attention inside bbox, tolerance=0 -> hit=True."""
        # 100x100 image, bbox covers center 50x50 (pixels 25-74)
        attention = torch.zeros(100, 100)
        attention[50, 50] = 1.0  # Max inside bbox

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hit, max_y, max_x = pointing_game_hit(attention, annotation, tolerance=0)

        assert hit is True
        assert max_y == 50
        assert max_x == 50

    def test_strict_miss(self):
        """Max attention 1px outside bbox, tolerance=0 -> hit=False."""
        attention = torch.zeros(100, 100)
        # bbox right edge is at pixel 74 (left=0.25, width=0.50 -> 25+50-1=74)
        # Place max attention 1px outside
        attention[50, 76] = 1.0

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hit, max_y, max_x = pointing_game_hit(attention, annotation, tolerance=0)

        assert hit is False
        assert max_y == 50
        assert max_x == 76

    def test_tolerance_hit(self):
        """Max attention 10px outside bbox, tolerance=15 -> hit=True."""
        attention = torch.zeros(100, 100)
        # Place max attention 10px outside right edge of bbox
        attention[50, 85] = 1.0

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hit, _, _ = pointing_game_hit(attention, annotation, tolerance=15)

        assert hit is True

    def test_tolerance_miss(self):
        """Max attention 20px outside bbox, tolerance=15 -> hit=False."""
        attention = torch.zeros(100, 100)
        # Place max attention 20px outside right edge (pixel 95)
        attention[50, 95] = 1.0

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hit, _, _ = pointing_game_hit(attention, annotation, tolerance=15)

        assert hit is False

    def test_tolerance_zero_matches_default(self):
        """tolerance=0 produces identical results to the no-tolerance call."""
        attention = torch.rand(100, 100)
        annotation = _make_annotation(left=0.1, top=0.2, width=0.3, height=0.4)

        hit_default, y_default, x_default = pointing_game_hit(attention, annotation)
        hit_zero, y_zero, x_zero = pointing_game_hit(
            attention, annotation, tolerance=0
        )

        assert hit_default == hit_zero
        assert y_default == y_zero
        assert x_default == x_zero

    def test_tolerance_corner_dilation(self):
        """Tolerance expands in all directions, including diagonals."""
        attention = torch.zeros(100, 100)
        # Place max attention diagonally outside bbox corner
        # bbox bottom-right corner at ~(74, 74), place attention at (79, 79)
        # distance = ~5 pixels diagonal, within 15px tolerance
        attention[79, 79] = 1.0

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hit_strict, _, _ = pointing_game_hit(attention, annotation, tolerance=0)
        hit_tolerant, _, _ = pointing_game_hit(attention, annotation, tolerance=15)

        assert hit_strict is False
        assert hit_tolerant is True


class TestTopKPointingTolerance:
    """Test tolerance parameter in top_k_pointing_accuracy()."""

    def test_top_k_with_tolerance(self):
        """Top-k points near bbox edges register as hits with tolerance."""
        attention = torch.zeros(100, 100)
        # Place 3 points: 1 inside bbox, 2 just outside (within 15px)
        attention[50, 50] = 3.0  # Inside bbox
        attention[50, 80] = 2.0  # 5px outside right edge
        attention[50, 85] = 1.0  # 10px outside right edge

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        hits_strict = top_k_pointing_accuracy(
            attention, annotation, k=3, tolerance=0
        )
        hits_tolerant = top_k_pointing_accuracy(
            attention, annotation, k=3, tolerance=15
        )

        assert hits_strict == 1  # Only the inside point
        assert hits_tolerant == 3  # All 3 points within tolerance

    def test_top_k_tolerance_zero_matches_default(self):
        """tolerance=0 produces identical results to the no-tolerance call."""
        attention = torch.rand(100, 100)
        annotation = _make_annotation(left=0.1, top=0.2, width=0.3, height=0.4)

        hits_default = top_k_pointing_accuracy(attention, annotation, k=5)
        hits_zero = top_k_pointing_accuracy(
            attention, annotation, k=5, tolerance=0
        )

        assert hits_default == hits_zero


class TestByFeatureTolerance:
    """Test tolerance parameter in pointing_game_by_feature()."""

    def test_by_feature_with_tolerance(self):
        """Per-bbox dilation works independently."""
        attention = torch.zeros(100, 100)
        # Max attention near bbox 1, far from bbox 2
        attention[15, 15] = 1.0

        bbox1 = BoundingBox(
            left=0.0, top=0.0, width=0.10, height=0.10, label=1, group_label=0
        )  # Covers pixels 0-9; max at 15 is 5px outside
        bbox2 = BoundingBox(
            left=0.5, top=0.5, width=0.10, height=0.10, label=2, group_label=0
        )  # Far away
        annotation = ImageAnnotation(
            image_id="test.jpg", styles=(), bboxes=(bbox1, bbox2)
        )

        results_strict = pointing_game_by_feature(attention, annotation, tolerance=0)
        results_tolerant = pointing_game_by_feature(
            attention, annotation, tolerance=10
        )

        assert results_strict[1] is False
        assert results_strict[2] is False
        assert results_tolerant[1] is True  # Within 10px tolerance
        assert results_tolerant[2] is False  # Still too far

    def test_duplicate_labels_any_hit_semantics(self):
        """When multiple bboxes share a label, result is True if any is hit."""
        attention = torch.zeros(100, 100)
        attention[50, 50] = 1.0  # Max attention at center

        # Two bboxes with the same label: one hit, one miss
        bbox_hit = BoundingBox(
            left=0.25, top=0.25, width=0.50, height=0.50, label=1, group_label=0
        )  # Covers center -> hit
        bbox_miss = BoundingBox(
            left=0.0, top=0.0, width=0.10, height=0.10, label=1, group_label=0
        )  # Top-left corner -> miss

        # Test both orderings to ensure result doesn't depend on bbox order
        for bboxes in [(bbox_hit, bbox_miss), (bbox_miss, bbox_hit)]:
            annotation = ImageAnnotation(
                image_id="test.jpg", styles=(), bboxes=bboxes
            )
            results = pointing_game_by_feature(attention, annotation, tolerance=0)
            assert results[1] is True, f"Failed with order {[b.left for b in bboxes]}"

    def test_by_feature_tolerance_zero_matches_default(self):
        """tolerance=0 produces identical results to the no-tolerance call."""
        attention = torch.rand(100, 100)
        annotation = _make_annotation(left=0.1, top=0.2, width=0.3, height=0.4)

        results_default = pointing_game_by_feature(attention, annotation)
        results_zero = pointing_game_by_feature(attention, annotation, tolerance=0)

        assert results_default == results_zero


class TestBatchFunctionsTolerance:
    """Test tolerance propagation in batch functions."""

    def test_compute_pointing_accuracy_tolerance(self):
        """Tolerance propagates through compute_pointing_accuracy()."""
        # Image 1: max inside bbox -> hit regardless
        attn1 = torch.zeros(100, 100)
        attn1[50, 50] = 1.0
        ann1 = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        # Image 2: max 5px outside bbox -> miss strict, hit tolerant
        attn2 = torch.zeros(100, 100)
        attn2[50, 80] = 1.0
        ann2 = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        acc_strict, results_strict = compute_pointing_accuracy(
            [attn1, attn2], [ann1, ann2], ["img1.jpg", "img2.jpg"], tolerance=0
        )
        acc_tolerant, results_tolerant = compute_pointing_accuracy(
            [attn1, attn2], [ann1, ann2], ["img1.jpg", "img2.jpg"], tolerance=15
        )

        assert acc_strict == 0.5  # 1/2 hits
        assert acc_tolerant == 1.0  # 2/2 hits

    def test_compute_top_k_accuracy_tolerance(self):
        """Tolerance propagates through compute_top_k_accuracy()."""
        attention = torch.zeros(100, 100)
        attention[50, 50] = 2.0  # Inside bbox
        attention[50, 80] = 1.0  # 5px outside bbox

        annotation = _make_annotation(left=0.25, top=0.25, width=0.50, height=0.50)

        acc_strict = compute_top_k_accuracy(
            [attention], [annotation], k=2, tolerance=0
        )
        acc_tolerant = compute_top_k_accuracy(
            [attention], [annotation], k=2, tolerance=15
        )

        assert acc_strict == 0.5  # 1/2 top-k hits
        assert acc_tolerant == 1.0  # 2/2 top-k hits
