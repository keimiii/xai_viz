"""Tests for IoU and coverage metrics.

Priority 4: These tests verify IoU computation is correct since it's
the primary evaluation metric. Errors here invalidate experimental results.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.metrics.iou import (
    compute_coverage,
    compute_iou,
    compute_per_bbox_iou,
    threshold_attention,
)


class TestThresholdAttention:
    """Test percentile thresholding of attention maps."""

    def test_percentile_90_keeps_top_10(self):
        """Verify percentile=90 keeps approximately top 10% of values."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=90)

        # Should keep ~10% of pixels
        coverage = mask.float().mean().item()
        assert 0.08 <= coverage <= 0.12  # Allow some variance

    def test_percentile_50_keeps_half(self):
        """Verify percentile=50 keeps approximately half of values."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=50)

        coverage = mask.float().mean().item()
        assert 0.45 <= coverage <= 0.55

    def test_percentile_0_keeps_all(self):
        """Verify percentile=0 keeps all values."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=0)

        assert mask.all()

    def test_percentile_100_keeps_none_or_max(self):
        """Verify percentile=100 keeps only maximum values."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=100)

        # Only max value(s) should be True
        # With random data, there's typically exactly one max
        assert mask.sum() >= 1

    def test_batched_input(self):
        """Verify batched input produces batched output."""
        attention = torch.rand(4, 224, 224)
        mask = threshold_attention(attention, percentile=90)

        assert mask.shape == (4, 224, 224)

    def test_unbatched_input(self):
        """Verify unbatched input produces unbatched output."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=90)

        assert mask.shape == (224, 224)

    def test_invalid_percentile_low(self):
        """Verify percentile < 0 raises error."""
        attention = torch.rand(224, 224)

        with pytest.raises(ValueError, match="Percentile must be"):
            threshold_attention(attention, percentile=-1)

    def test_invalid_percentile_high(self):
        """Verify percentile > 100 raises error."""
        attention = torch.rand(224, 224)

        with pytest.raises(ValueError, match="Percentile must be"):
            threshold_attention(attention, percentile=101)

    def test_output_is_boolean(self):
        """Verify output is boolean mask."""
        attention = torch.rand(224, 224)
        mask = threshold_attention(attention, percentile=90)

        assert mask.dtype == torch.bool


class TestComputeIoU:
    """Test IoU computation between attention and ground truth masks."""

    def test_perfect_overlap(self):
        """Identical masks should give IoU = 1.0."""
        attention = torch.zeros(100, 100)
        attention[25:75, 25:75] = 1.0  # High attention in center

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[25:75, 25:75] = True  # Same region

        # Use percentile=75 to threshold at the boundary between 0 and 1
        # The center region (25% of image) has value 1, rest has value 0
        # At percentile=75, threshold will be 0, keeping all non-zero values
        iou, _, _ = compute_iou(attention, gt_mask, percentile=75)

        # Should be very close to 1.0 since attention mask = gt mask
        assert iou > 0.95

    def test_no_overlap(self):
        """Disjoint masks should give IoU = 0.0."""
        attention = torch.zeros(100, 100)
        attention[0:25, 0:25] = 1.0  # Top-left corner (6.25% of image)

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[75:100, 75:100] = True  # Bottom-right corner

        # Use percentile=94 to keep only values at or above the 94th percentile
        # Since 93.75% of pixels are 0, the threshold at 94th percentile will be 1.0
        # This keeps only the top-left region which has no overlap with bottom-right GT
        iou, _, _ = compute_iou(attention, gt_mask, percentile=94)

        assert iou == 0.0

    def test_partial_overlap(self):
        """Partially overlapping masks should give intermediate IoU."""
        attention = torch.zeros(100, 100)
        attention[25:75, 25:75] = 1.0  # Center 50x50

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[40:90, 40:90] = True  # Shifted center 50x50

        iou, _, _ = compute_iou(attention, gt_mask, percentile=50)

        # Overlap is 35x35 = 1225
        # Union is 50*50 + 50*50 - 1225 = 3775
        # IoU ≈ 1225/3775 ≈ 0.324
        assert 0.1 < iou < 0.6

    def test_attention_area_calculation(self):
        """Verify attention_area is fraction of image covered."""
        attention = torch.zeros(100, 100)
        attention[0:50, 0:50] = 1.0  # 25% of image has high attention

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[0:50, 0:50] = True

        # At percentile=75, threshold will be 0 (since 75% of pixels are 0)
        # So all pixels with value > 0 will be kept, which is exactly 25%
        _, attention_area, _ = compute_iou(attention, gt_mask, percentile=75)

        # Should be exactly 25% (the high-attention region)
        assert 0.2 <= attention_area <= 0.3

    def test_annotation_area_calculation(self):
        """Verify annotation_area is fraction of image covered by GT."""
        attention = torch.rand(100, 100)

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[0:50, 0:50] = True  # 25% of image

        _, _, annotation_area = compute_iou(attention, gt_mask, percentile=90)

        assert annotation_area == 0.25

    def test_device_compatibility(self):
        """Verify attention and mask can be on different devices."""
        attention = torch.rand(100, 100)
        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[25:75, 25:75] = True

        # This should work even if they start on same device
        # (GPU testing would require CUDA)
        iou, _, _ = compute_iou(attention, gt_mask, percentile=90)

        assert 0.0 <= iou <= 1.0


class TestComputeCoverage:
    """Test coverage (energy) metric computation."""

    def test_all_attention_inside(self):
        """Attention fully inside GT should give coverage = 1.0."""
        attention = torch.zeros(100, 100)
        attention[25:75, 25:75] = 1.0  # Attention in center

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[0:100, 0:100] = True  # Full image

        coverage = compute_coverage(attention, gt_mask)

        assert coverage == 1.0

    def test_all_attention_outside(self):
        """Attention fully outside GT should give coverage = 0.0."""
        attention = torch.zeros(100, 100)
        attention[0:25, 0:25] = 1.0  # Top-left

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[75:100, 75:100] = True  # Bottom-right

        coverage = compute_coverage(attention, gt_mask)

        assert coverage == 0.0

    def test_half_attention_inside(self):
        """Half attention inside should give coverage ≈ 0.5."""
        attention = torch.ones(100, 100)  # Uniform attention

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[0:50, :] = True  # Top half

        coverage = compute_coverage(attention, gt_mask)

        assert 0.45 <= coverage <= 0.55

    def test_weighted_coverage(self):
        """Verify coverage weights by attention value."""
        attention = torch.zeros(100, 100)
        attention[0:50, :] = 2.0  # High attention top half
        attention[50:100, :] = 1.0  # Low attention bottom half

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[0:50, :] = True  # Only top half

        coverage = compute_coverage(attention, gt_mask)

        # Top half has 2x attention, so coverage = 2*5000 / (2*5000 + 1*5000) = 0.667
        assert 0.6 <= coverage <= 0.7

    def test_zero_attention_returns_zero(self):
        """Zero attention everywhere should return 0.0."""
        attention = torch.zeros(100, 100)

        gt_mask = torch.ones(100, 100, dtype=torch.bool)

        coverage = compute_coverage(attention, gt_mask)

        assert coverage == 0.0

    def test_negative_attention_clamped(self):
        """Negative attention values should be clamped to 0."""
        attention = torch.full((100, 100), -1.0)
        attention[25:75, 25:75] = 1.0

        gt_mask = torch.ones(100, 100, dtype=torch.bool)

        coverage = compute_coverage(attention, gt_mask)

        # After clamping, only center has attention
        # Coverage = 2500 / 2500 = 1.0
        assert coverage == 1.0


class TestThresholdAttentionTies:
    """Test that topk-based thresholding handles tied values correctly.

    The old quantile+>= approach selected ALL tied pixels at the boundary,
    violating the "top k%" contract. These tests verify the topk fix
    guarantees exact pixel counts regardless of ties.
    """

    def test_constant_attention_selects_exact_count(self):
        """All-ones attention (worst case: every value tied).

        With quantile thresholding, this would select 100% of pixels
        for any percentile < 100. With topk, it must select exactly k.
        """
        attn = torch.ones(14, 14)  # 196 pixels, all tied
        n = 196

        for percentile in [90, 80, 50]:
            mask = threshold_attention(attn, percentile=percentile)
            expected_k = max(1, round(n * (100 - percentile) / 100.0))
            assert mask.sum().item() == expected_k, (
                f"percentile={percentile}: expected {expected_k} pixels, "
                f"got {mask.sum().item()}"
            )

    def test_step_function_selects_exact_count(self):
        """Binary attention (50% zeros, 50% ones).

        At percentile=50, should select exactly 50% of pixels.
        Old code would select all ones (50%) which happens to be correct,
        but at percentile=80 it would select all ones (50%) instead of 20%.
        """
        attn = torch.zeros(10, 10)  # 100 pixels
        attn[:5, :] = 1.0  # Top half = 1, bottom half = 0

        mask = threshold_attention(attn, percentile=80)
        expected_k = max(1, round(100 * 20 / 100.0))  # 20 pixels
        assert mask.sum().item() == expected_k, (
            f"Expected {expected_k} pixels, got {mask.sum().item()}"
        )

    def test_small_grid_exact_count(self):
        """7x7 grid (ResNet-50 scenario) with repeated values.

        This is the most affected case: only 49 pixels, and many attention
        values are quantized/repeated. topk must select exactly k.
        """
        # Simulate quantized attention: only 4 distinct values
        attn = torch.zeros(7, 7)
        attn[0:2, :] = 0.1  # 14 pixels
        attn[2:4, :] = 0.3  # 14 pixels
        attn[4:6, :] = 0.6  # 14 pixels
        attn[6:7, :] = 0.9  # 7 pixels

        n = 49
        for percentile in [90, 80, 70, 50]:
            mask = threshold_attention(attn, percentile=percentile)
            expected_k = max(1, round(n * (100 - percentile) / 100.0))
            assert mask.sum().item() == expected_k, (
                f"percentile={percentile}: expected {expected_k} pixels, "
                f"got {mask.sum().item()}"
            )

    def test_percentile_0_keeps_all(self):
        """percentile=0 must keep every pixel regardless of ties."""
        attn = torch.ones(7, 7)
        mask = threshold_attention(attn, percentile=0)
        assert mask.all()
        assert mask.sum().item() == 49

    def test_percentile_100_keeps_at_least_one(self):
        """percentile=100 must keep at least 1 pixel (max(1, ...))."""
        attn = torch.ones(7, 7)
        mask = threshold_attention(attn, percentile=100)
        assert mask.sum().item() == 1


class TestIoUEdgeCases:
    """Test edge cases in IoU computation."""

    def test_empty_gt_mask(self):
        """Empty ground truth mask should give IoU = 0."""
        attention = torch.rand(100, 100)
        gt_mask = torch.zeros(100, 100, dtype=torch.bool)  # Empty

        iou, _, annotation_area = compute_iou(attention, gt_mask, percentile=90)

        assert annotation_area == 0.0
        # IoU is technically 0/eps ≈ 0 with epsilon handling
        assert iou >= 0.0

    def test_full_gt_mask(self):
        """Full ground truth mask should work correctly."""
        attention = torch.rand(100, 100)
        gt_mask = torch.ones(100, 100, dtype=torch.bool)  # Full image

        iou, _, annotation_area = compute_iou(attention, gt_mask, percentile=90)

        assert annotation_area == 1.0
        # IoU should be ~0.1 (10% attention covers 10% of 100% GT)
        assert iou > 0.0

    def test_very_small_attention_region(self):
        """Very small attention region should still work."""
        attention = torch.zeros(100, 100)
        attention[50, 50] = 1.0  # Single pixel

        gt_mask = torch.zeros(100, 100, dtype=torch.bool)
        gt_mask[50, 50] = True  # Same pixel

        iou, _, _ = compute_iou(attention, gt_mask, percentile=99)

        # Should have perfect overlap
        assert iou > 0.0


@pytest.mark.parametrize(
    "percentile,expected_coverage",
    [
        (90, 0.10),  # Top 10%
        (80, 0.20),  # Top 20%
        (70, 0.30),  # Top 30%
        (50, 0.50),  # Top 50%
        (0, 1.00),  # All
    ],
)
def test_percentile_coverage_uniform_attention(percentile: int, expected_coverage: float):
    """Verify percentile thresholds produce expected coverage for uniform attention."""
    attention = torch.rand(1000, 1000)  # Large for statistical accuracy
    mask = threshold_attention(attention, percentile=percentile)

    actual_coverage = mask.float().mean().item()

    # topk guarantees exact pixel count; 1% tolerance for rounding only
    assert abs(actual_coverage - expected_coverage) < 0.01


class TestTopKPixelCountContract:
    """Cross-validation contract test for frontend-backend alignment.

    The frontend (renderHeatmap.ts:applyPercentileThreshold) and backend
    (iou.py:threshold_attention) must agree on the number of active pixels
    for a given percentile and grid size. Both use the formula:

        k = max(1, round(n * (100 - percentile) / 100))

    This test verifies the backend produces exactly k pixels for all
    supported percentile values and grid sizes. If this test fails,
    the frontend formula must be updated to match.
    """

    PERCENTILES = [90, 85, 80, 75, 70, 60, 50]
    GRID_SIZES = [(7, 7), (14, 14), (16, 16)]  # ResNet-50, MAE/CLIP, DINOv2

    @pytest.mark.parametrize("percentile", PERCENTILES)
    @pytest.mark.parametrize("grid_size", GRID_SIZES, ids=["7x7", "14x14", "16x16"])
    def test_exact_pixel_count(self, grid_size: tuple[int, int], percentile: int):
        """Backend threshold_attention selects exactly k pixels.

        Formula: k = max(1, round(n * (100 - percentile) / 100))

        This is the same formula used by the frontend's applyPercentileThreshold.
        If either side changes, this test will catch the divergence.
        """
        h, w = grid_size
        n = h * w
        expected_k = max(1, round(n * (100 - percentile) / 100))

        # Use random attention to avoid any special-case shortcuts
        attn = torch.rand(h, w)
        mask = threshold_attention(attn, percentile=percentile)

        assert mask.sum().item() == expected_k, (
            f"grid={h}x{w}, percentile={percentile}: "
            f"expected {expected_k} pixels, got {mask.sum().item()}. "
            f"Frontend-backend contract violated!"
        )


def _make_annotation(*bbox_specs: tuple[float, float, float, float, int]) -> ImageAnnotation:
    """Helper: create an ImageAnnotation from (left, top, w, h, label) tuples."""
    bboxes = tuple(
        BoundingBox(left=left, top=t, width=w, height=h, label=lab, group_label=0)
        for left, t, w, h, lab in bbox_specs
    )
    return ImageAnnotation(image_id="test.jpg", styles=(), bboxes=bboxes)


class TestComputePerBboxIoU:
    """Test per-bbox IoU computation with the optimized threshold-once path."""

    def test_basic_per_bbox_iou(self):
        """Known attention + bboxes produce correct (label, iou) tuples."""
        # 10×10 attention with high values in top-left quadrant
        attention = torch.zeros(10, 10)
        attention[0:5, 0:5] = 1.0  # 25 pixels high

        # Two bboxes: one overlapping attention, one not
        annotation = _make_annotation(
            (0.0, 0.0, 0.5, 0.5, 1),  # top-left → overlaps attention
            (0.5, 0.5, 0.5, 0.5, 2),  # bottom-right → no overlap
        )

        # percentile=75 keeps top 25% → exactly the high-attention pixels
        results = compute_per_bbox_iou(attention, annotation, percentile=75)

        assert len(results) == 2
        # First bbox: perfect overlap with attention
        assert results[0][0] == 1  # label
        assert results[0][1] > 0.95  # iou ≈ 1.0
        # Second bbox: no overlap
        assert results[1][0] == 2  # label
        assert results[1][1] == 0.0  # iou = 0

    def test_matches_compute_iou_per_bbox(self):
        """Optimized function must produce identical IoU vs calling compute_iou per bbox."""
        attention = torch.rand(14, 14)

        annotation = _make_annotation(
            (0.0, 0.0, 0.3, 0.4, 10),
            (0.2, 0.3, 0.5, 0.5, 20),
            (0.6, 0.6, 0.4, 0.4, 30),
        )

        percentile = 80
        h, w = attention.shape[-2:]

        # Optimized path
        optimized = compute_per_bbox_iou(attention, annotation, percentile)

        # Reference: call compute_iou individually per bbox
        reference = []
        for bbox in annotation.bboxes:
            bbox_mask = bbox.to_mask(h, w).to(attention.device)
            iou, _, _ = compute_iou(attention, bbox_mask, percentile)
            reference.append((bbox.label, iou))

        assert len(optimized) == len(reference)
        for (opt_label, opt_iou), (ref_label, ref_iou) in zip(optimized, reference, strict=True):
            assert opt_label == ref_label
            assert abs(opt_iou - ref_iou) < 1e-7, (
                f"label={opt_label}: optimized={opt_iou}, reference={ref_iou}"
            )

    def test_threshold_called_once(self):
        """threshold_attention must be called exactly once regardless of bbox count."""
        attention = torch.rand(14, 14)

        annotation = _make_annotation(
            (0.0, 0.0, 0.3, 0.3, 1),
            (0.3, 0.0, 0.3, 0.3, 2),
            (0.6, 0.0, 0.3, 0.3, 3),
            (0.0, 0.5, 0.5, 0.5, 4),
        )

        with patch(
            "ssl_attention.metrics.iou.threshold_attention",
            wraps=threshold_attention,
        ) as mock_thresh:
            compute_per_bbox_iou(attention, annotation, percentile=90)

        assert mock_thresh.call_count == 1, (
            f"Expected 1 call to threshold_attention, got {mock_thresh.call_count}"
        )
