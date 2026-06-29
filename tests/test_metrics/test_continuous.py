"""Tests for Gaussian soft-target generation and MSE alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ssl_attention.config import DEFAULT_IMAGE_SIZE, EPSILON
from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.metrics import continuous as continuous_module
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_image_emd,
    compute_image_kl,
    compute_image_mse,
    compute_kl_divergence,
    compute_mse,
    gaussian_bbox_heatmap,
    prepare_bounded_heatmap,
    prepare_emd_distribution,
    prepare_probability_distribution,
    sanitize_nonnegative_heatmap,
    soft_union_heatmap,
)
from ssl_attention.metrics.iou import compute_coverage


def _make_annotation(*bbox_specs: tuple[float, float, float, float, int]) -> ImageAnnotation:
    bboxes = tuple(
        BoundingBox(left=left, top=top, width=width, height=height, label=label, group_label=0)
        for left, top, width, height, label in bbox_specs
    )
    return ImageAnnotation(image_id="test.jpg", styles=(), bboxes=bboxes)


def _make_invalid_attention_heatmap() -> torch.Tensor:
    return torch.tensor(
        [
            [float("nan"), -2.0, float("inf"), 0.25],
            [float("-inf"), 0.75, 1.5, -0.1],
            [0.0, 2.0, float("nan"), 3.0],
            [float("inf"), -5.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )


def _make_finite_gt_heatmap() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9],
        ],
        dtype=torch.float32,
    )


def _make_gt_mask(gt_heatmap: torch.Tensor) -> torch.Tensor:
    return gt_heatmap > 0.45


@dataclass(frozen=True)
class KLSmoothingMeasurement:
    raw_added_mass: float
    pseudo_mass_fraction: float
    peak_probability_drop: float
    l1_distance: float


def _normalize_without_epsilon(values: torch.Tensor) -> torch.Tensor:
    sanitized = sanitize_nonnegative_heatmap(values)
    total = sanitized.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(sanitized, 1.0 / sanitized.numel())
    return sanitized / total


def _make_single_pixel_spike_heatmap(size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    heatmap = torch.zeros((size, size), dtype=torch.float32)
    heatmap[size // 2, size // 2] = 1.0
    return heatmap


def _make_hotspot_heatmap(size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    heatmap = torch.zeros((size, size), dtype=torch.float32)
    center = size // 2
    heatmap[center - 1 : center + 2, center - 1 : center + 2] = 1.0 / 9.0
    return heatmap


def _make_broad_gaussian_heatmap(size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    sigma = size / 10.0
    heatmap = torch.exp(-(((xx - center) ** 2) + ((yy - center) ** 2)) / (2 * sigma**2))
    return heatmap / heatmap.sum()


def _make_uniform_heatmap(size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    return torch.full((size, size), 1.0 / (size * size), dtype=torch.float32)


def _measure_kl_smoothing(values: torch.Tensor) -> KLSmoothingMeasurement:
    unsmoothed = _normalize_without_epsilon(values)
    smoothed = prepare_probability_distribution(values)
    sanitized_total = sanitize_nonnegative_heatmap(values).sum().item()
    raw_added_mass = values.numel() * EPSILON
    pseudo_mass_fraction = raw_added_mass / (sanitized_total + raw_added_mass)
    return KLSmoothingMeasurement(
        raw_added_mass=raw_added_mass,
        pseudo_mass_fraction=pseudo_mass_fraction,
        peak_probability_drop=(unsmoothed.max() - smoothed.max()).item(),
        l1_distance=torch.sum(torch.abs(unsmoothed - smoothed)).item(),
    )


class TestGaussianGroundTruth:
    """Verify Gaussian GT construction semantics."""

    def test_single_bbox_produces_centered_unit_heatmap(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))

        heatmap = annotation_to_gaussian_heatmap(annotation, 100, 100)

        assert heatmap.shape == (100, 100)
        assert heatmap.min().item() >= 0.0
        assert heatmap.max().item() == 1.0
        assert heatmap[50, 50].item() > 0.95
        assert heatmap[50, 50].item() > heatmap[10, 10].item()

    def test_multiple_bboxes_use_pixelwise_max_soft_union(self):
        bbox1 = BoundingBox(left=0.1, top=0.1, width=0.2, height=0.2, label=1, group_label=0)
        bbox2 = BoundingBox(left=0.7, top=0.7, width=0.2, height=0.2, label=2, group_label=0)
        annotation = ImageAnnotation(image_id="test.jpg", styles=(), bboxes=(bbox1, bbox2))

        combined = annotation_to_gaussian_heatmap(annotation, 100, 100)
        expected = soft_union_heatmap(
            [
                gaussian_bbox_heatmap(bbox1, 100, 100),
                gaussian_bbox_heatmap(bbox2, 100, 100),
            ]
        )

        assert torch.allclose(combined, expected)

    def test_empty_annotations_produce_zero_heatmap(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())

        heatmap = annotation_to_gaussian_heatmap(annotation, 32, 32)

        assert torch.count_nonzero(heatmap).item() == 0


class TestComputeMse:
    """Verify MSE behaves sensibly for aligned and misaligned attention."""

    def test_identical_attention_and_gt_have_near_zero_mse(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        mse = compute_mse(gt, gt)

        assert mse == 0.0

    def test_spatial_shift_increases_mse(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted = torch.roll(gt, shifts=10, dims=1)

        assert compute_mse(gt, gt) < compute_mse(shifted, gt)

    def test_uniform_and_random_attention_are_worse_than_aligned_attention(self):
        annotation = _make_annotation((0.3, 0.3, 0.25, 0.25, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        uniform = torch.full_like(gt, 0.5)
        torch.manual_seed(7)
        random_attention = torch.rand_like(gt)

        aligned = compute_mse(gt, gt)
        uniform_mse = compute_mse(uniform, gt)
        random_mse = compute_mse(random_attention, gt)

        assert aligned < uniform_mse
        assert aligned < random_mse

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        mse = compute_image_mse(attention, annotation)

        assert torch.isfinite(torch.tensor(mse))
        assert 0.0 <= mse <= 1.0


class TestComputeKlDivergence:
    """Verify KL(GT || attention) behaves sensibly and stays finite."""

    def test_identical_distributions_have_near_zero_kl(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        kl = compute_kl_divergence(gt, gt)

        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_kl_is_non_negative_and_finite_for_sparse_inputs(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 32, 32)
        attention = torch.tensor(
            [[float("nan"), -1.0], [float("inf"), 0.0]],
            dtype=torch.float32,
        )

        kl = compute_kl_divergence(attention, gt[:2, :2])

        assert torch.isfinite(torch.tensor(kl))
        assert kl >= 0.0

    def test_controlled_probability_mass_shift_increases_kl(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted_small = torch.roll(gt, shifts=4, dims=1)
        shifted_large = torch.roll(gt, shifts=12, dims=1)

        aligned = compute_kl_divergence(gt, gt)
        small_shift = compute_kl_divergence(shifted_small, gt)
        large_shift = compute_kl_divergence(shifted_large, gt)

        assert aligned <= small_shift
        assert small_shift < large_shift

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        kl = compute_image_kl(attention, annotation)

        assert torch.isfinite(torch.tensor(kl))
        assert kl >= 0.0


class TestComputeEmd:
    """Verify EMD/Wasserstein-1 behaves sensibly for spatial mismatches."""

    def test_identical_distributions_have_near_zero_emd(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        emd = compute_emd(gt, gt)

        assert emd == pytest.approx(0.0, abs=1e-10)

    def test_larger_spatial_shift_increases_emd(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted_small = torch.roll(gt, shifts=4, dims=1)
        shifted_large = torch.roll(gt, shifts=12, dims=1)

        aligned = compute_emd(gt, gt)
        small_shift = compute_emd(shifted_small, gt)
        large_shift = compute_emd(shifted_large, gt)

        assert aligned <= small_shift
        assert small_shift < large_shift

    def test_near_miss_scores_better_than_far_miss(self):
        annotation = _make_annotation((0.3, 0.3, 0.25, 0.25, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        near_miss = torch.roll(gt, shifts=(2, 2), dims=(0, 1))
        far_miss = torch.roll(gt, shifts=(10, 10), dims=(0, 1))

        assert compute_emd(near_miss, gt) < compute_emd(far_miss, gt)

    def test_falls_back_to_exact_linprog_when_scipy_helper_fails(self, monkeypatch):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted = torch.roll(gt, shifts=6, dims=1)

        monkeypatch.setattr(continuous_module, "wasserstein_distance_nd", lambda *args, **kwargs: None)

        emd = compute_emd(shifted, gt)

        assert torch.isfinite(torch.tensor(emd))
        assert emd > 0.0

    def test_exact_linprog_handles_zero_tail_mass(self, monkeypatch):
        attention = torch.zeros((64, 64), dtype=torch.float32)
        attention[:8, :8] = 1.0
        gt = torch.zeros((64, 64), dtype=torch.float32)
        gt[16:24, 16:24] = 1.0

        # This shape produces a zero-mass final support cell after resizing,
        # which previously made the exact LP fallback numerically infeasible.
        assert prepare_emd_distribution(attention).reshape(-1)[-1].item() == 0.0
        assert prepare_emd_distribution(gt).reshape(-1)[-1].item() == 0.0

        monkeypatch.setattr(continuous_module, "wasserstein_distance_nd", lambda *args, **kwargs: None)

        emd = compute_emd(attention, gt)

        assert torch.isfinite(torch.tensor(emd))
        assert emd > 0.0

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        emd = compute_image_emd(attention, annotation)

        assert torch.isfinite(torch.tensor(emd))
        assert emd >= 0.0


class TestSharedContinuousMetricPreprocessing:
    """Verify shared preprocessing stays aligned across continuous metrics."""

    def test_sanitize_nonnegative_heatmap_zeroes_invalid_and_negative_values(self):
        attention = _make_invalid_attention_heatmap()

        sanitized = sanitize_nonnegative_heatmap(attention)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.25],
                [0.0, 0.75, 1.5, 0.0],
                [0.0, 2.0, 0.0, 3.0],
                [0.0, 0.0, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        assert torch.equal(sanitized, expected)

    def test_prepare_bounded_heatmap_caps_values_after_shared_sanitization(self):
        attention = _make_invalid_attention_heatmap()

        bounded = prepare_bounded_heatmap(attention)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.25],
                [0.0, 0.75, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        assert torch.equal(bounded, expected)

    @pytest.mark.parametrize(
        "values",
        [
            torch.tensor(
                [
                    [float("nan"), float("inf")],
                    [float("-inf"), -3.0],
                ],
                dtype=torch.float32,
            ),
            torch.zeros((2, 2), dtype=torch.float32),
        ],
        ids=["all_invalid", "all_zero"],
    )
    def test_prepare_probability_distribution_returns_uniform_distribution_for_degenerate_inputs(
        self, values: torch.Tensor
    ):
        distribution = prepare_probability_distribution(values)

        expected = torch.full_like(distribution, 1.0 / distribution.numel())
        assert torch.isfinite(distribution).all()
        assert torch.all(distribution >= 0.0)
        assert distribution.sum().item() == pytest.approx(1.0)
        assert torch.allclose(distribution, expected)

    @pytest.mark.parametrize(
        "values",
        [
            torch.tensor(
                [
                    [float("nan"), float("inf")],
                    [float("-inf"), -3.0],
                ],
                dtype=torch.float32,
            ),
            torch.zeros((2, 2), dtype=torch.float32),
        ],
        ids=["all_invalid", "all_zero"],
    )
    def test_prepare_emd_distribution_returns_uniform_distribution_for_degenerate_inputs(
        self, values: torch.Tensor
    ):
        distribution = prepare_emd_distribution(values, size=4)

        expected = torch.full_like(distribution, 1.0 / distribution.numel())
        assert torch.isfinite(distribution).all()
        assert torch.all(distribution >= 0.0)
        assert distribution.sum().item() == pytest.approx(1.0)
        assert torch.allclose(distribution, expected)

    def test_invalid_attention_preprocessing_contract_stays_aligned_across_metrics(self):
        attention = _make_invalid_attention_heatmap()
        gt_heatmap = _make_finite_gt_heatmap()
        gt_mask = _make_gt_mask(gt_heatmap)

        mse = compute_mse(attention, gt_heatmap)
        coverage = compute_coverage(attention, gt_mask)
        kl = compute_kl_divergence(attention, gt_heatmap)
        emd = compute_emd(attention, gt_heatmap)

        assert torch.isfinite(torch.tensor(mse))
        assert torch.isfinite(torch.tensor(coverage))
        assert torch.isfinite(torch.tensor(kl))
        assert torch.isfinite(torch.tensor(emd))

        expected_mse = compute_mse(
            prepare_bounded_heatmap(attention),
            prepare_bounded_heatmap(gt_heatmap),
        )
        expected_coverage = compute_coverage(
            sanitize_nonnegative_heatmap(attention),
            gt_mask,
        )
        attention_distribution = prepare_probability_distribution(attention)
        gt_distribution = prepare_probability_distribution(gt_heatmap)
        expected_kl = torch.sum(
            gt_distribution * (torch.log(gt_distribution) - torch.log(attention_distribution))
        ).item()
        expected_emd = compute_emd(sanitize_nonnegative_heatmap(attention), gt_heatmap)

        assert mse == pytest.approx(expected_mse)
        assert coverage == pytest.approx(expected_coverage)
        assert kl == pytest.approx(expected_kl, rel=1e-6, abs=1e-8)
        assert emd == pytest.approx(expected_emd, rel=1e-6, abs=1e-8)


class TestKlEpsilonSmoothing:
    """Characterize how KL epsilon smoothing affects sparse vs diffuse maps."""

    def test_uniform_distribution_is_unchanged_after_epsilon_smoothing(self):
        uniform = _make_uniform_heatmap()

        unsmoothed = _normalize_without_epsilon(uniform)
        smoothed = prepare_probability_distribution(uniform)
        measurement = _measure_kl_smoothing(uniform)

        assert torch.allclose(smoothed, unsmoothed)
        assert measurement.peak_probability_drop == pytest.approx(0.0, abs=1e-12)
        assert measurement.l1_distance == pytest.approx(0.0, abs=1e-12)

    def test_standard_grid_epsilon_mass_stays_below_acceptability_threshold(self):
        spike = _make_single_pixel_spike_heatmap()
        measurement = _measure_kl_smoothing(spike)

        assert measurement.raw_added_mass == pytest.approx(
            DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE * EPSILON,
            rel=1e-12,
        )
        assert measurement.raw_added_mass == pytest.approx(5.0176e-4, rel=1e-6)
        assert measurement.pseudo_mass_fraction == pytest.approx(
            measurement.raw_added_mass / (1.0 + measurement.raw_added_mass),
            rel=1e-12,
        )
        assert measurement.pseudo_mass_fraction < 1e-3

    def test_single_pixel_spike_has_largest_peak_drop_of_representative_sparse_fixtures(self):
        spike = _measure_kl_smoothing(_make_single_pixel_spike_heatmap())
        hotspot = _measure_kl_smoothing(_make_hotspot_heatmap())
        broad = _measure_kl_smoothing(_make_broad_gaussian_heatmap())
        uniform = _measure_kl_smoothing(_make_uniform_heatmap())

        assert spike.peak_probability_drop > hotspot.peak_probability_drop
        assert hotspot.peak_probability_drop > broad.peak_probability_drop
        assert broad.peak_probability_drop > uniform.peak_probability_drop

    def test_diffuse_fixture_stays_closer_to_unchanged_than_sparse_fixtures(self):
        spike = _measure_kl_smoothing(_make_single_pixel_spike_heatmap())
        hotspot = _measure_kl_smoothing(_make_hotspot_heatmap())
        broad = _measure_kl_smoothing(_make_broad_gaussian_heatmap())
        uniform = _measure_kl_smoothing(_make_uniform_heatmap())

        assert broad.l1_distance < spike.l1_distance
        assert broad.l1_distance < hotspot.l1_distance
        assert uniform.l1_distance == pytest.approx(0.0, abs=1e-12)
