"""Tests for baseline calibration helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.metrics import baselines as baseline_module
from ssl_attention.metrics.baselines import (
    compute_baseline_continuous_metrics,
    random_baseline,
    saliency_prior_baseline,
)
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_kl_divergence,
    compute_mse,
)


def _make_annotation(
    image_id: str,
    *bbox_specs: tuple[float, float, float, float, int],
) -> ImageAnnotation:
    bboxes = tuple(
        BoundingBox(
            left=left,
            top=top,
            width=width,
            height=height,
            label=label,
            group_label=0,
        )
        for left, top, width, height, label in bbox_specs
    )
    return ImageAnnotation(image_id=image_id, styles=(), bboxes=bboxes)


def _make_test_images() -> list[Image.Image]:
    images: list[Image.Image] = []

    image_a = Image.new("RGB", (32, 32), "black")
    draw_a = ImageDraw.Draw(image_a)
    draw_a.rectangle((6, 6, 24, 24), fill="white")
    images.append(image_a)

    image_b = Image.new("RGB", (32, 32), "black")
    draw_b = ImageDraw.Draw(image_b)
    draw_b.line((0, 16, 31, 16), fill="white", width=3)
    draw_b.line((16, 0, 16, 31), fill="white", width=3)
    images.append(image_b)

    return images


def _patch_lightweight_metric_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        baseline_module,
        "annotation_to_gaussian_heatmap",
        lambda *args, **kwargs: torch.zeros((1, 1), dtype=torch.float32),
    )
    monkeypatch.setattr(
        baseline_module,
        "random_baseline",
        lambda *args, seed=None, **kwargs: torch.tensor(
            [[float(seed if seed is not None else -1)]], dtype=torch.float32
        ),
    )
    monkeypatch.setattr(
        baseline_module,
        "center_gaussian_baseline",
        lambda *args, **kwargs: torch.tensor([[10.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        baseline_module,
        "saliency_prior_baseline",
        lambda *args, **kwargs: torch.tensor([[20.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        baseline_module,
        "sobel_edge_baseline",
        lambda *args, **kwargs: torch.tensor([[30.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        baseline_module,
        "compute_mse",
        lambda attention, gt: float(attention.squeeze().item()),
    )
    monkeypatch.setattr(
        baseline_module,
        "compute_kl_divergence",
        lambda attention, gt: float(attention.squeeze().item() + 0.1),
    )
    monkeypatch.setattr(
        baseline_module,
        "compute_emd",
        lambda attention, gt: float(attention.squeeze().item() + 0.2),
    )


def _manual_continuous_baselines(
    annotations: list[ImageAnnotation],
    images: list[Image.Image],
    *,
    n_random_trials: int,
    include_sobel: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    metric_names = ("mse", "kl", "emd")
    per_image_scores: dict[str, dict[str, list[float]]] = {
        "random": {metric_name: [] for metric_name in metric_names},
        "center_gaussian": {metric_name: [] for metric_name in metric_names},
        "saliency_prior": {metric_name: [] for metric_name in metric_names},
    }

    if include_sobel:
        per_image_scores["sobel_edge"] = {
            metric_name: [] for metric_name in metric_names
        }

    center_attention = baseline_module.center_gaussian_baseline()
    saliency_attention = saliency_prior_baseline()

    gt_heatmaps = [
        annotation_to_gaussian_heatmap(annotation, 224, 224)
        for annotation in annotations
    ]

    for annotation, gt_heatmap in zip(annotations, gt_heatmaps, strict=True):
        random_metric_trials = {metric_name: [] for metric_name in metric_names}
        for trial in range(n_random_trials):
            attention = random_baseline(
                seed=trial * 1000
                + baseline_module._deterministic_hash(annotation.image_id) % 1000
            )
            scores = {
                "mse": compute_mse(attention, gt_heatmap),
                "kl": compute_kl_divergence(attention, gt_heatmap),
                "emd": compute_emd(attention, gt_heatmap),
            }
            for metric_name, value in scores.items():
                random_metric_trials[metric_name].append(value)

        for metric_name, values in random_metric_trials.items():
            per_image_scores["random"][metric_name].append(float(np.mean(values)))

        center_scores = {
            "mse": compute_mse(center_attention, gt_heatmap),
            "kl": compute_kl_divergence(center_attention, gt_heatmap),
            "emd": compute_emd(center_attention, gt_heatmap),
        }
        for metric_name, value in center_scores.items():
            per_image_scores["center_gaussian"][metric_name].append(value)

        saliency_scores = {
            "mse": compute_mse(saliency_attention, gt_heatmap),
            "kl": compute_kl_divergence(saliency_attention, gt_heatmap),
            "emd": compute_emd(saliency_attention, gt_heatmap),
        }
        for metric_name, value in saliency_scores.items():
            per_image_scores["saliency_prior"][metric_name].append(value)

    if include_sobel:
        for image, gt_heatmap in zip(images, gt_heatmaps, strict=True):
            sobel_attention = baseline_module.sobel_edge_baseline(image)
            sobel_scores = {
                "mse": compute_mse(sobel_attention, gt_heatmap),
                "kl": compute_kl_divergence(sobel_attention, gt_heatmap),
                "emd": compute_emd(sobel_attention, gt_heatmap),
            }
            for metric_name, value in sobel_scores.items():
                per_image_scores["sobel_edge"][metric_name].append(value)

    return {
        baseline_name: {
            metric_name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
            for metric_name, values in metric_scores.items()
        }
        for baseline_name, metric_scores in per_image_scores.items()
    }


class TestComputeBaselineContinuousMetrics:
    def test_returns_summary_shape_and_omits_sobel_without_images(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_lightweight_metric_stack(monkeypatch)
        annotations = [
            _make_annotation("image_a.jpg", (0.1, 0.1, 0.2, 0.2, 1)),
            _make_annotation("image_b.jpg", (0.6, 0.6, 0.2, 0.2, 2)),
        ]

        results = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=None,
            n_random_trials=2,
        )

        assert set(results) == {"random", "center_gaussian", "saliency_prior"}
        for baseline_result in results.values():
            assert set(baseline_result) == {"mse", "kl", "emd"}
            for metric_summary in baseline_result.values():
                assert set(metric_summary) == {"mean", "std"}
                assert isinstance(metric_summary["mean"], float)
                assert isinstance(metric_summary["std"], float)

    def test_is_deterministic_for_same_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_lightweight_metric_stack(monkeypatch)
        annotations = [
            _make_annotation("image_a.jpg", (0.1, 0.1, 0.2, 0.2, 1)),
            _make_annotation("image_b.jpg", (0.6, 0.6, 0.2, 0.2, 2)),
        ]

        first = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=None,
            n_random_trials=3,
        )
        second = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=None,
            n_random_trials=3,
        )

        assert first == second

    def test_random_baseline_aggregates_trial_means_per_image_before_population_std(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_lightweight_metric_stack(monkeypatch)
        monkeypatch.setattr(
            baseline_module,
            "_deterministic_hash",
            lambda image_id: {"image_a.jpg": 1, "image_b.jpg": 2}[image_id],
        )
        annotations = [
            _make_annotation("image_a.jpg", (0.1, 0.1, 0.2, 0.2, 1)),
            _make_annotation("image_b.jpg", (0.6, 0.6, 0.2, 0.2, 2)),
        ]

        results = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=None,
            n_random_trials=2,
        )

        random_mse = results["random"]["mse"]
        assert random_mse["mean"] == pytest.approx(501.5)
        assert random_mse["std"] == pytest.approx(0.5)

    def test_real_scores_are_finite_non_negative_and_include_sobel(self) -> None:
        annotations = [
            _make_annotation("image_a.jpg", (0.1, 0.1, 0.25, 0.25, 1)),
            _make_annotation("image_b.jpg", (0.55, 0.55, 0.2, 0.2, 2)),
        ]
        images = _make_test_images()

        results = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=images,
            n_random_trials=2,
        )

        assert "sobel_edge" in results
        for baseline_result in results.values():
            for metric_summary in baseline_result.values():
                assert math.isfinite(metric_summary["mean"])
                assert math.isfinite(metric_summary["std"])
                assert metric_summary["mean"] >= 0.0
                assert metric_summary["std"] >= 0.0

    def test_matches_manual_metric_computation(self) -> None:
        annotations = [
            _make_annotation("image_a.jpg", (0.1, 0.1, 0.25, 0.25, 1)),
            _make_annotation("image_b.jpg", (0.55, 0.55, 0.2, 0.2, 2)),
        ]
        images = _make_test_images()

        results = compute_baseline_continuous_metrics(
            annotations,
            image_ids=[annotation.image_id for annotation in annotations],
            images=images,
            n_random_trials=2,
        )
        expected = _manual_continuous_baselines(
            annotations,
            images,
            n_random_trials=2,
            include_sobel=True,
        )

        assert set(results) == set(expected)
        for baseline_name, metric_results in expected.items():
            for metric_name, summary in metric_results.items():
                assert results[baseline_name][metric_name]["mean"] == pytest.approx(
                    summary["mean"]
                )
                assert results[baseline_name][metric_name]["std"] == pytest.approx(
                    summary["std"]
                )
