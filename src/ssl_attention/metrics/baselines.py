"""Baseline attention maps for metric calibration.

This module provides baseline attention generators to calibrate expectations
for the IoU, pointing game, and Gaussian-target continuous metrics:

- Random: Lower bound - what's the expected IoU from random attention?
- Center Gaussian: Tests center bias - do models just attend to image centers?
- Sobel edges: Tests low-level structure - do models just follow gradients?

If SSL model attention doesn't significantly outperform these baselines,
the attention isn't meaningfully aligned with architectural semantics.

Example:
    from ssl_attention.metrics import (
        compute_baseline_continuous_metrics,
        compute_baseline_ious,
        random_baseline,
    )

    # Generate a single random baseline
    attn = random_baseline(224, 224)

    # Compute baseline IoUs for comparison
    baseline_results = compute_baseline_ious(
        annotations, image_ids, images, percentiles=[90, 50]
    )
    print(f"Random IoU@90: {baseline_results['random'][90]:.3f}")

    # Compute threshold-free Gaussian-target baseline references
    continuous_results = compute_baseline_continuous_metrics(
        annotations, image_ids, images
    )
    print(f"Random MSE: {continuous_results['random']['mse']['mean']:.3f}")
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ssl_attention.config import DEFAULT_IMAGE_SIZE
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_kl_divergence,
    compute_mse,
)

if TYPE_CHECKING:
    from ssl_attention.data.annotations import ImageAnnotation


CONTINUOUS_BASELINE_METRICS: tuple[str, ...] = ("mse", "kl", "emd")


def _deterministic_hash(s: str) -> int:
    """Return a deterministic integer hash for a string, stable across Python sessions."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def random_baseline(
    height: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_IMAGE_SIZE,
    seed: int | None = None,
) -> Tensor:
    """Generate random uniform attention map.

    This is the theoretical lower bound for attention quality - random
    attention should achieve ~5-10% IoU depending on annotation coverage.

    Args:
        height: Output height in pixels.
        width: Output width in pixels.
        seed: Random seed for reproducibility.

    Returns:
        Random attention map of shape (height, width), values in [0, 1].
    """
    if seed is not None:
        torch.manual_seed(seed)

    return torch.rand(height, width)


def center_gaussian_baseline(
    height: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_IMAGE_SIZE,
    sigma: float | None = None,
) -> Tensor:
    """Generate center-biased Gaussian attention map.

    Tests whether model attention is just exploiting photographer bias
    (subjects tend to be centered in photos). If models barely beat this
    baseline, they may not be learning semantic attention.

    Args:
        height: Output height in pixels.
        width: Output width in pixels.
        sigma: Gaussian sigma in pixels. If None, defaults to 1/4 of image size.

    Returns:
        Center Gaussian attention map of shape (height, width), values in [0, 1].
    """
    if sigma is None:
        sigma = max(height, width) / 4.0

    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32) - (height - 1) / 2
    x = torch.arange(width, dtype=torch.float32) - (width - 1) / 2

    # Compute Gaussian
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize to [0, 1]
    gaussian = gaussian / gaussian.max()

    return gaussian


def sobel_edge_baseline(
    image: Image.Image | np.ndarray | Tensor,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tensor:
    """Generate edge-based attention using Sobel filter.

    Tests whether model attention is just following low-level image gradients.
    Architectural features have strong edges, so this is an important baseline.

    Args:
        image: Input image as PIL Image, numpy array (H, W, C), or tensor (C, H, W).
        image_size: Output size for the attention map.

    Returns:
        Sobel edge attention map of shape (image_size, image_size), values in [0, 1].
    """
    # Convert to grayscale numpy array
    if isinstance(image, Image.Image):
        image = image.convert("L").resize((image_size, image_size))
        gray = np.array(image, dtype=np.float32) / 255.0
    elif isinstance(image, Tensor):
        # Assume (C, H, W) format - convert to grayscale
        gray = image.mean(dim=0).numpy() if image.dim() == 3 else image.numpy()
        # Resize if needed (simple nearest neighbor)
        if gray.shape != (image_size, image_size):
            from torch.nn.functional import interpolate

            gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
            gray_tensor = interpolate(
                gray_tensor, size=(image_size, image_size), mode="bilinear"
            )
            gray = gray_tensor.squeeze().numpy()
    else:
        # Assume numpy array (H, W, C) or (H, W) - convert to grayscale
        gray = np.mean(image, axis=2) if image.ndim == 3 else image.copy()
        # Resize using PIL
        gray_img = Image.fromarray((gray * 255).astype(np.uint8))
        gray_img = gray_img.resize((image_size, image_size))
        gray = np.array(gray_img, dtype=np.float32) / 255.0

    # Apply Sobel filter manually (avoid scipy/cv2 dependency)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Pad image for convolution
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")

    # Manual 2D convolution
    h, w = gray.shape
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i : i + 3, j : j + 3]
            gx[i, j] = np.sum(window * sobel_x)
            gy[i, j] = np.sum(window * sobel_y)

    # Compute gradient magnitude
    edge_mag = np.sqrt(gx**2 + gy**2)

    # Normalize to [0, 1]
    if edge_mag.max() > 0:
        edge_mag = edge_mag / edge_mag.max()

    return torch.from_numpy(edge_mag)


def saliency_prior_baseline(
    height: int = DEFAULT_IMAGE_SIZE,
    width: int = DEFAULT_IMAGE_SIZE,
) -> Tensor:
    """Generate combined center + border suppression saliency prior.

    This is a common saliency prior that combines center bias with
    reduced attention at image borders (where objects are less likely).

    Args:
        height: Output height in pixels.
        width: Output width in pixels.

    Returns:
        Saliency prior attention map of shape (height, width), values in [0, 1].
    """
    # Start with center gaussian
    center = center_gaussian_baseline(height, width, sigma=max(height, width) / 3)

    # Apply border suppression
    # Distance from border (normalized)
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)

    # Distance from nearest border
    y_dist = torch.minimum(y, height - 1 - y) / ((height - 1) / 2)
    x_dist = torch.minimum(x, width - 1 - x) / ((width - 1) / 2)

    yy, xx = torch.meshgrid(y_dist, x_dist, indexing="ij")
    border_mask = torch.minimum(yy, xx).clamp(0, 1)

    # Combine
    combined = center * border_mask

    # Normalize
    if combined.max() > 0:
        combined = combined / combined.max()

    return combined


def _summarize_population(values: list[float]) -> dict[str, float]:
    """Return population summary statistics for one metric across images."""
    if not values:
        raise ValueError("Cannot summarize an empty baseline metric series")

    values_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values_array.mean()),
        "std": float(values_array.std(ddof=0)),
    }


def _compute_continuous_metric_scores(
    attention: Tensor,
    gt_heatmap: Tensor,
) -> dict[str, float]:
    """Score one attention map against a Gaussian soft target across metrics."""
    return {
        "mse": compute_mse(attention, gt_heatmap),
        "kl": compute_kl_divergence(attention, gt_heatmap),
        "emd": compute_emd(attention, gt_heatmap),
    }


def compute_baseline_ious(
    annotations: list[ImageAnnotation],
    image_ids: list[str],
    images: list[Image.Image] | None = None,
    percentiles: list[int] | None = None,
    n_random_trials: int = 100,
    include_sobel: bool = True,
) -> dict[str, dict[int, float]]:
    """Compute IoU for all baselines across the dataset.

    Args:
        annotations: List of ImageAnnotation objects.
        image_ids: List of image filenames.
        images: List of PIL Images (required for Sobel baseline).
        percentiles: List of percentile thresholds. Defaults to [90, 80, 70, 60, 50].
        n_random_trials: Number of random trials for averaging.
        include_sobel: Whether to compute Sobel baseline (requires images).

    Returns:
        Dict mapping baseline name to {percentile: mean_iou}.

    Example:
        >>> results = compute_baseline_ious(annotations, image_ids, images)
        >>> print(f"Random@90: {results['random'][90]:.3f}")
        >>> print(f"Center@90: {results['center_gaussian'][90]:.3f}")
    """
    from ssl_attention.metrics.iou import compute_iou

    if percentiles is None:
        percentiles = [90, 80, 70, 60, 50]

    results: dict[str, dict[int, float]] = {
        "random": {},
        "center_gaussian": {},
        "saliency_prior": {},
    }

    if include_sobel and images is not None:
        results["sobel_edge"] = {}

    # Pre-generate center gaussian (same for all images)
    center_attn = center_gaussian_baseline()
    saliency_attn = saliency_prior_baseline()

    for percentile in percentiles:
        # Random baseline - average over multiple trials
        random_ious: list[float] = []
        for trial in range(n_random_trials):
            trial_ious: list[float] = []
            for annotation in annotations:
                attn = random_baseline(seed=trial * 1000 + _deterministic_hash(annotation.image_id) % 1000)
                gt_mask = annotation.get_union_mask(
                    DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE
                )
                iou, _, _ = compute_iou(attn, gt_mask, percentile)
                trial_ious.append(iou)
            random_ious.append(sum(trial_ious) / len(trial_ious))
        results["random"][percentile] = sum(random_ious) / len(random_ious)

        # Center gaussian baseline
        center_ious: list[float] = []
        for annotation in annotations:
            gt_mask = annotation.get_union_mask(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            iou, _, _ = compute_iou(center_attn, gt_mask, percentile)
            center_ious.append(iou)
        results["center_gaussian"][percentile] = sum(center_ious) / len(center_ious)

        # Saliency prior baseline
        saliency_ious: list[float] = []
        for annotation in annotations:
            gt_mask = annotation.get_union_mask(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            iou, _, _ = compute_iou(saliency_attn, gt_mask, percentile)
            saliency_ious.append(iou)
        results["saliency_prior"][percentile] = sum(saliency_ious) / len(saliency_ious)

        # Sobel baseline (if images provided)
        if include_sobel and images is not None:
            sobel_ious: list[float] = []
            for image, annotation in zip(images, annotations, strict=True):
                sobel_attn = sobel_edge_baseline(image)
                gt_mask = annotation.get_union_mask(
                    DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE
                )
                iou, _, _ = compute_iou(sobel_attn, gt_mask, percentile)
                sobel_ious.append(iou)
            results["sobel_edge"][percentile] = sum(sobel_ious) / len(sobel_ious)

    return results


def compute_baseline_continuous_metrics(
    annotations: list[ImageAnnotation],
    image_ids: list[str],
    images: list[Image.Image] | None = None,
    n_random_trials: int = 100,
    include_sobel: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute Gaussian-target MSE/KL/EMD summaries for all baselines.

    Returns:
        Dict mapping baseline name to metric summaries with ``mean`` and ``std``
        across images. Random baselines are first averaged per image across
        deterministic trials before the dataset summary is computed.
    """
    _ = image_ids

    per_image_scores: dict[str, dict[str, list[float]]] = {
        "random": {metric_name: [] for metric_name in CONTINUOUS_BASELINE_METRICS},
        "center_gaussian": {
            metric_name: [] for metric_name in CONTINUOUS_BASELINE_METRICS
        },
        "saliency_prior": {
            metric_name: [] for metric_name in CONTINUOUS_BASELINE_METRICS
        },
    }

    if include_sobel and images is not None:
        per_image_scores["sobel_edge"] = {
            metric_name: [] for metric_name in CONTINUOUS_BASELINE_METRICS
        }

    gt_heatmaps = [
        annotation_to_gaussian_heatmap(
            annotation,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE,
        )
        for annotation in annotations
    ]

    center_attn = center_gaussian_baseline()
    saliency_attn = saliency_prior_baseline()

    for annotation, gt_heatmap in zip(annotations, gt_heatmaps, strict=True):
        random_trial_scores: dict[str, list[float]] = {
            metric_name: [] for metric_name in CONTINUOUS_BASELINE_METRICS
        }
        for trial in range(n_random_trials):
            attention = random_baseline(
                seed=trial * 1000 + _deterministic_hash(annotation.image_id) % 1000
            )
            metric_scores = _compute_continuous_metric_scores(attention, gt_heatmap)
            for metric_name, metric_value in metric_scores.items():
                random_trial_scores[metric_name].append(metric_value)

        for metric_name, metric_values in random_trial_scores.items():
            per_image_scores["random"][metric_name].append(
                float(np.mean(metric_values))
            )

        center_scores = _compute_continuous_metric_scores(center_attn, gt_heatmap)
        for metric_name, metric_value in center_scores.items():
            per_image_scores["center_gaussian"][metric_name].append(metric_value)

        saliency_scores = _compute_continuous_metric_scores(saliency_attn, gt_heatmap)
        for metric_name, metric_value in saliency_scores.items():
            per_image_scores["saliency_prior"][metric_name].append(metric_value)

    if include_sobel and images is not None:
        for image, gt_heatmap in zip(images, gt_heatmaps, strict=True):
            sobel_scores = _compute_continuous_metric_scores(
                sobel_edge_baseline(image),
                gt_heatmap,
            )
            for metric_name, metric_value in sobel_scores.items():
                per_image_scores["sobel_edge"][metric_name].append(metric_value)

    return {
        baseline_name: {
            metric_name: _summarize_population(metric_values)
            for metric_name, metric_values in metric_map.items()
        }
        for baseline_name, metric_map in per_image_scores.items()
    }


def compute_baseline_pointing(
    annotations: list[ImageAnnotation],
    image_ids: list[str],
    images: list[Image.Image] | None = None,
    n_random_trials: int = 100,
    include_sobel: bool = True,
    tolerance: int = 0,
) -> dict[str, float]:
    """Compute pointing game accuracy for all baselines.

    Args:
        annotations: List of ImageAnnotation objects.
        image_ids: List of image filenames.
        images: List of PIL Images (required for Sobel baseline).
        n_random_trials: Number of random trials for averaging.
        include_sobel: Whether to compute Sobel baseline (requires images).
        tolerance: Pixel margin to dilate bboxes before checking hit.

    Returns:
        Dict mapping baseline name to pointing accuracy.
    """
    from ssl_attention.metrics.pointing_game import pointing_game_hit

    results: dict[str, float] = {}

    # Random baseline - average over trials
    random_hits: list[float] = []
    for trial in range(n_random_trials):
        trial_hits = 0
        for annotation in annotations:
            attn = random_baseline(seed=trial * 1000 + _deterministic_hash(annotation.image_id) % 1000)
            hit, _, _ = pointing_game_hit(attn, annotation, tolerance=tolerance)
            if hit:
                trial_hits += 1
        random_hits.append(trial_hits / len(annotations))
    results["random"] = sum(random_hits) / len(random_hits)

    # Center gaussian - always hits center
    center_attn = center_gaussian_baseline()
    center_hits = 0
    for annotation in annotations:
        hit, _, _ = pointing_game_hit(center_attn, annotation, tolerance=tolerance)
        if hit:
            center_hits += 1
    results["center_gaussian"] = center_hits / len(annotations)

    # Saliency prior
    saliency_attn = saliency_prior_baseline()
    saliency_hits = 0
    for annotation in annotations:
        hit, _, _ = pointing_game_hit(saliency_attn, annotation, tolerance=tolerance)
        if hit:
            saliency_hits += 1
    results["saliency_prior"] = saliency_hits / len(annotations)

    # Sobel baseline
    if include_sobel and images is not None:
        sobel_hits = 0
        for image, annotation in zip(images, annotations, strict=True):
            sobel_attn = sobel_edge_baseline(image)
            hit, _, _ = pointing_game_hit(sobel_attn, annotation, tolerance=tolerance)
            if hit:
                sobel_hits += 1
        results["sobel_edge"] = sobel_hits / len(annotations)

    return results
