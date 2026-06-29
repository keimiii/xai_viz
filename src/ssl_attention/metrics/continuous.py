"""Continuous metrics for threshold-free attention alignment.

This module adds Gaussian soft ground-truth generation plus dense,
threshold-free metrics on the cached 224x224 attention heatmaps used by the
visualization app. Unlike percentile-thresholded IoU, these metrics compare
dense heatmaps directly and therefore do not depend on a threshold selection.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance_nd
from torch import Tensor

from ssl_attention.config import EPSILON

if TYPE_CHECKING:
    from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


EMD_GRID_SIZE = 8


def sanitize_nonnegative_heatmap(values: Tensor) -> Tensor:
    """Sanitize a heatmap for metrics that require non-negative support."""
    return values.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def prepare_bounded_heatmap(values: Tensor) -> Tensor:
    """Sanitize a heatmap into the shared bounded [0, 1] range.

    Both attention and Gaussian ground truth use the same bounded
    normalization rule so the MSE path does not rescale attention only.
    """
    return sanitize_nonnegative_heatmap(values).clamp(0.0, 1.0)


def prepare_probability_distribution(values: Tensor) -> Tensor:
    """Convert a heatmap into a numerically safe probability distribution.

    The KL metric treats both attention and Gaussian ground truth as
    distributions. To keep the divergence finite, values are sanitized,
    clamped non-negative, epsilon-smoothed, and normalized to sum to 1.
    """
    distribution = sanitize_nonnegative_heatmap(values) + EPSILON

    total = distribution.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(distribution, 1.0 / distribution.numel())

    return distribution / total


def resize_heatmap_for_emd(values: Tensor, size: int = EMD_GRID_SIZE) -> Tensor:
    """Resize a heatmap to the shared support used by the EMD metric."""
    sanitized = sanitize_nonnegative_heatmap(values)
    resized = F.interpolate(
        sanitized.unsqueeze(0).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def prepare_emd_distribution(values: Tensor, size: int = EMD_GRID_SIZE) -> Tensor:
    """Resize and normalize a heatmap into a transport-ready probability map."""
    distribution = resize_heatmap_for_emd(values, size=size)
    total = distribution.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(distribution, 1.0 / distribution.numel())

    return distribution / total


def normalize_transport_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize transport weights while keeping the total mass exactly 1.0.

    The exact LP fallback omits one target constraint to remove redundancy.
    If the total mass drifts below or above 1.0 by floating-point rounding,
    the implied last-cell mass can become slightly negative and make the
    transport problem infeasible even though the underlying distributions are
    valid. We absorb the tiny residual into the largest support cell so the
    final vector is non-negative and sums to exactly 1.0.
    """
    normalized = np.asarray(weights, dtype=np.float64).copy()
    normalized[~np.isfinite(normalized)] = 0.0
    normalized = np.clip(normalized, 0.0, None)

    total = normalized.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(normalized, 1.0 / normalized.size, dtype=np.float64)

    normalized /= total

    anchor_index = int(np.argmax(normalized))
    other_total = normalized.sum() - normalized[anchor_index]
    anchor_value = 1.0 - other_total
    if anchor_value < 0.0 and np.isclose(anchor_value, 0.0, atol=1e-12):
        anchor_value = 0.0
    normalized[anchor_index] = anchor_value
    return normalized


@cache
def emd_support_grid(height: int = EMD_GRID_SIZE, width: int = EMD_GRID_SIZE) -> list[list[float]]:
    """Build the normalized cell-center support grid for exact 2D Wasserstein."""
    x_coords = (torch.arange(width, dtype=torch.float64) + 0.5) / width
    y_coords = (torch.arange(height, dtype=torch.float64) + 0.5) / height
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    support = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    return support.tolist()


@cache
def emd_transport_problem(
    height: int = EMD_GRID_SIZE,
    width: int = EMD_GRID_SIZE,
    omitted_target_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Cache the shared support, transport cost vector, and reduced constraints."""
    support = np.asarray(emd_support_grid(height, width), dtype=np.float64)
    num_points = support.shape[0]
    omitted_target = num_points - 1 if omitted_target_index is None else omitted_target_index

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    for source_index in range(num_points):
        row_offset = source_index * num_points
        for target_index in range(num_points):
            variable_index = row_offset + target_index
            row_indices.append(source_index)
            col_indices.append(variable_index)
            data.append(1.0)
            if target_index != omitted_target:
                target_row = num_points + target_index - (1 if target_index > omitted_target else 0)
                row_indices.append(target_row)
                col_indices.append(variable_index)
                data.append(1.0)

    constraints = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=((num_points * 2) - 1, num_points * num_points),
    ).tocsr()
    cost = distance_matrix(support, support, p=2).reshape(-1)
    return support, cost, constraints


def compute_emd_via_linprog(
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    height: int,
    width: int,
) -> float:
    """Solve the exact discrete OT problem via linear programming."""
    source_weights = normalize_transport_weights(source_weights)
    target_weights = normalize_transport_weights(target_weights)
    omitted_target = int(np.argmax(target_weights))
    _, cost, constraints = emd_transport_problem(
        height,
        width,
        omitted_target_index=omitted_target,
    )
    result = linprog(
        c=cost,
        A_eq=constraints,
        b_eq=np.concatenate((source_weights, np.delete(target_weights, omitted_target))),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"Failed to solve exact EMD transport problem: {result.message}")
    return float(result.fun)


def gaussian_bbox_heatmap(
    bbox: BoundingBox,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Generate an anisotropic Gaussian target heatmap for a single bbox."""
    dtype = torch.float32
    x_coords = torch.arange(width, dtype=dtype, device=device) + 0.5
    y_coords = torch.arange(height, dtype=dtype, device=device) + 0.5
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    bbox_width_px = max((bbox.right - bbox.left) * width, 1.0)
    bbox_height_px = max((bbox.bottom - bbox.top) * height, 1.0)

    center_x = ((bbox.left + bbox.right) * width) / 2.0
    center_y = ((bbox.top + bbox.bottom) * height) / 2.0

    sigma_x = max(bbox_width_px / 4.0, 1.0)
    sigma_y = max(bbox_height_px / 4.0, 1.0)

    exponent = ((xx - center_x) ** 2) / (2.0 * sigma_x**2) + ((yy - center_y) ** 2) / (
        2.0 * sigma_y**2
    )
    heatmap = torch.exp(-exponent)
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def soft_union_heatmap(heatmaps: list[Tensor]) -> Tensor:
    """Combine multiple soft targets using pixelwise max."""
    if not heatmaps:
        raise ValueError("soft_union_heatmap requires at least one heatmap")

    union = heatmaps[0]
    for heatmap in heatmaps[1:]:
        union = torch.maximum(union, heatmap)

    max_value = union.max()
    if max_value > 0:
        union = union / max_value
    return union.clamp(0.0, 1.0)


def annotation_to_gaussian_heatmap(
    annotation: ImageAnnotation,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Generate a soft-union Gaussian heatmap for all bboxes in an annotation."""
    if not annotation.bboxes:
        return torch.zeros((height, width), dtype=torch.float32, device=device)

    heatmaps = [
        gaussian_bbox_heatmap(bbox, height, width, device=device)
        for bbox in annotation.bboxes
    ]
    return soft_union_heatmap(heatmaps)


def compute_mse(attention: Tensor, gt_heatmap: Tensor) -> float:
    """Compute mean squared error between normalized attention and soft GT."""
    if attention.shape != gt_heatmap.shape:
        raise ValueError(
            f"Attention and GT heatmap must have the same shape, got {attention.shape} vs {gt_heatmap.shape}"
        )

    if gt_heatmap.device != attention.device:
        gt_heatmap = gt_heatmap.to(attention.device)

    normalized_attention = prepare_bounded_heatmap(attention)
    normalized_gt = prepare_bounded_heatmap(gt_heatmap)
    return torch.mean((normalized_attention - normalized_gt) ** 2).item()


def compute_image_mse(attention: Tensor, annotation: ImageAnnotation) -> float:
    """Compute MSE between an attention map and an annotation's Gaussian soft union."""
    height, width = attention.shape[-2:]
    gt_heatmap = annotation_to_gaussian_heatmap(annotation, height, width, device=attention.device)
    return compute_mse(attention, gt_heatmap)


def compute_kl_divergence(attention: Tensor, gt_heatmap: Tensor) -> float:
    """Compute KL divergence with the reporting direction KL(GT || attention)."""
    if attention.shape != gt_heatmap.shape:
        raise ValueError(
            f"Attention and GT heatmap must have the same shape, got {attention.shape} vs {gt_heatmap.shape}"
        )

    if gt_heatmap.device != attention.device:
        gt_heatmap = gt_heatmap.to(attention.device)

    attention_distribution = prepare_probability_distribution(attention)
    gt_distribution = prepare_probability_distribution(gt_heatmap)
    divergence = torch.sum(
        gt_distribution * (torch.log(gt_distribution) - torch.log(attention_distribution))
    )
    return divergence.item()


def compute_image_kl(attention: Tensor, annotation: ImageAnnotation) -> float:
    """Compute KL(GT || attention) for an annotation's Gaussian soft union."""
    height, width = attention.shape[-2:]
    gt_heatmap = annotation_to_gaussian_heatmap(annotation, height, width, device=attention.device)
    return compute_kl_divergence(attention, gt_heatmap)


def compute_emd(attention: Tensor, gt_heatmap: Tensor) -> float:
    """Compute exact 2D Wasserstein-1 distance on the shared downsampled support."""
    if attention.shape != gt_heatmap.shape:
        raise ValueError(
            f"Attention and GT heatmap must have the same shape, got {attention.shape} vs {gt_heatmap.shape}"
        )

    if gt_heatmap.device != attention.device:
        gt_heatmap = gt_heatmap.to(attention.device)

    attention_distribution = prepare_emd_distribution(attention)
    gt_distribution = prepare_emd_distribution(gt_heatmap)
    height, width = attention_distribution.shape[-2:]
    support, _, _ = emd_transport_problem(height, width)
    source_weights = normalize_transport_weights(
        attention_distribution.detach().cpu().reshape(-1).numpy().astype(np.float64)
    )
    target_weights = normalize_transport_weights(
        gt_distribution.detach().cpu().reshape(-1).numpy().astype(np.float64)
    )

    try:
        distance = wasserstein_distance_nd(
            support,
            support,
            u_weights=source_weights,
            v_weights=target_weights,
        )
        if distance is not None and np.isfinite(distance):
            return float(distance)
    except TypeError:
        pass

    # SciPy's helper can occasionally return an unsolved LP result (`fun=None`)
    # on real image distributions, so fall back to the exact primal formulation.
    return compute_emd_via_linprog(source_weights, target_weights, height, width)


def compute_image_emd(attention: Tensor, annotation: ImageAnnotation) -> float:
    """Compute EMD/Wasserstein-1 for an annotation's Gaussian soft union."""
    height, width = attention.shape[-2:]
    gt_heatmap = annotation_to_gaussian_heatmap(annotation, height, width, device=attention.device)
    return compute_emd(attention, gt_heatmap)
