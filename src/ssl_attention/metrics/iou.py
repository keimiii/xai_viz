"""IoU and coverage metrics for attention-annotation alignment.

This module provides the primary quantitative metrics for evaluating how well
SSL model attention maps align with expert architectural feature annotations.

Key metrics:
- IoU (Intersection over Union): Primary alignment metric at various percentile thresholds
- Coverage (Energy): Threshold-free metric measuring % of attention inside bboxes
- CorLoc: Binary metric for DINO literature comparison (IoU >= 0.5)

Example:
    from ssl_attention.metrics import compute_image_iou, compute_coverage
    from ssl_attention.data import AnnotatedSubset

    dataset = AnnotatedSubset(DATASET_PATH)
    sample = dataset[0]
    attention = model.get_attention(sample["image"])  # (H, W) tensor

    result = compute_image_iou(attention, sample["annotation"], sample["image_id"], percentile=90)
    print(f"IoU@90: {result.iou:.3f}, Coverage: {result.coverage:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ssl_attention.config import EPSILON
from ssl_attention.metrics.continuous import sanitize_nonnegative_heatmap

if TYPE_CHECKING:
    from ssl_attention.data.annotations import ImageAnnotation


@dataclass
class IoUResult:
    """Result of IoU computation for a single image.

    Attributes:
        image_id: Image filename.
        percentile: Percentile threshold used (e.g., 90 means top 10% attention).
        iou: Intersection over Union score [0, 1].
        coverage: Fraction of attention energy inside annotated regions [0, 1].
        attention_area: Fraction of image covered by thresholded attention.
        annotation_area: Fraction of image covered by annotation bboxes.
    """

    image_id: str
    percentile: int
    iou: float
    coverage: float
    attention_area: float
    annotation_area: float


@dataclass
class BatchIoUResult:
    """Aggregated IoU results across multiple images.

    Attributes:
        percentile: Percentile threshold used.
        mean_iou: Mean IoU across all images.
        std_iou: Standard deviation of IoU scores.
        median_iou: Median IoU score.
        mean_coverage: Mean coverage (energy) score.
        per_image: Individual results for each image.
    """

    percentile: int
    mean_iou: float
    std_iou: float
    median_iou: float
    mean_coverage: float
    per_image: list[IoUResult] = field(default_factory=list)


def threshold_attention(attention: Tensor, percentile: int) -> Tensor:
    """Create binary mask from attention by keeping top percentile values.

    Uses torch.topk to select exactly k = round(n * (100 - percentile) / 100)
    pixels, guaranteeing a fixed pixel count regardless of tied values.

    Args:
        attention: Attention map of shape (H, W) or (B, H, W).
        percentile: Percentile threshold (e.g., 90 keeps top 10% of values).
            Must be in range [0, 100].

    Returns:
        Binary mask of same shape with True for the top-k pixels.

    Example:
        >>> attn = torch.rand(224, 224)
        >>> mask = threshold_attention(attn, percentile=90)
        >>> coverage = mask.float().mean()  # ~0.10
    """
    if percentile < 0 or percentile > 100:
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

    # Handle batched and unbatched input
    is_batched = attention.dim() == 3
    if not is_batched:
        attention = attention.unsqueeze(0)

    batch_size = attention.shape[0]
    masks = []

    for i in range(batch_size):
        attn = attention[i]  # (H, W)
        flat = attn.flatten().float()
        n = flat.numel()
        k = max(1, round(n * (100 - percentile) / 100.0))

        if percentile == 0:
            mask = torch.ones_like(attn, dtype=torch.bool)
        else:
            _, top_indices = torch.topk(flat, k)
            mask = torch.zeros(n, dtype=torch.bool, device=attn.device)
            mask[top_indices] = True
            mask = mask.view(attn.shape)
        masks.append(mask)

    result = torch.stack(masks, dim=0)

    if not is_batched:
        result = result.squeeze(0)

    return result


def compute_iou(
    attention: Tensor,
    gt_mask: Tensor,
    percentile: int,
) -> tuple[float, float, float]:
    """Compute IoU between thresholded attention and ground truth mask.

    Args:
        attention: Attention map of shape (H, W), values in any range.
        gt_mask: Binary ground truth mask of shape (H, W).
        percentile: Percentile threshold for attention binarization.

    Returns:
        Tuple of (iou, attention_area, annotation_area):
        - iou: Intersection over Union score [0, 1]
        - attention_area: Fraction of image covered by thresholded attention
        - annotation_area: Fraction of image covered by ground truth
    """
    # Ensure same device
    if attention.device != gt_mask.device:
        gt_mask = gt_mask.to(attention.device)

    # Binarize attention at percentile threshold
    attn_mask = threshold_attention(attention, percentile)

    # Ensure boolean type
    attn_mask = attn_mask.bool()
    gt_mask = gt_mask.bool()

    # Compute areas
    total_pixels = float(gt_mask.numel())
    attention_area = attn_mask.sum().item() / total_pixels
    annotation_area = gt_mask.sum().item() / total_pixels

    # Compute intersection and union
    intersection = (attn_mask & gt_mask).sum().item()
    union = (attn_mask | gt_mask).sum().item()

    # Compute IoU with epsilon to avoid division by zero
    iou = intersection / (union + EPSILON) if union > 0 else 0.0

    return iou, attention_area, annotation_area


def compute_coverage(attention: Tensor, gt_mask: Tensor) -> float:
    """Compute coverage (energy) metric - threshold-free attention alignment.

    This metric measures what fraction of the total attention energy falls
    inside the annotated bounding boxes. Unlike IoU, it doesn't require
    a percentile threshold, making it more robust.

    Args:
        attention: Attention map of shape (H, W), non-negative values.
        gt_mask: Binary ground truth mask of shape (H, W).

    Returns:
        Coverage score [0, 1]: fraction of attention inside annotated regions.

    Example:
        >>> attn = torch.rand(224, 224)
        >>> gt = torch.zeros(224, 224, dtype=torch.bool)
        >>> gt[50:150, 50:150] = True  # Box covers 50% of image
        >>> coverage = compute_coverage(attn, gt)  # ~0.5 for random attention
    """
    # Ensure same device
    if attention.device != gt_mask.device:
        gt_mask = gt_mask.to(attention.device)

    gt_mask = gt_mask.bool()

    attention = sanitize_nonnegative_heatmap(attention)

    # Total attention energy
    total_energy = attention.sum().item()

    if total_energy < EPSILON:
        return 0.0

    # Energy inside annotated regions
    inside_energy = (attention * gt_mask.float()).sum().item()

    return inside_energy / total_energy


def compute_image_iou(
    attention: Tensor,
    annotation: ImageAnnotation,
    image_id: str,
    percentile: int,
) -> IoUResult:
    """Compute IoU metrics for a single image.

    Args:
        attention: Attention map of shape (H, W).
        annotation: ImageAnnotation with bounding boxes.
        image_id: Image filename for result tracking.
        percentile: Percentile threshold for attention binarization.

    Returns:
        IoUResult with IoU, coverage, and area statistics.
    """
    # Get attention dimensions
    h, w = attention.shape[-2:]

    # Generate union mask at attention resolution
    gt_mask = annotation.get_union_mask(h, w)
    gt_mask = gt_mask.to(attention.device)

    # Compute IoU
    iou, attention_area, annotation_area = compute_iou(attention, gt_mask, percentile)

    # Compute coverage (threshold-free)
    coverage = compute_coverage(attention, gt_mask)

    return IoUResult(
        image_id=image_id,
        percentile=percentile,
        iou=iou,
        coverage=coverage,
        attention_area=attention_area,
        annotation_area=annotation_area,
    )


def compute_batch_iou(
    attention_maps: list[Tensor] | Tensor,
    annotations: list[ImageAnnotation],
    image_ids: list[str],
    percentiles: list[int] | None = None,
) -> dict[int, BatchIoUResult]:
    """Compute IoU metrics for a batch of images at multiple percentiles.

    Args:
        attention_maps: List of attention maps (H, W) or batched tensor (B, H, W).
        annotations: List of ImageAnnotation objects.
        image_ids: List of image filenames.
        percentiles: List of percentile thresholds. Defaults to [90, 80, 70, 60, 50].

    Returns:
        Dict mapping percentile to BatchIoUResult.

    Example:
        >>> results = compute_batch_iou(attn_maps, annotations, image_ids)
        >>> print(f"IoU@90: {results[90].mean_iou:.3f}")
        >>> print(f"IoU@50: {results[50].mean_iou:.3f}")
    """
    if percentiles is None:
        percentiles = [90, 80, 70, 60, 50]

    # Convert batched tensor to list
    if isinstance(attention_maps, Tensor) and attention_maps.dim() == 3:
        attention_maps = [attention_maps[i] for i in range(attention_maps.shape[0])]

    # Validate input lengths
    if not (len(attention_maps) == len(annotations) == len(image_ids)):
        raise ValueError(
            f"Length mismatch: attention_maps={len(attention_maps)}, "
            f"annotations={len(annotations)}, image_ids={len(image_ids)}"
        )

    results: dict[int, BatchIoUResult] = {}

    for percentile in percentiles:
        per_image_results: list[IoUResult] = []

        for attention, annotation, image_id in zip(
            attention_maps, annotations, image_ids, strict=True
        ):
            result = compute_image_iou(
                attention=attention,
                annotation=annotation,
                image_id=image_id,
                percentile=percentile,
            )
            per_image_results.append(result)

        # Compute aggregate statistics
        ious = torch.tensor([r.iou for r in per_image_results])
        coverages = torch.tensor([r.coverage for r in per_image_results])

        results[percentile] = BatchIoUResult(
            percentile=percentile,
            mean_iou=ious.mean().item(),
            std_iou=ious.std().item(),
            median_iou=ious.median().item(),
            mean_coverage=coverages.mean().item(),
            per_image=per_image_results,
        )

    return results


def compute_per_bbox_iou(
    attention: Tensor,
    annotation: ImageAnnotation,
    percentile: int,
) -> list[tuple[int, float]]:
    """Compute IoU for each bounding box individually.

    Useful for per-feature-type analysis (e.g., "does model attend to
    windows more than doors?").

    Args:
        attention: Attention map of shape (H, W).
        annotation: ImageAnnotation with bounding boxes.
        percentile: Percentile threshold for attention binarization.

    Returns:
        List of (label, iou) tuples for each bbox.
        Label is the feature type index (0-105).
    """
    h, w = attention.shape[-2:]
    results: list[tuple[int, float]] = []

    # Threshold attention ONCE for all bboxes (avoids redundant topk per bbox)
    attn_mask = threshold_attention(attention, percentile).bool()

    for bbox in annotation.bboxes:
        bbox_mask = bbox.to_mask(h, w).to(attention.device).bool()
        intersection = (attn_mask & bbox_mask).sum().item()
        union = (attn_mask | bbox_mask).sum().item()
        iou = intersection / (union + EPSILON) if union > 0 else 0.0
        results.append((bbox.label, iou))

    return results


def compute_corloc(
    attention_maps: list[Tensor] | Tensor,
    annotations: list[ImageAnnotation],
    percentile: int = 90,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Correct Localization (CorLoc) metric.

    CorLoc@50 is a standard metric in weakly-supervised object localization
    literature. An image is "correctly localized" if the IoU between the
    predicted attention region and ground truth is >= 0.5.

    Args:
        attention_maps: List of attention maps (H, W) or batched tensor (B, H, W).
        annotations: List of ImageAnnotation objects.
        percentile: Percentile threshold for attention binarization.
        iou_threshold: IoU threshold for "correct" localization. Default 0.5.

    Returns:
        CorLoc score [0, 1]: fraction of images correctly localized.
    """
    # Convert batched tensor to list
    if isinstance(attention_maps, Tensor) and attention_maps.dim() == 3:
        attention_maps = [attention_maps[i] for i in range(attention_maps.shape[0])]

    num_correct = 0

    for attention, annotation in zip(attention_maps, annotations, strict=True):
        h, w = attention.shape[-2:]
        gt_mask = annotation.get_union_mask(h, w)
        gt_mask = gt_mask.to(attention.device)

        iou, _, _ = compute_iou(attention, gt_mask, percentile)

        if iou >= iou_threshold:
            num_correct += 1

    return num_correct / len(attention_maps) if attention_maps else 0.0


def aggregate_by_feature_type(
    per_bbox_results: list[list[tuple[int, float]]],
    feature_names: list[str] | None = None,
) -> dict[int, dict[str, float | str]]:
    """Aggregate per-bbox IoU results by feature type.

    Args:
        per_bbox_results: List of per-image bbox IoU results from compute_per_bbox_iou.
        feature_names: Optional list of feature names indexed by label.

    Returns:
        Dict mapping label to aggregated stats:
        {label: {"mean_iou": ..., "std_iou": ..., "count": ..., "name": ...}}
    """
    from collections import defaultdict

    # Collect all IoUs per label
    label_ious: dict[int, list[float]] = defaultdict(list)

    for image_results in per_bbox_results:
        for label, iou in image_results:
            label_ious[label].append(iou)

    # Aggregate
    result: dict[int, dict[str, float | str]] = {}

    for label, ious in label_ious.items():
        ious_tensor = torch.tensor(ious)
        stats: dict[str, float | str] = {
            "mean_iou": ious_tensor.mean().item(),
            "std_iou": ious_tensor.std().item() if len(ious) > 1 else 0.0,
            "count": float(len(ious)),
        }
        if feature_names and label < len(feature_names):
            stats["name"] = feature_names[label]
        result[label] = stats

    return result
