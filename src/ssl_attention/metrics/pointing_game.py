"""Pointing Game metric for attention evaluation.

The Pointing Game is a binary localization metric: given an attention map,
does the point of maximum attention fall inside any annotated bounding box?

This is a simpler, more interpretable metric than IoU - it asks "is the model
looking at the right thing?" without requiring threshold selection.

Reference:
    Zhang et al. "Top-Down Neural Attention by Excitation Backprop" (ECCV 2016)

Example:
    from ssl_attention.metrics import pointing_game_hit, compute_pointing_accuracy

    hit, max_y, max_x = pointing_game_hit(attention, annotation)
    print(f"Hit: {hit}, Max attention at ({max_y}, {max_x})")

    accuracy, results = compute_pointing_accuracy(attn_maps, annotations, image_ids)
    print(f"Pointing accuracy: {accuracy:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ssl_attention.data.annotations import ImageAnnotation


@dataclass
class PointingResult:
    """Result of pointing game evaluation for a single image.

    Attributes:
        image_id: Image filename.
        hit: Whether max attention point falls inside any bbox.
        max_y: Y-coordinate of maximum attention point.
        max_x: X-coordinate of maximum attention point.
        num_bboxes: Number of bboxes in the image.
    """

    image_id: str
    hit: bool
    max_y: int
    max_x: int
    num_bboxes: int


def pointing_game_hit(
    attention: Tensor,
    annotation: ImageAnnotation,
    tolerance: int = 0,
) -> tuple[bool, int, int]:
    """Check if maximum attention point falls inside any annotated bbox.

    Args:
        attention: Attention map of shape (H, W).
        annotation: ImageAnnotation with bounding boxes.
        tolerance: Pixel margin to dilate bboxes before checking hit.
            The standard Pointing Game protocol (Zhang et al. 2016) uses 15.
            Default 0 preserves strict containment behavior.

    Returns:
        Tuple of (hit, max_y, max_x):
        - hit: True if max attention is inside any bbox (after dilation)
        - max_y: Y-coordinate of maximum attention
        - max_x: X-coordinate of maximum attention
    """
    h, w = attention.shape[-2:]

    # Find maximum attention location
    flat_idx = int(attention.flatten().argmax().item())
    max_y = flat_idx // w
    max_x = flat_idx % w

    # Generate union mask of all bboxes
    gt_mask = annotation.get_union_mask(h, w)
    gt_mask = gt_mask.to(attention.device)

    # Dilate mask by tolerance pixels using max-pooling
    if tolerance > 0:
        kernel = 2 * tolerance + 1
        gt_mask = torch.nn.functional.max_pool2d(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel,
            stride=1,
            padding=tolerance,
        ).squeeze().bool()

    # Check if max point is inside mask
    hit = gt_mask[max_y, max_x].item()

    return bool(hit), max_y, max_x


def compute_pointing_accuracy(
    attention_maps: list[Tensor] | Tensor,
    annotations: list[ImageAnnotation],
    image_ids: list[str],
    tolerance: int = 0,
) -> tuple[float, list[PointingResult]]:
    """Compute pointing game accuracy across multiple images.

    Args:
        attention_maps: List of attention maps (H, W) or batched tensor (B, H, W).
        annotations: List of ImageAnnotation objects.
        image_ids: List of image filenames.
        tolerance: Pixel margin to dilate bboxes before checking hit.

    Returns:
        Tuple of (accuracy, results):
        - accuracy: Fraction of images where max attention hits a bbox [0, 1]
        - results: List of PointingResult for each image
    """
    # Convert batched tensor to list
    if isinstance(attention_maps, Tensor) and attention_maps.dim() == 3:
        attention_maps = [attention_maps[i] for i in range(attention_maps.shape[0])]

    # Validate input lengths
    if not (len(attention_maps) == len(annotations) == len(image_ids)):
        raise ValueError(
            f"Length mismatch: attention_maps={len(attention_maps)}, "
            f"annotations={len(annotations)}, image_ids={len(image_ids)}"
        )

    results: list[PointingResult] = []

    for attention, annotation, image_id in zip(
        attention_maps, annotations, image_ids, strict=True
    ):
        hit, max_y, max_x = pointing_game_hit(attention, annotation, tolerance=tolerance)

        results.append(
            PointingResult(
                image_id=image_id,
                hit=hit,
                max_y=max_y,
                max_x=max_x,
                num_bboxes=annotation.num_bboxes,
            )
        )

    # Compute accuracy
    num_hits = sum(1 for r in results if r.hit)
    accuracy = num_hits / len(results) if results else 0.0

    return accuracy, results


def top_k_pointing_accuracy(
    attention: Tensor,
    annotation: ImageAnnotation,
    k: int = 5,
    tolerance: int = 0,
) -> int:
    """Softer pointing game: how many of top-k attention points hit bboxes?

    This is a relaxed version of the pointing game that counts how many
    of the top k attention points fall inside annotated regions. More
    forgiving than the binary pointing game.

    Args:
        attention: Attention map of shape (H, W).
        annotation: ImageAnnotation with bounding boxes.
        k: Number of top attention points to check.
        tolerance: Pixel margin to dilate bboxes before checking hits.

    Returns:
        Number of top-k points that fall inside bboxes (0 to k).
    """
    h, w = attention.shape[-2:]

    # Get top k indices
    flat = attention.flatten()
    _, top_indices = torch.topk(flat, min(k, flat.numel()))

    # Convert to 2D coordinates
    top_y = top_indices // w
    top_x = top_indices % w

    # Generate union mask
    gt_mask = annotation.get_union_mask(h, w)
    gt_mask = gt_mask.to(attention.device)

    # Dilate mask by tolerance pixels using max-pooling
    if tolerance > 0:
        kernel = 2 * tolerance + 1
        gt_mask = torch.nn.functional.max_pool2d(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel,
            stride=1,
            padding=tolerance,
        ).squeeze().bool()

    # Count hits
    hits = 0
    for y, x in zip(top_y.tolist(), top_x.tolist(), strict=True):
        if gt_mask[y, x]:
            hits += 1

    return hits


def compute_top_k_accuracy(
    attention_maps: list[Tensor] | Tensor,
    annotations: list[ImageAnnotation],
    k: int = 5,
    tolerance: int = 0,
) -> float:
    """Compute mean top-k pointing accuracy across images.

    Args:
        attention_maps: List of attention maps (H, W) or batched tensor (B, H, W).
        annotations: List of ImageAnnotation objects.
        k: Number of top attention points to check per image.
        tolerance: Pixel margin to dilate bboxes before checking hits.

    Returns:
        Mean fraction of top-k points hitting bboxes [0, 1].
    """
    # Convert batched tensor to list
    if isinstance(attention_maps, Tensor) and attention_maps.dim() == 3:
        attention_maps = [attention_maps[i] for i in range(attention_maps.shape[0])]

    if not attention_maps:
        return 0.0

    total_hits = 0
    total_k = 0

    for attention, annotation in zip(attention_maps, annotations, strict=True):
        hits = top_k_pointing_accuracy(attention, annotation, k=k, tolerance=tolerance)
        total_hits += hits
        total_k += k

    return total_hits / total_k if total_k > 0 else 0.0


def pointing_game_by_feature(
    attention: Tensor,
    annotation: ImageAnnotation,
    tolerance: int = 0,
) -> dict[int, bool]:
    """Check pointing game result for each individual bbox.

    Useful for analyzing which feature types are better localized.

    Args:
        attention: Attention map of shape (H, W).
        annotation: ImageAnnotation with bounding boxes.
        tolerance: Pixel margin to dilate each bbox before checking hit.

    Returns:
        Dict mapping feature label to whether max attention hits **any** bbox with that label.
    """
    h, w = attention.shape[-2:]

    # Find maximum attention location
    flat_idx = int(attention.flatten().argmax().item())
    max_y = flat_idx // w
    max_x = flat_idx % w

    # Check each bbox individually
    results: dict[int, bool] = {}

    for bbox in annotation.bboxes:
        mask = bbox.to_mask(h, w)
        mask = mask.to(attention.device)

        # Dilate mask by tolerance pixels using max-pooling
        if tolerance > 0:
            kernel = 2 * tolerance + 1
            mask = torch.nn.functional.max_pool2d(
                mask.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=kernel,
                stride=1,
                padding=tolerance,
            ).squeeze().bool()

        hit = mask[max_y, max_x].item()
        # OR semantics: if any bbox with this label is hit, result is True
        results[bbox.label] = results.get(bbox.label, False) or bool(hit)

    return results
