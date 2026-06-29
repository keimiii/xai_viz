"""Metrics for evaluating attention-annotation alignment.

This module provides quantitative evaluation metrics for measuring how well
SSL model attention maps align with 631 expert architectural feature annotations
across 139 WikiChurches images.

Primary Metrics:
- IoU (percentile-thresholded): Primary alignment metric at 90/80/70/60/50 percentiles
- Coverage (Energy): Threshold-free metric - % of attention inside bboxes
- MSE (Gaussian GT): Threshold-free mean squared error against soft bbox targets
- KL Divergence (Gaussian GT): Threshold-free KL(GT || attention) distribution alignment
- EMD / Wasserstein-1 (Gaussian GT): Threshold-free spatial transport distance
- Pointing Game: Binary metric - does max attention hit any bbox?

Secondary Metrics:
- CorLoc@50: Binary IoU≥0.5 for DINO literature comparison
- Per-feature-type IoU: Breakdown across 106 architectural feature labels

Baselines:
- Random: Lower bound (~5-10% IoU expected)
- Center Gaussian: Tests center bias
- Sobel edges: Tests low-level structure attention

Statistical Tests:
- Paired comparisons (t-test, Wilcoxon) between models
- Bootstrap confidence intervals
- Multiple comparison correction (Holm)

Example:
    >>> from ssl_attention.metrics import compute_batch_iou, compute_pointing_accuracy
    >>> from ssl_attention.data import AnnotatedSubset
    >>>
    >>> dataset = AnnotatedSubset(DATASET_PATH)
    >>> # ... extract attention maps ...
    >>> iou_results = compute_batch_iou(attention_maps, annotations, image_ids)
    >>> print(f"IoU@90: {iou_results[90].mean_iou:.3f}")
    >>>
    >>> accuracy, _ = compute_pointing_accuracy(attention_maps, annotations, image_ids)
    >>> print(f"Pointing accuracy: {accuracy:.1%}")
"""

# IoU and coverage metrics
# Baseline generators
from ssl_attention.metrics.baselines import (
    center_gaussian_baseline,
    compute_baseline_continuous_metrics,
    compute_baseline_ious,
    compute_baseline_pointing,
    random_baseline,
    saliency_prior_baseline,
    sobel_edge_baseline,
)
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_image_emd,
    compute_image_kl,
    compute_image_mse,
    compute_kl_divergence,
    compute_mse,
    emd_support_grid,
    gaussian_bbox_heatmap,
    prepare_bounded_heatmap,
    prepare_emd_distribution,
    prepare_probability_distribution,
    resize_heatmap_for_emd,
    sanitize_nonnegative_heatmap,
    soft_union_heatmap,
)
from ssl_attention.metrics.iou import (
    BatchIoUResult,
    IoUResult,
    aggregate_by_feature_type,
    compute_batch_iou,
    compute_corloc,
    compute_coverage,
    compute_image_iou,
    compute_iou,
    compute_per_bbox_iou,
    threshold_attention,
)

# Pointing game metrics
from ssl_attention.metrics.pointing_game import (
    PointingResult,
    compute_pointing_accuracy,
    compute_top_k_accuracy,
    pointing_game_by_feature,
    pointing_game_hit,
    top_k_pointing_accuracy,
)

# Statistical tests
from ssl_attention.metrics.statistics import (
    ComparisonResult,
    bootstrap_ci,
    cohens_d,
    compare_all_models,
    multiple_comparison_correction,
    paired_comparison,
    paired_ttest,
    rank_models,
    wilcoxon_signed_rank,
)

__all__ = [
    # IoU
    "IoUResult",
    "BatchIoUResult",
    "threshold_attention",
    "compute_iou",
    "compute_coverage",
    "compute_image_iou",
    "compute_mse",
    "compute_image_mse",
    "compute_kl_divergence",
    "compute_image_kl",
    "compute_emd",
    "compute_image_emd",
    "compute_batch_iou",
    "compute_per_bbox_iou",
    "compute_corloc",
    "aggregate_by_feature_type",
    "prepare_bounded_heatmap",
    "prepare_probability_distribution",
    "sanitize_nonnegative_heatmap",
    "resize_heatmap_for_emd",
    "prepare_emd_distribution",
    "emd_support_grid",
    "gaussian_bbox_heatmap",
    "soft_union_heatmap",
    "annotation_to_gaussian_heatmap",
    # Pointing game
    "PointingResult",
    "pointing_game_hit",
    "compute_pointing_accuracy",
    "top_k_pointing_accuracy",
    "compute_top_k_accuracy",
    "pointing_game_by_feature",
    # Baselines
    "random_baseline",
    "center_gaussian_baseline",
    "sobel_edge_baseline",
    "saliency_prior_baseline",
    "compute_baseline_continuous_metrics",
    "compute_baseline_ious",
    "compute_baseline_pointing",
    # Statistics
    "ComparisonResult",
    "paired_ttest",
    "wilcoxon_signed_rank",
    "cohens_d",
    "paired_comparison",
    "bootstrap_ci",
    "multiple_comparison_correction",
    "compare_all_models",
    "rank_models",
]
