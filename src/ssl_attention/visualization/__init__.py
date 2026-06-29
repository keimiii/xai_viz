"""Visualization utilities for SSL attention analysis.

This module provides tools for generating attention heatmaps, overlays,
and statistical comparison charts for the WikiChurches dataset.

Example:
    from ssl_attention.visualization import render_heatmap, create_overlay
    from ssl_attention.attention import attention_to_heatmap

    # Render heatmap with colormap
    heatmap_tensor = attention_to_heatmap(attention, image_size=224)
    heatmap_image = render_heatmap(heatmap_tensor, colormap="viridis")

    # Create overlay with original image
    overlay = create_overlay(original_image, heatmap_image, alpha=0.5)
"""

from ssl_attention.visualization.heatmaps import (
    apply_colormap,
    render_heatmap,
    render_heatmap_batch,
)
from ssl_attention.visualization.overlays import (
    create_attention_overlay,
    create_overlay,
    draw_bboxes,
)
from ssl_attention.visualization.plots import (
    plot_iou_comparison,
    plot_layer_progression,
    plot_model_leaderboard,
    plot_style_breakdown,
)

__all__ = [
    # Heatmaps
    "apply_colormap",
    "render_heatmap",
    "render_heatmap_batch",
    # Overlays
    "create_overlay",
    "draw_bboxes",
    "create_attention_overlay",
    # Plots
    "plot_iou_comparison",
    "plot_layer_progression",
    "plot_model_leaderboard",
    "plot_style_breakdown",
]
