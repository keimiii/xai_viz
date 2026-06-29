"""Statistical visualization plots for attention analysis.

This module provides matplotlib-based charts for comparing model performance,
analyzing layer progression, and visualizing metrics across categories.

Key functions:
- `plot_iou_comparison()`: Compare IoU across models
- `plot_layer_progression()`: Show how attention evolves across layers
- `plot_model_leaderboard()`: Ranked bar chart of model performance
- `plot_style_breakdown()`: IoU by architectural style
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# Use non-interactive backend
matplotlib.use("Agg")

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGURE_DPI = 100
DEFAULT_FIGSIZE = (10, 6)


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    from PIL import Image as PILImage

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf)
    return img.copy()  # Copy so we can close buffer


def plot_iou_comparison(
    model_names: list[str],
    iou_scores: dict[str, list[float]],
    percentiles: list[int] | None = None,
    title: str = "IoU Comparison Across Models",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Image.Image:
    """Create bar chart comparing IoU scores across models.

    Args:
        model_names: List of model names.
        iou_scores: Dict mapping model name to list of IoU scores at different percentiles.
        percentiles: List of percentile thresholds (e.g., [90, 80, 70]).
        title: Chart title.
        figsize: Figure size as (width, height).

    Returns:
        PIL Image of the chart.

    Example:
        >>> scores = {"dinov2": [0.65, 0.55, 0.45], "clip": [0.60, 0.50, 0.40]}
        >>> img = plot_iou_comparison(["dinov2", "clip"], scores, [90, 80, 70])
    """
    if percentiles is None:
        percentiles = [90, 80, 70, 60, 50]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(model_names))
    width = 0.8 / len(percentiles)

    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0.2, 0.8, len(percentiles)))

    for i, percentile in enumerate(percentiles):
        values = [iou_scores[model][i] if model in iou_scores else 0 for model in model_names]
        offset = width * (i - len(percentiles) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=f"Top {100-percentile}%", color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values, strict=True):
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("IoU Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(title="Attention Threshold")
    ax.set_ylim(0, min(1.0, max(max(v) for v in iou_scores.values()) * 1.2))

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_layer_progression(
    layer_ious: list[float],
    model_name: str = "Model",
    percentile: int = 90,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> Image.Image:
    """Plot IoU evolution across transformer layers.

    Args:
        layer_ious: IoU score for each layer (length = num_layers).
        model_name: Name of the model for legend.
        percentile: Percentile threshold used.
        title: Chart title. If None, auto-generated.
        figsize: Figure size.

    Returns:
        PIL Image of the line chart.

    Example:
        >>> ious = [0.2, 0.3, 0.4, 0.5, 0.55, 0.58, 0.6, 0.62, 0.63, 0.64, 0.65, 0.66]
        >>> img = plot_layer_progression(ious, model_name="DINOv2")
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = list(range(len(layer_ious)))

    ax.plot(layers, layer_ious, marker="o", linewidth=2, markersize=6, label=model_name)

    # Fill area under curve
    ax.fill_between(layers, layer_ious, alpha=0.3)

    # Mark the best layer
    best_layer = int(np.argmax(layer_ious))
    best_iou = layer_ious[best_layer]
    ax.annotate(
        f"Best: L{best_layer}\n{best_iou:.3f}",
        xy=(float(best_layer), best_iou),
        xytext=(float(best_layer) + 1, best_iou + 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"IoU @ {percentile}th percentile")
    ax.set_title(title or f"Attention-Annotation Alignment Across Layers ({model_name})")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.set_ylim(0, min(1.0, max(layer_ious) * 1.2))
    ax.legend()

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_multi_model_layer_progression(
    model_layer_ious: dict[str, list[float]],
    percentile: int = 90,
    title: str = "Layer Progression Across Models",
    figsize: tuple[float, float] = (12, 6),
) -> Image.Image:
    """Plot layer progression for multiple models on same axes.

    Args:
        model_layer_ious: Dict mapping model name to list of per-layer IoUs.
        percentile: Percentile threshold used.
        title: Chart title.
        figsize: Figure size.

    Returns:
        PIL Image with overlaid line charts.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v"]

    for i, (model_name, layer_ious) in enumerate(model_layer_ious.items()):
        layers = list(range(len(layer_ious)))
        ax.plot(
            layers,
            layer_ious,
            marker=markers[i % len(markers)],
            color=cmap(i % 10),
            linewidth=2,
            markersize=6,
            label=model_name,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"IoU @ {percentile}th percentile")
    ax.set_title(title)

    # Set x-ticks to layer indices
    max_layers = max(len(v) for v in model_layer_ious.values())
    ax.set_xticks(range(max_layers))
    ax.set_xticklabels([f"L{i}" for i in range(max_layers)])

    ax.legend(loc="lower right")

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_model_leaderboard(
    model_scores: dict[str, float],
    metric_name: str = "Mean IoU",
    title: str = "Model Leaderboard",
    figsize: tuple[float, float] = (8, 5),
    show_values: bool = True,
) -> Image.Image:
    """Create horizontal bar chart ranking models by score.

    Args:
        model_scores: Dict mapping model name to score.
        metric_name: Name of the metric for x-axis label.
        title: Chart title.
        figsize: Figure size.
        show_values: Whether to show value labels on bars.

    Returns:
        PIL Image of the leaderboard chart.

    Example:
        >>> scores = {"dinov2": 0.65, "clip": 0.58, "mae": 0.55}
        >>> img = plot_model_leaderboard(scores)
    """
    # Sort by score descending
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    models = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    fig, ax = plt.subplots(figsize=figsize)

    # Color gradient based on rank
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0.8, 0.3, len(models)))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, scores, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # Top model at top
    ax.set_xlabel(metric_name)
    ax.set_title(title)

    # Add value labels
    if show_values:
        for bar, score in zip(bars, scores, strict=True):
            ax.annotate(
                f"{score:.3f}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
            )

    # Add rank indicators
    for i, _ in enumerate(models[:3]):
        rank = ["#1", "#2", "#3"][i]
        color = ["gold", "silver", "#CD7F32"][i]  # Gold, silver, bronze
        ax.annotate(
            rank,
            xy=(0, i),
            xytext=(-25, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(0, max(scores) * 1.15)

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_style_breakdown(
    style_ious: dict[str, float],
    style_counts: dict[str, int] | None = None,
    title: str = "IoU by Architectural Style",
    figsize: tuple[float, float] = (8, 5),
) -> Image.Image:
    """Create bar chart showing IoU breakdown by architectural style.

    Args:
        style_ious: Dict mapping style name to mean IoU.
        style_counts: Optional dict with count of images per style.
        title: Chart title.
        figsize: Figure size.

    Returns:
        PIL Image of the breakdown chart.

    Example:
        >>> styles = {"Romanesque": 0.58, "Gothic": 0.65, "Renaissance": 0.52, "Baroque": 0.48}
        >>> img = plot_style_breakdown(styles)
    """
    fig, ax = plt.subplots(figsize=figsize)

    styles = list(style_ious.keys())
    ious = list(style_ious.values())

    # Color by IoU value
    norm = plt.Normalize(vmin=min(ious) * 0.9, vmax=max(ious) * 1.1)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = cmap(norm(ious))

    bars = ax.bar(styles, ious, color=colors)

    # Add value labels
    for bar, iou in zip(bars, ious, strict=True):
        ax.annotate(
            f"{iou:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add counts if provided
    if style_counts:
        for bar, style in zip(bars, styles, strict=True):
            count = style_counts.get(style, 0)
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
            )

    ax.set_xlabel("Architectural Style")
    ax.set_ylabel("Mean IoU")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(ious) * 1.2))

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_feature_breakdown(
    feature_ious: dict[str, float],
    top_n: int = 10,
    title: str = "IoU by Feature Type",
    figsize: tuple[float, float] = (10, 6),
) -> Image.Image:
    """Create horizontal bar chart showing IoU by architectural feature type.

    Args:
        feature_ious: Dict mapping feature name to mean IoU.
        top_n: Number of top features to show.
        title: Chart title.
        figsize: Figure size.

    Returns:
        PIL Image of the feature breakdown chart.
    """
    # Sort and take top N
    sorted_features = sorted(feature_ious.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    ious = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(features))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0.8, 0.3, len(features)))

    bars = ax.barh(y_pos, ious, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean IoU")
    ax.set_title(title)

    # Add value labels
    for bar, iou in zip(bars, ious, strict=True):
        ax.annotate(
            f"{iou:.3f}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    ax.set_xlim(0, min(1.0, max(ious) * 1.15))

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def plot_coverage_vs_iou(
    coverages: list[float],
    ious: list[float],
    labels: list[str] | None = None,
    title: str = "Coverage vs IoU",
    figsize: tuple[float, float] = (8, 6),
) -> Image.Image:
    """Create scatter plot comparing coverage and IoU metrics.

    Args:
        coverages: List of coverage scores.
        ious: List of IoU scores.
        labels: Optional labels for each point.
        title: Chart title.
        figsize: Figure size.

    Returns:
        PIL Image of the scatter plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(coverages, ious, alpha=0.6, s=50)

    # Add diagonal reference line
    lims = [0, 1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="x=y")

    # Add labels if provided
    if labels:
        for cov, iou, label in zip(coverages, ious, labels, strict=True):
            ax.annotate(label, (cov, iou), fontsize=8, alpha=0.7)

    ax.set_xlabel("Coverage (Energy)")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img
