#!/usr/bin/env python3
"""Generate the "Layer Progression (All Models)" report figure (Fig 2).

Plots per-layer mean IoU@90 over the 139 annotated images for the six frozen
transformer models, using each model's default extraction method. ResNet-50 is
excluded (CNN baseline, no transformer layers).

Data source:
  outputs/cache/metrics.db -> image_metrics table.
  The schema has no `variant`/`metric` columns: "frozen" == the base model name
  (no `_finetuned_*` suffix) and "iou_at_90" == the `iou` column at
  `percentile = 90`. Per-layer means are AVG(iou) grouped by model x layer.

Model order, colours, and display names come from
`experiments.scripts.figures._palette` so this figure stays consistent with
Tables 4 & 6 and Figures 5 & 6.

Usage:
  uv run python experiments/scripts/generate_layer_progression_figure.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from experiments.scripts.figures._palette import (  # noqa: E402
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
)
from ssl_attention.config import CACHE_PATH  # noqa: E402

METRICS_DB_PATH = CACHE_PATH / "metrics.db"

# Default extraction method per model (matches DEFAULT_METHOD in config and the
# DEFAULT_DB_METHOD map in analyze_model_correlation.py).
DEFAULT_DB_METHOD = {
    "dinov2": "cls",
    "dinov3": "cls",
    "mae": "cls",
    "clip": "cls",
    "siglip": "mean",
    "siglip2": "mean",
}

N_LAYERS = 12  # L0..L11
PERCENTILE = 90


def load_layer_means(
    conn: sqlite3.Connection, model: str, method: str
) -> np.ndarray:
    """Return per-layer mean IoU@90 as an array indexed by layer 0..11.

    Missing layers are NaN so the line breaks rather than drawing a false zero.
    """
    rows = conn.execute(
        """
        SELECT layer, AVG(iou)
        FROM image_metrics
        WHERE model = ? AND method = ? AND percentile = ?
        GROUP BY layer
        """,
        (model, method, PERCENTILE),
    ).fetchall()
    means = np.full(N_LAYERS, np.nan)
    for layer_str, mean_iou in rows:
        idx = int(layer_str.replace("layer", ""))
        if 0 <= idx < N_LAYERS:
            means[idx] = mean_iou
    return means


def generate_figure(output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not METRICS_DB_PATH.exists():
        raise FileNotFoundError(
            f"Metrics DB not found at {METRICS_DB_PATH}. "
            "Run generate_metrics_cache.py first."
        )

    conn = sqlite3.connect(METRICS_DB_PATH)
    try:
        series = {
            model: load_layer_means(conn, model, DEFAULT_DB_METHOD[model])
            for model in MODEL_ORDER
        }
    finally:
        conn.close()

    layers = np.arange(N_LAYERS)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    for model in MODEL_ORDER:
        ax.plot(
            layers,
            series[model],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=MODEL_COLORS[model],
            label=MODEL_DISPLAY_NAMES[model],
        )

    ax.set_xlim(-0.3, N_LAYERS - 0.7)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.set_ylim(0, 0.15)
    ax.set_yticks(np.arange(0, 0.15 + 1e-9, 0.025))

    ax.set_xlabel("Transformer layer", fontsize=11)
    # Y-label dropped one size (11pt) so the long string fits without clipping.
    ax.set_ylabel("Mean IoU@90 across 139 annotated images", fontsize=11)
    # No in-figure title: the report caption carries it.
    ax.tick_params(labelsize=11)

    # Single-row legend below the plot.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(MODEL_ORDER),
        fontsize=11,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" so the below-axes legend is never cropped.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Layer-progression figure saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "docs/final_report/figures/layer_progression_all_models.png"
        ),
        help=(
            "Output PNG path. Defaults to "
            "docs/final_report/figures/layer_progression_all_models.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.output)


if __name__ == "__main__":
    main()
