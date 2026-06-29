#!/usr/bin/env python3
"""Generate the updated paper Figure 6 grouped forest plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_EXPERIMENT_PATH = PROJECT_ROOT / "outputs" / "results" / "active_experiment.json"
Q2_JSON_FALLBACK = PROJECT_ROOT / "outputs" / "results" / "q2_metrics_analysis.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "paper" / "714_fig6.png"

MODEL_ORDER = ["dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"]
MODEL_LABELS = {
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "mae": "MAE",
    "clip": "CLIP",
    "siglip": "SigLIP",
    "siglip2": "SigLIP2",
}

STRATEGIES = ["lora", "full"]
STRATEGY_LABELS = {
    "lora": "LoRA",
    "full": "Full",
}
STRATEGY_MARKERS = {
    "lora": "D",
    "full": "o",
}
STRATEGY_COLORS = {
    "lora": "#2f6fae",
    "full": "#d46f45",
}

METRICS = [
    ("iou", 90, "IoU@90", "higher"),
    ("coverage", None, "Coverage", "higher"),
    ("mse", None, "MSE", "lower"),
    ("kl", None, "KL", "lower"),
    ("emd", None, "EMD", "lower"),
]

TEXT = "#343a40"
GRID = "#d7dce2"
ZERO = "#9aa1a9"
NOTE_BG = "#f7f8fa"


def load_rows() -> list[dict[str, Any]]:
    q2_json = Q2_JSON_FALLBACK
    if ACTIVE_EXPERIMENT_PATH.exists():
        with open(ACTIVE_EXPERIMENT_PATH, encoding="utf-8") as handle:
            active = json.load(handle)
        active_q2_path = active.get("q2_metrics_path")
        if active_q2_path:
            q2_json = PROJECT_ROOT / active_q2_path

    with open(q2_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["rows"])


def lookup(
    rows: list[dict[str, Any]],
    model: str,
    strategy: str,
    metric: str,
    percentile: int | None,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["model_name"] == model
            and row["strategy_id"] == strategy
            and row["metric"] == metric
            and row.get("percentile") == percentile
        ):
            return row
    raise KeyError(f"Missing row for {model}/{strategy}/{metric}/{percentile}")


def sign_normalized(row: dict[str, Any], direction: str) -> tuple[float, float, float]:
    mean = float(row["mean_delta"])
    ci_low = float(row["delta_ci_lower"])
    ci_high = float(row["delta_ci_upper"])
    if direction == "lower":
        return -mean, -ci_high, -ci_low
    return mean, ci_low, ci_high


def nice_xlim(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(span * 0.14, 0.002)
    low = min(low - pad, -pad)
    high = max(high + pad, pad)
    return low, high


def draw_metric_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    percentile: int | None,
    title: str,
    direction: str,
) -> None:
    base_y = np.arange(len(MODEL_ORDER))
    offsets = {"lora": -0.14, "full": 0.14}
    all_x: list[float] = []

    for model_index, model in enumerate(MODEL_ORDER):
        for strategy in STRATEGIES:
            row = lookup(rows, model, strategy, metric, percentile)
            mean, ci_low, ci_high = sign_normalized(row, direction)
            y = base_y[model_index] + offsets[strategy]
            all_x.extend([ci_low, ci_high, mean])

            ax.plot(
                [ci_low, ci_high],
                [y, y],
                color=STRATEGY_COLORS[strategy],
                linewidth=2.2,
                solid_capstyle="round",
                zorder=2,
            )

            ax.scatter(
                mean,
                y,
                marker=STRATEGY_MARKERS[strategy],
                s=60,
                color=STRATEGY_COLORS[strategy],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

            if row.get("significant"):
                star_pad = max(abs(ci_high - ci_low) * 0.18, 0.001)
                ax.text(
                    ci_high + star_pad,
                    y,
                    "*",
                    ha="left",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=TEXT,
                )

    for y in base_y:
        ax.axhline(y + 0.5, color="#eef1f4", linewidth=0.8, zorder=0)

    ax.axvline(0, color=ZERO, linewidth=1.1, linestyle="--", zorder=1)
    display_title = f"{title} ((sign flipped) --> better)" if direction == "lower" else title
    ax.set_title(display_title, fontsize=20, fontweight="bold", color=TEXT, pad=12)
    ax.set_yticks(base_y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=15, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(*nice_xlim(all_x))
    ax.tick_params(axis="x", labelsize=16, colors=TEXT)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def draw_note_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_facecolor(NOTE_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#e1e5ea")
    ax.text(
        0.05,
        0.76,
        "Legend",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        transform=ax.transAxes,
    )
    for idx, strategy in enumerate(STRATEGIES):
        y = 0.60 - idx * 0.16
        ax.scatter(
            0.08,
            y,
            marker=STRATEGY_MARKERS[strategy],
            s=80,
            color=STRATEGY_COLORS[strategy],
            edgecolor="white",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.text(
            0.15,
            y,
            STRATEGY_LABELS[strategy],
            va="center",
            fontsize=15,
            color=TEXT,
            transform=ax.transAxes,
        )
    ax.text(
        0.05,
        0.28,
        "* significant after Holm correction",
        fontsize=15,
        linespacing=1.35,
        color=TEXT,
        transform=ax.transAxes,
    )


def main() -> None:
    rows = load_rows()
    fig, axes = plt.subplots(2, 3, figsize=(21, 13.6), dpi=220)
    axes_flat = axes.flatten()

    for ax, metric_spec in zip(axes_flat[:5], METRICS, strict=True):
        draw_metric_panel(ax, rows, *metric_spec)
    draw_note_panel(axes_flat[5])

    fig.suptitle(
        "Mean Attention-Alignment Change with 95% Bootstrap CI",
        fontsize=24,
        fontweight="bold",
        color=TEXT,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "Mean change vs frozen attention (right = better)",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.08, wspace=0.25, hspace=0.33)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
