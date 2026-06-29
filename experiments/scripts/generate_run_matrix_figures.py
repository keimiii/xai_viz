#!/usr/bin/env python3
"""Generate visualizations for the fine-tuning run matrix.

Produces 9 publication-quality figures from Q2 metric deltas and hardcoded
run matrix data, saved to outputs/figures/.

Design principles (Tufte + Datawrapper best practices):
  - Maximize data-ink ratio: minimal gridlines, no chartjunk
  - Fewer colors: gray for context, color only for what matters
  - Muted, colorblind-safe palette (Tableau-inspired)
  - Direct labeling over legends where practical
  - Consistent typography and spacing

Usage:
    python experiments/scripts/generate_run_matrix_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    resolve_active_artifact_path,
)

Q2_JSON = resolve_active_artifact_path(
    "q2_metrics_path",
    PROJECT_ROOT / "outputs" / "results" / "q2_metrics_analysis.json",
)
RUN_MATRIX_JSON = resolve_active_artifact_path(
    "run_matrix_path",
    PROJECT_ROOT / "outputs" / "results" / "run_matrix.json",
)
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ---------------------------------------------------------------------------
# Design system — muted, colorblind-safe palette
# ---------------------------------------------------------------------------
MODELS = ["clip", "dinov2", "dinov3", "mae", "siglip", "siglip2"]
MODEL_LABELS = {
    "clip": "CLIP",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "mae": "MAE",
    "siglip": "SigLIP",
    "siglip2": "SigLIP 2",
}
STRATEGIES = ["linear_probe", "lora", "full"]
STRATEGY_LABELS = {
    "linear_probe": "Linear Probe",
    "lora": "LoRA",
    "full": "Full",
}

# Muted Tableau-inspired strategy palette
STRATEGY_COLORS = {
    "linear_probe": "#93b7be",  # muted teal (de-emphasized — baseline strategy)
    "lora": "#4e79a7",          # steel blue
    "full": "#d4764e",          # warm terracotta
}
FROZEN_COLOR = "#8a817c"        # darker warm gray for frozen baselines
TEXT_COLOR = "#4a4a4a"          # dark gray for text (softer than black)
GRID_COLOR = "#cccccc"          # medium-light gray for axes/spines
IMPROVED_COLOR = "#3a7d44"      # forest green (muted)
DEGRADED_COLOR = "#c04e4e"      # muted red
SIG_COLOR = "#2b2b2b"           # near-black for significance markers

HEATMAP_METRICS = [
    ("iou", 90, "IoU@90"),
    ("iou", 50, "IoU@50"),
    ("coverage", None, "Coverage"),
    ("mse", None, "MSE"),
    ("kl", None, "KL"),
    ("emd", None, "EMD"),
]

METRIC_DIRECTION = {
    "iou": "higher",
    "coverage": "higher",
    "mse": "lower",
    "kl": "lower",
    "emd": "lower",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_q2_data() -> list[dict[str, Any]]:
    """Load Q2 metrics analysis JSON and return the rows list."""
    with open(Q2_JSON) as f:
        data = json.load(f)
    return cast(list[dict[str, Any]], data["rows"])


def load_run_matrix() -> dict[tuple[str, str], dict[str, Any]]:
    """Load run-matrix entries keyed by (model, strategy)."""
    with open(RUN_MATRIX_JSON, encoding="utf-8") as handle:
        payload = json.load(handle)

    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for run in payload.get("runs", {}).values():
        if run.get("run_scope") != "primary":
            continue
        score = float(run.get("best_val_score", 0.0))
        matrix[(run["model"], run["strategy"])] = {
            "val_acc": score * 100 if score <= 1.0 else score,
            "best_epoch": int(run.get("selected_epoch", 0)),
        }
    return matrix


RUN_MATRIX = load_run_matrix()


def lookup_row(
    rows: list[dict],
    model: str,
    strategy: str,
    metric: str,
    percentile: int | None = None,
) -> dict | None:
    """Find a specific row by (model, strategy, metric, percentile)."""
    for r in rows:
        if (
            r["model_name"] == model
            and r["strategy_id"] == strategy
            and r["metric"] == metric
            and r.get("percentile") == percentile
        ):
            return r
    return None


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """Set a clean, minimal style inspired by Tufte + modern dataviz."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "text.color": TEXT_COLOR,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def clean_axes(ax: plt.Axes) -> None:
    """Apply minimal Tufte-inspired cleanup to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(False)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure to the figures directory."""
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: Grouped Bar Chart — Validation Accuracy
# ---------------------------------------------------------------------------
def fig_validation_accuracy() -> str:
    """Grouped bar chart of validation accuracy per model x strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODELS))
    width = 0.24

    for i, strat in enumerate(STRATEGIES):
        vals = [RUN_MATRIX[(m, strat)]["val_acc"] for m in MODELS]
        bars = ax.bar(
            x + i * width,
            vals,
            width * 0.9,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            edgecolor="white",
            linewidth=0.3,
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=7, color=TEXT_COLOR,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy by Model and Fine-Tuning Strategy")
    ax.set_ylim(50, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    clean_axes(ax)

    save_figure(fig, "01_val_accuracy_by_model_strategy.png")

    best_overall = max(RUN_MATRIX.items(), key=lambda kv: kv[1]["val_acc"])
    worst_overall = min(RUN_MATRIX.items(), key=lambda kv: kv[1]["val_acc"])
    return (
        f"Best: {MODEL_LABELS[best_overall[0][0]]} {STRATEGY_LABELS[best_overall[0][1]]} "
        f"({best_overall[1]['val_acc']:.1f}%). "
        f"Worst: {MODEL_LABELS[worst_overall[0][0]]} {STRATEGY_LABELS[worst_overall[0][1]]} "
        f"({worst_overall[1]['val_acc']:.1f}%). MAE consistently lags; DINOv3 LoRA "
        "leads. Full fine-tuning does not always beat LoRA on classification accuracy. "
        "The practical takeaway is that stronger adaptation helps, but lightweight "
        "LoRA often captures most of the downstream gain without needing full backbone "
        "updates."
    )


# ---------------------------------------------------------------------------
# Figure 2: Delta Heatmap Grid
# ---------------------------------------------------------------------------
def fig_delta_heatmap(rows: list[dict]) -> str:
    """2x3 grid of annotated heatmaps — improvement direction normalized."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]

        matrix = np.zeros((len(MODELS), len(STRATEGIES)))
        sig_mask = np.zeros_like(matrix, dtype=bool)
        for mi, m in enumerate(MODELS):
            for si, s in enumerate(STRATEGIES):
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    delta = row["mean_delta"]
                    matrix[mi, si] = delta if direction == "higher" else -delta
                    sig_mask[mi, si] = row.get("significant", False)

        vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.01

        # Custom diverging colormap: muted blue-white-red
        sns.heatmap(
            matrix, ax=ax,
            cmap="RdBu", center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt="+.3f",
            annot_kws={"size": 8, "color": TEXT_COLOR},
            linewidths=1, linecolor="white",
            xticklabels=[STRATEGY_LABELS[s] for s in STRATEGIES],
            yticklabels=[MODEL_LABELS[m] for m in MODELS] if idx % 3 == 0 else False,
            cbar_kws={"shrink": 0.7, "aspect": 15},
        )

        for mi in range(len(MODELS)):
            for si in range(len(STRATEGIES)):
                if sig_mask[mi, si]:
                    ax.text(
                        si + 0.5, mi + 0.82, "*",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=SIG_COLOR,
                    )

        hint = "higher=better" if direction == "higher" else "sign flipped"
        ax.set_title(f"{title}  ({hint})", fontsize=10)

    fig.suptitle(
        "Improvement Heatmaps  |  Blue = improved, Red = degraded  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "02_all_metrics_improvement_heatmap.png")

    return (
        "All deltas normalized to improvement direction: positive (blue) = better, "
        "negative (red) = worse. For MSE/KL/EMD the sign is flipped so blue still "
        "means the metric decreased (improved). Linear probe near-zero confirms frozen "
        "backbones don't shift attention. SigLIP shows the most significant "
        "improvements. Read the blue clusters less as \"every metric moved\" and more "
        "as \"this model genuinely reallocated attention in a better direction across "
        "multiple views.\""
    )


# ---------------------------------------------------------------------------
# Figure 3: Diverging Bar Chart — Per-Metric Deep Dive
# ---------------------------------------------------------------------------
def fig_diverging_bars(rows: list[dict]) -> str:
    """2x3 faceted horizontal diverging bars, sorted descending, colored by model."""
    active_strategies = ["lora", "full"]

    # One color per model (consistent across subplots)
    model_colors = {
        "clip": "#4e79a7",
        "dinov2": "#f28e2b",
        "dinov3": "#59a14f",
        "mae": "#e15759",
        "siglip": "#76b7b2",
        "siglip2": "#b07aa1",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]

        # Collect entries: (label, delta, sig, model_key)
        entries = []
        for m in MODELS:
            for s in active_strategies:
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    entries.append({
                        "label": f"{MODEL_LABELS[m]} — {STRATEGY_LABELS[s]}",
                        "delta": row["mean_delta"],
                        "sig": row.get("significant", False),
                        "model": m,
                    })

        # Sort: best performers at top (direction-aware)
        if direction == "higher":
            entries.sort(key=lambda e: e["delta"], reverse=True)
        else:
            entries.sort(key=lambda e: e["delta"], reverse=False)

        labels = [e["label"] for e in entries]
        deltas = np.array([e["delta"] for e in entries])
        colors = [model_colors[e["model"]] for e in entries]
        y_pos = np.arange(len(entries))

        ax.barh(y_pos, deltas, color=colors, height=0.6, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()  # best performer (index 0) at top
        ax.axvline(0, color=TEXT_COLOR, linewidth=0.6)
        hint = "(higher=better)" if direction == "higher" else "(lower=better)"
        ax.set_title(f"{title}  {hint}", fontsize=10)
        clean_axes(ax)

        # Significance markers
        for i, e in enumerate(entries):
            if e["sig"]:
                d = e["delta"]
                offset = abs(d) * 0.1 + 0.0005
                x_pos = d + offset if d >= 0 else d - offset
                ax.text(
                    x_pos, i, "*",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=SIG_COLOR,
                )

    # Model color legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=model_colors[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=6, fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Per-Metric Deltas: LoRA & Full, best at top  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "03_all_metrics_diverging_bars.png")

    return (
        "Bars sorted by delta (largest improvement at top). Colors represent models, "
        "making it easy to spot which models consistently rank high or low. "
        "Linear probe omitted (all near zero). SigLIP entries cluster near the top "
        "across most metrics. The ranking makes the model hierarchy easier to trust: "
        "improvement is not evenly distributed, and CLIP/SigLIP benefit much more "
        "than the DINO family here."
    )


# ---------------------------------------------------------------------------
# Figure 4: IoU Percentile Slope Chart
# ---------------------------------------------------------------------------
def fig_iou_percentile_slopes(rows: list[dict]) -> str:
    """2x3 grid of line charts showing IoU delta across percentiles."""
    percentiles = [50, 60, 70, 80, 90]
    active_strategies = ["lora", "full"]
    line_styles = {"lora": "--", "full": "-"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        for strat in active_strategies:
            deltas, sigs = [], []
            for p in percentiles:
                row = lookup_row(rows, model, strat, "iou", p)
                if row:
                    deltas.append(row["mean_delta"])
                    sigs.append(row.get("significant", False))
                else:
                    deltas.append(0.0)
                    sigs.append(False)

            ax.plot(
                percentiles, deltas,
                marker="o", linestyle=line_styles[strat],
                color=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat],
                linewidth=1.8, markersize=4,
            )

            for p, d, sig in zip(percentiles, deltas, sigs, strict=True):
                if sig:
                    ax.annotate(
                        "*", (p, d),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=10,
                        color=SIG_COLOR, fontweight="bold",
                    )

        ax.axhline(0, color=FROZEN_COLOR, linewidth=0.8, linestyle="-")

        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("IoU Delta")
        ax.set_xticks(percentiles)
        clean_axes(ax)

    fig.legend(
        [STRATEGY_LABELS[s] for s in active_strategies],
        loc="upper center", ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "IoU Delta Across Percentile Thresholds  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "04_iou_delta_by_percentile.png")

    return (
        "Slopes reveal how fine-tuning affects top-k attention alignment at different "
        "thresholds. Steeper slopes at high percentiles indicate that fine-tuning "
        "particularly shifts the most-attended regions. Flat slopes near zero confirm "
        "linear probe has no effect (omitted). The interesting pattern is that gains "
        "are often strongest where attention is most concentrated, which suggests "
        "sharpening around diagnostic parts rather than a uniform boost everywhere."
    )


# ---------------------------------------------------------------------------
# Figure 5: Radar Chart — Multi-Metric Model Profiles
# ---------------------------------------------------------------------------
def fig_radar_profiles(rows: list[dict]) -> str:
    """2x3 grid of radar charts showing normalized metric profiles."""
    radar_metrics = [
        ("iou", 90, "IoU@90"),
        ("coverage", None, "Coverage"),
        ("mse", None, "MSE"),
        ("kl", None, "KL"),
        ("emd", None, "EMD"),
    ]
    n_metrics = len(radar_metrics)

    all_vals: dict[str, list[float]] = {m[0] + str(m[1]): [] for m in radar_metrics}
    for m in MODELS:
        for s in STRATEGIES:
            for metric, pctl, _ in radar_metrics:
                key = metric + str(pctl)
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    all_vals[key].append(row["frozen_mean"])
                    all_vals[key].append(row["finetuned_mean"])

    bounds: dict[str, tuple[float, float]] = {}
    for key, vals in all_vals.items():
        bounds[key] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    def normalize(value: float, metric: str, pctl: int | None) -> float:
        key = metric + str(pctl)
        lo, hi = bounds[key]
        if hi == lo:
            return 0.5
        normed = (value - lo) / (hi - lo)
        if METRIC_DIRECTION[metric] == "lower":
            normed = 1.0 - normed
        return normed

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    strat_styles = {
        "lora": {"color": STRATEGY_COLORS["lora"], "linestyle": "--", "linewidth": 1.8},
        "full": {"color": STRATEGY_COLORS["full"], "linestyle": "-", "linewidth": 1.8},
    }

    for mi, model in enumerate(MODELS):
        ax = axes[mi]

        # Frozen baseline
        frozen_vals = []
        for metric, pctl, _ in radar_metrics:
            row = lookup_row(rows, model, "linear_probe", metric, pctl)
            frozen_vals.append(normalize(row["frozen_mean"], metric, pctl) if row else 0.0)
        frozen_vals += frozen_vals[:1]
        ax.plot(angles, frozen_vals, color=FROZEN_COLOR, linewidth=1.8, linestyle="--", label="Frozen")
        ax.fill(angles, frozen_vals, color=FROZEN_COLOR, alpha=0.08)

        # LoRA and Full
        for strat in ["lora", "full"]:
            vals = []
            for metric, pctl, _ in radar_metrics:
                row = lookup_row(rows, model, strat, metric, pctl)
                vals.append(normalize(row["finetuned_mean"], metric, pctl) if row else 0.0)
            vals += vals[:1]
            ax.plot(angles, vals, label=STRATEGY_LABELS[strat], **strat_styles[strat])
            ax.fill(angles, vals, color=strat_styles[strat]["color"], alpha=0.06)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([rm[2] for rm in radar_metrics], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(MODEL_LABELS[model], fontsize=10, pad=15)
        ax.tick_params(axis="y", labelsize=6, labelcolor=FROZEN_COLOR)
        if mi == 0:
            ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.5, 1.15))

    fig.suptitle(
        "Multi-Metric Radar Profiles (outward = better, normalized [0,1])",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "05_iou_coverage_mse_kl_emd_radar.png")

    return (
        "Radar charts show each model's metric profile normalized globally. "
        "Outward = better for all axes (lower-is-better metrics are inverted). "
        "Frozen baseline in gray; fine-tuned strategies overlaid. "
        "Models where the colored polygon expands beyond gray show genuine "
        "improvement. Use this chart to judge breadth, not raw size: outward "
        "expansion across several axes means the gain is multi-metric rather than a "
        "single-metric artifact."
    )


# ---------------------------------------------------------------------------
# Figure 6: Faceted Scatter — Val Accuracy vs Attention Delta
# ---------------------------------------------------------------------------
def fig_accuracy_vs_attention(rows: list[dict]) -> str:
    """Faceted scatter (2x3): one panel per model."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    strategy_markers = {"linear_probe": "s", "lora": "D", "full": "o"}

    all_x: list[float] = []
    for m in MODELS:
        for strat in STRATEGIES:
            all_x.append(RUN_MATRIX[(m, strat)]["val_acc"])
    x_pad = (max(all_x) - min(all_x)) * 0.15

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        xs, ys = [], []
        for strat in STRATEGIES:
            val_acc = RUN_MATRIX[(model, strat)]["val_acc"]
            row = lookup_row(rows, model, strat, "iou", 90)
            delta = row["mean_delta"] if row else 0.0
            xs.append(val_acc)
            ys.append(delta)

        for i, strat in enumerate(STRATEGIES):
            ax.scatter(
                xs[i], ys[i],
                c=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat] if mi == 0 else None,
                s=90, edgecolors="white", linewidth=0.5, zorder=4,
                marker=strategy_markers[strat],
            )
            ax.annotate(
                STRATEGY_LABELS[strat],
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(7, -10 if i == 0 else 7),
                fontsize=7, color=STRATEGY_COLORS[strat],
            )

        ax.axhline(0, color=FROZEN_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 0.005
        y_pad = y_range * 0.4
        y_lo = min(min(ys) - y_pad, -y_pad * 0.3)
        y_hi = max(max(ys) + y_pad, y_pad * 0.3)
        ax.set_ylim(y_lo, y_hi)
        clean_axes(ax)
        if mi >= 3:
            ax.set_xlabel("Val Accuracy (%)")
        if mi % 3 == 0:
            ax.set_ylabel("IoU@90 Delta")

    fig.legend(
        [STRATEGY_LABELS[s] for s in STRATEGIES],
        loc="upper center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Classification Accuracy vs Attention Alignment Change",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "06_val_accuracy_vs_iou90_delta.png")

    return (
        "Each panel shows one model's trajectory from Linear Probe (square) through "
        "LoRA (diamond) to Full (circle). Shared x-axis enables accuracy comparison, "
        "while panel-specific y-axes keep small IoU changes legible. "
        "Points above the dashed y=0 line improved attention alignment. "
        "Models like CLIP/SigLIP show clear upward trajectories; DINOv2/DINOv3 stay "
        "flat. The main read is that better classification and more expert-like "
        "attention often travel together, but not tightly enough to treat one as a "
        "substitute for the other."
    )


# ---------------------------------------------------------------------------
# Figure 7: Preserve / Enhance / Destroy Classification Matrix
# ---------------------------------------------------------------------------
CLASSIFY_STRATEGIES = ["lora", "full"]  # LP omitted (always Preserve)

# Colors for the three outcome categories
PED_COLORS = {
    "Enhance": IMPROVED_COLOR,
    "Preserve": FROZEN_COLOR,
    "Destroy": DEGRADED_COLOR,
}


def _classify_ped(row: dict | None) -> str:
    """Classify a row into Preserve / Enhance / Destroy."""
    if row is None or not row.get("significant"):
        return "Preserve"
    direction = METRIC_DIRECTION[row["metric"]]
    delta = row["mean_delta"]
    improved = (direction == "higher" and delta > 0) or (
        direction == "lower" and delta < 0
    )
    return "Enhance" if improved else "Destroy"


def fig_preserve_enhance_destroy(rows: list[dict]) -> str:
    """6-metric panel: each metric shows a models × strategies grid colored by
    Preserve / Enhance / Destroy classification based on significance + delta
    direction.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        n_models = len(MODELS)
        n_strats = len(CLASSIFY_STRATEGIES)

        # Build classification matrix
        cell_colors = np.zeros((n_models, n_strats, 3))
        cell_labels: list[list[str]] = []
        cell_deltas: list[list[str]] = []

        for mi, m in enumerate(MODELS):
            row_labels: list[str] = []
            row_deltas: list[str] = []
            for si, strat in enumerate(CLASSIFY_STRATEGIES):
                row = lookup_row(rows, m, strat, metric, pctl)
                category = _classify_ped(row)
                # Convert hex to RGB
                hex_c = PED_COLORS[category]
                rgb = tuple(int(hex_c[i:i+2], 16) / 255.0 for i in (1, 3, 5))
                cell_colors[mi, si] = rgb
                row_labels.append(category[0])  # E, P, or D

                if row:
                    direction = METRIC_DIRECTION[metric]
                    d = row["mean_delta"]
                    # Normalize: positive = improvement for display
                    display_d = d if direction == "higher" else -d
                    row_deltas.append(f"{display_d:+.3f}")
                else:
                    row_deltas.append("")
            cell_labels.append(row_labels)
            cell_deltas.append(row_deltas)

        # Draw cells manually
        for mi in range(n_models):
            for si in range(n_strats):
                color = cell_colors[mi, si]
                # Lighter fill (blend with white)
                fill = tuple(0.3 * c + 0.7 for c in color)
                rect = plt.Rectangle(
                    (si, mi), 1, 1,
                    facecolor=fill, edgecolor="white", linewidth=2,
                )
                ax.add_patch(rect)

                # Category letter
                cat_letter = cell_labels[mi][si]
                ax.text(
                    si + 0.5, mi + 0.38, cat_letter,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=tuple(cell_colors[mi, si]),
                )
                # Delta value
                ax.text(
                    si + 0.5, mi + 0.68, cell_deltas[mi][si],
                    ha="center", va="center",
                    fontsize=8, color=TEXT_COLOR,
                )

        ax.set_xlim(0, n_strats)
        ax.set_ylim(0, n_models)
        ax.invert_yaxis()
        ax.set_xticks([s + 0.5 for s in range(n_strats)])
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in CLASSIFY_STRATEGIES])
        ax.set_yticks([m + 0.5 for m in range(n_models)])
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in MODELS] if idx % 3 == 0 else [],
        )

        direction = METRIC_DIRECTION[metric]
        hint = "higher=better" if direction == "higher" else "lower=better"
        ax.set_title(f"{title}  ({hint})", fontsize=10)

        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for cat in ["Enhance", "Preserve", "Destroy"]:
        rgb = tuple(int(PED_COLORS[cat][i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        fill = cast(tuple[float, float, float], tuple(0.3 * c + 0.7 for c in rgb))
        legend_elements.append(
            Patch(
                facecolor=fill,
                edgecolor=PED_COLORS[cat],
                linewidth=1.5,
                label=cat,
            )
        )
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=3,
        fontsize=10, bbox_to_anchor=(0.5, 1.02), columnspacing=3,
    )
    fig.suptitle(
        "Preserve / Enhance / Destroy Classification by Metric",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.06,
    )

    fig.tight_layout()
    save_figure(fig, "07_preserve_enhance_destroy.png")

    # Count categories across all metrics
    counts = {"Enhance": 0, "Preserve": 0, "Destroy": 0}
    for metric, pctl, _ in HEATMAP_METRICS:
        for m in MODELS:
            for strat in CLASSIFY_STRATEGIES:
                row = lookup_row(rows, m, strat, metric, pctl)
                counts[_classify_ped(row)] += 1
    total = sum(counts.values())

    return (
        f"Across all 6 metrics: {counts['Enhance']} Enhance, "
        f"{counts['Preserve']} Preserve, {counts['Destroy']} Destroy "
        f"(out of {total} model-strategy-metric combinations). Linear Probe omitted "
        "(frozen backbone = always Preserve). The encouraging part is that "
        "improvement is the dominant outcome; the caution is that regressions are "
        "still common enough that strategy choice matters."
    )


# ---------------------------------------------------------------------------
# Figure 8: Forest Plot with 95% Confidence Intervals
# ---------------------------------------------------------------------------
def fig_forest_plot_ci(rows: list[dict]) -> str:
    """2x3 faceted forest plot: point estimate + 95% CI whisker per model×strategy.

    Each panel = one metric. Rows = 12 (6 models × 2 strategies), color by
    strategy. Vertical dashed line at zero. Significance star on significant CIs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    strats = CLASSIFY_STRATEGIES  # LoRA + Full only
    y_labels = []
    for m in MODELS:
        for strat in strats:
            y_labels.append(f"{MODEL_LABELS[m]} — {STRATEGY_LABELS[strat]}")

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]
        y_pos = np.arange(len(y_labels))

        means = []
        ci_lows = []
        ci_highs = []
        colors = []
        sigs = []
        markers = []

        for m in MODELS:
            for strat in strats:
                row = lookup_row(rows, m, strat, metric, pctl)
                if row:
                    d = row["mean_delta"]
                    ci_lo = row.get("delta_ci_lower", d)
                    ci_hi = row.get("delta_ci_upper", d)
                    # Flip sign for lower-is-better so positive = improvement
                    if direction == "lower":
                        d, ci_lo, ci_hi = -d, -ci_hi, -ci_lo
                    means.append(d)
                    ci_lows.append(ci_lo)
                    ci_highs.append(ci_hi)
                    sigs.append(row.get("significant", False))
                else:
                    means.append(0.0)
                    ci_lows.append(0.0)
                    ci_highs.append(0.0)
                    sigs.append(False)
                colors.append(STRATEGY_COLORS[strat])
                markers.append("D" if strat == "lora" else "o")

        means_arr = np.array(means)
        ci_lows_arr = np.array(ci_lows)
        ci_highs_arr = np.array(ci_highs)

        # CI whiskers
        for i in range(len(y_pos)):
            ax.plot(
                [ci_lows_arr[i], ci_highs_arr[i]], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=1.5, solid_capstyle="round",
            )

        # Point estimates
        for i in range(len(y_pos)):
            ax.scatter(
                means_arr[i], y_pos[i],
                c=colors[i], marker=markers[i], s=45, zorder=5,
                edgecolors="white", linewidth=0.3,
            )

        # Significance stars
        for i in range(len(y_pos)):
            if sigs[i]:
                ax.text(
                    ci_highs_arr[i] + abs(ci_highs_arr[i] - ci_lows_arr[i]) * 0.15 + 0.0003,
                    y_pos[i], "*",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=SIG_COLOR,
                )

        # Zero line
        ax.axvline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--", zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels if idx % 3 == 0 else [], fontsize=8)
        hint = "→ better" if direction == "higher" else "(sign flipped) → better"
        ax.set_title(f"{title}  ({hint})", fontsize=10)
        ax.invert_yaxis()
        clean_axes(ax)

    # Strategy legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=STRATEGY_COLORS["lora"],
               markersize=7, label="LoRA"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STRATEGY_COLORS["full"],
               markersize=7, label="Full"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2,
        fontsize=10, bbox_to_anchor=(0.5, 1.02), columnspacing=3,
    )
    fig.suptitle(
        "Mean Δ with 95% Bootstrap CI  (* = significant after Holm correction)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )

    fig.tight_layout()
    save_figure(fig, "08_forest_plot_ci.png")

    # Count how many displayed bootstrap CIs exclude zero.
    n_ci_exclude_zero = sum(
        1 for (metric, pctl, _) in HEATMAP_METRICS
        for m in MODELS for strat in strats
        if (r := lookup_row(rows, m, strat, metric, pctl))
        and (
            (r.get("delta_ci_lower", 0.0) > 0 and r.get("delta_ci_upper", 0.0) > 0)
            or (r.get("delta_ci_lower", 0.0) < 0 and r.get("delta_ci_upper", 0.0) < 0)
        )
    )
    n_significant = sum(
        1 for (metric, pctl, _) in HEATMAP_METRICS
        for m in MODELS for strat in strats
        if (r := lookup_row(rows, m, strat, metric, pctl))
        and r.get("significant", False)
    )
    return (
        f"Forest plot with 95% bootstrap CIs for LoRA and Full across all 6 metrics. "
        f"{n_ci_exclude_zero} of {len(MODELS) * len(strats) * len(HEATMAP_METRICS)} "
        f"displayed CIs exclude zero, and {n_significant} of "
        f"{len(MODELS) * len(strats) * len(HEATMAP_METRICS)} rows remain significant "
        f"after Holm correction. Sign flipped for lower-is-better metrics so rightward "
        f"always means improvement. This is the clearest \"signal vs noise\" figure: "
        f"when CLIP and SigLIP sit clearly to the right of zero across multiple "
        f"panels, the improvement looks robust rather than anecdotal."
    )


# ---------------------------------------------------------------------------
# Figure 9: Per-Image Delta Strip Plot (Full fine-tuning only)
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "clip": "#4e79a7",
    "dinov2": "#f28e2b",
    "dinov3": "#59a14f",
    "mae": "#e15759",
    "siglip": "#76b7b2",
    "siglip2": "#b07aa1",
}


def fig_per_image_strips(rows: list[dict]) -> str:
    """2x3 faceted strip plot: 139 per-image deltas as jittered dots per model.

    Full fine-tuning only. Bold marker for mean, thin CI whisker overlay.
    Reveals whether improvements are consistent or driven by outliers.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    rng = np.random.default_rng(42)

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]
        y_pos = np.arange(len(MODELS))

        for mi, m in enumerate(MODELS):
            row = lookup_row(rows, m, "full", metric, pctl)
            if not row or "per_image_deltas" not in row:
                continue

            deltas = np.array(list(row["per_image_deltas"].values()))
            # Flip sign for lower-is-better
            if direction == "lower":
                deltas = -deltas

            mean_d = row["mean_delta"]
            ci_lo = row.get("delta_ci_lower", mean_d)
            ci_hi = row.get("delta_ci_upper", mean_d)
            if direction == "lower":
                mean_d, ci_lo, ci_hi = -mean_d, -ci_hi, -ci_lo

            # Jittered y for strip
            jitter = rng.uniform(-0.3, 0.3, size=len(deltas))
            color = MODEL_COLORS[m]

            ax.scatter(
                deltas, mi + jitter,
                c=color, alpha=0.25, s=8, edgecolors="none", zorder=2,
            )

            # CI whisker
            ax.plot(
                [ci_lo, ci_hi], [mi, mi],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=4,
            )

            # Mean marker
            ax.scatter(
                mean_d, mi,
                c=color, s=60, marker="D", zorder=5,
                edgecolors="white", linewidth=0.8,
            )

        # Zero line
        ax.axvline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--", zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in MODELS] if idx % 3 == 0 else [],
        )
        hint = "→ better" if direction == "higher" else "(sign flipped) → better"
        ax.set_title(f"{title}  ({hint})", fontsize=10)
        ax.invert_yaxis()
        clean_axes(ax)

    fig.suptitle(
        "Per-Image Δ Distribution  (Full Fine-Tuning, n = 139 images)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    # Annotation for markers
    fig.text(
        0.5, 1.01,
        "dots = individual images  |  diamond = mean  |  bar = 95% CI",
        ha="center", fontsize=9, color=TEXT_COLOR,
    )

    fig.tight_layout()
    save_figure(fig, "09_per_image_delta_strips.png")

    return (
        "Strip plot showing all 139 per-image deltas for Full fine-tuning. "
        "Diamond = mean, thick bar = 95% bootstrap CI. Jittered dots reveal "
        "whether improvement is consistent across images or driven by outliers. "
        "This figure is useful for spotting unevenness: strong mean gains are more "
        "convincing when most dots lean positive, but the negative tails remind us "
        "that some images still regress."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = load_q2_data()

    figures = [
        ("01 Validation Accuracy", lambda: fig_validation_accuracy()),
        ("02 Delta Heatmap Grid", lambda: fig_delta_heatmap(rows)),
        ("03 Diverging Bars", lambda: fig_diverging_bars(rows)),
        ("04 IoU Percentile Slopes", lambda: fig_iou_percentile_slopes(rows)),
        ("05 Radar Profiles", lambda: fig_radar_profiles(rows)),
        ("06 Accuracy vs Attention", lambda: fig_accuracy_vs_attention(rows)),
        ("07 Preserve / Enhance / Destroy", lambda: fig_preserve_enhance_destroy(rows)),
        ("08 Forest Plot with CIs", lambda: fig_forest_plot_ci(rows)),
        ("09 Per-Image Delta Strips", lambda: fig_per_image_strips(rows)),
    ]

    commentary_lines: list[str] = []
    for name, gen_fn in figures:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        commentary = gen_fn()
        print(f"  {commentary}")
        commentary_lines.append(f"## {name}\n{commentary}\n")

    commentary_path = FIGURES_DIR / "commentary.txt"
    commentary_path.write_text(
        "# Run Matrix Figure Commentary\n\n" + "\n".join(commentary_lines)
    )
    print(f"\n{'='*60}")
    print(f"  All 9 figures + commentary saved to {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
