#!/usr/bin/env python3
"""Cross-model image-level correlation analysis.

Asks: do different models agree on which images are 'easy' (high IoU) or
'improvable' (large Δ IoU)?

Two main analyses:
  1. Frozen-vs-delta correlation: does DINOv3's high frozen IoU predict where
     CLIP gains the most after fine-tuning? (Tests whether the two alignment
     mechanisms are independent or share 'easy images'.)
  2. Pairwise delta correlation matrix: which model pairs improve on the same
     images? (Reveals whether CLIP/SigLIP form a cluster separate from MAE/DINO.)

Data sources:
  - Per-image Δ IoU: Q2 metrics analysis JSON (full strategy, layer 11)
  - Per-image frozen IoU: metrics.db (frozen baselines, layer 11, p90)

Outputs:
  - Console: correlation table and key findings
  - JSON: full correlation matrix + per-image data
  - PNG: scatter plot (DINOv3 frozen vs CLIP Δ) + pairwise heatmap

Usage:
  python experiments/scripts/analyze_model_correlation.py
  python experiments/scripts/analyze_model_correlation.py --experiment-id fine_tuning_primary_20260327
  python experiments/scripts/analyze_model_correlation.py --strategy full --percentile 90
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import CACHE_PATH  # noqa: E402
from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    get_experiment_paths,
    load_active_experiment,
)

METRICS_DB_PATH = CACHE_PATH / "metrics.db"

# Models that participate in both frozen-IoU and Δ-IoU analysis
ANALYSIS_MODELS = ("clip", "siglip", "siglip2", "mae", "dinov2", "dinov3")

# Default method per model (matches DEFAULT_METHOD in config)
DEFAULT_DB_METHOD = {
    "clip": "cls",
    "siglip": "mean",
    "siglip2": "mean",
    "mae": "cls",
    "dinov2": "cls",
    "dinov3": "cls",
}

MODEL_COLORS = {
    "clip": "#4C72B0",
    "siglip": "#DD8452",
    "siglip2": "#C44E52",
    "mae": "#55A868",
    "dinov2": "#8172B3",
    "dinov3": "#937860",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_q2_metrics(experiment_id: str | None) -> tuple[dict[str, Any], str]:
    """Load the Q2 metrics analysis JSON for the given (or active) experiment."""
    active = load_active_experiment()
    if experiment_id is None:
        if active is None:
            raise FileNotFoundError(
                "No active experiment found. Pass --experiment-id explicitly."
            )
        experiment_id = str(active["experiment_id"])

    assert isinstance(experiment_id, str)
    paths = get_experiment_paths(experiment_id)
    if not paths.q2_metrics_path.exists():
        raise FileNotFoundError(
            f"Q2 metrics not found at {paths.q2_metrics_path}. "
            "Run analyze_q2_metrics.py first."
        )

    with open(paths.q2_metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, experiment_id


def load_frozen_iou_per_image(
    model: str,
    *,
    layer: int = 11,
    percentile: int = 90,
) -> dict[str, float]:
    """Load per-image frozen IoU from metrics.db for one model at a given layer."""
    if not METRICS_DB_PATH.exists():
        raise FileNotFoundError(
            f"Metrics DB not found at {METRICS_DB_PATH}. "
            "Run generate_metrics_cache.py first."
        )

    method = DEFAULT_DB_METHOD[model]
    layer_str = f"layer{layer}"

    conn = sqlite3.connect(METRICS_DB_PATH)
    try:
        cursor = conn.execute(
            """
            SELECT image_id, iou
            FROM image_metrics
            WHERE model = ? AND method = ? AND layer = ? AND percentile = ?
            """,
            (model, method, layer_str, percentile),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


def extract_delta_per_image(
    q2_data: dict[str, Any],
    *,
    model: str,
    strategy: str,
    metric: str = "iou",
    percentile: int = 90,
) -> dict[str, float]:
    """Extract per-image deltas from Q2 JSON for one model/strategy."""
    target_percentile: int | None = percentile if metric == "iou" else None
    for row in q2_data["rows"]:
        if (
            row["model_name"] == model
            and row["strategy_id"] == strategy
            and row["metric"] == metric
            and row.get("percentile") == target_percentile
        ):
            return dict(row.get("per_image_deltas", {}))
    return {}


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def pearson_and_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute Pearson r and Spearman ρ with p-values."""
    if len(x) < 3:
        nan = float("nan")
        return {"pearson_r": nan, "pearson_p": nan, "spearman_r": nan, "spearman_p": nan, "n": len(x)}

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n": len(x),
    }


def align_vectors(
    a: dict[str, float],
    b: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return aligned numpy arrays for image IDs present in both dicts."""
    common_ids = sorted(set(a) & set(b))
    return (
        np.array([a[i] for i in common_ids]),
        np.array([b[i] for i in common_ids]),
        common_ids,
    )


def build_pairwise_delta_matrix(
    delta_vectors: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute pairwise Pearson r between all model Δ IoU vectors."""
    models = list(delta_vectors.keys())
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for m1 in models:
        matrix[m1] = {}
        for m2 in models:
            x, y, _ = align_vectors(delta_vectors[m1], delta_vectors[m2])
            matrix[m1][m2] = pearson_and_spearman(x, y)
    return matrix


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_frozen_vs_delta_table(
    frozen_vs_delta: dict[str, dict[str, float]],
    *,
    strategy: str = "full",
) -> None:
    """Print correlations between each model's frozen IoU and CLIP's Δ IoU."""
    print(f"\n{'=' * 80}")
    print(f"Frozen IoU vs CLIP Δ IoU (strategy={strategy})")
    print("Rows: frozen IoU of <model> vs CLIP Δ IoU per image")
    print(f"{'=' * 80}")
    print(f"{'Model (frozen)':20}  {'Pearson r':>10}  {'p':>8}  {'Spearman ρ':>11}  {'p':>8}  {'n':>5}")
    print("-" * 80)
    for model, corr in sorted(frozen_vs_delta.items(), key=lambda kv: -abs(kv[1]["pearson_r"])):
        pr = corr["pearson_r"]
        pp = corr["pearson_p"]
        sr = corr["spearman_r"]
        sp = corr["spearman_p"]
        n = corr["n"]
        sig_p = "*" if pp < 0.05 else ""
        sig_s = "*" if sp < 0.05 else ""
        print(f"{model:20}  {pr:>+10.3f}  {pp:>7.4f}{sig_p}  {sr:>+11.3f}  {sp:>7.4f}{sig_s}  {n:>5}")
    print("* p < 0.05")


def print_delta_correlation_matrix(
    matrix: dict[str, dict[str, dict[str, float]]],
    models: list[str],
) -> None:
    """Print pairwise Pearson r between per-image Δ IoU vectors."""
    print(f"\n{'=' * 80}")
    print("Pairwise Δ IoU Pearson r (per-image, full strategy)")
    print(f"{'=' * 80}")
    col_w = 10
    header = f"{'':12}" + "".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print("-" * (12 + col_w * len(models)))
    for m1 in models:
        row_str = f"{m1:<12}"
        for m2 in models:
            r = matrix[m1][m2]["pearson_r"]
            row_str += f"{r:>{col_w}.3f}"
        print(row_str)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved to: {output_path}")


def save_dinov3_vs_clip_figure(
    *,
    dinov3_frozen: dict[str, float],
    clip_delta: dict[str, float],
    output_path: Path,
) -> None:
    """Render the single-panel DINOv3-frozen-IoU vs CLIP-Δ-IoU scatter (Fig 7).

    Publication figure for the final report: one point per annotated image,
    regression line with a 95% confidence band, and correlation stats annotated
    in the upper-left corner. No title — the report caption carries it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping DINOv3-vs-CLIP figure.")
        return

    x_vals, y_vals, _ = align_vectors(dinov3_frozen, clip_delta)
    if len(x_vals) < 3:
        print("Not enough aligned points for DINOv3-vs-CLIP figure — skipping.")
        return

    corr = pearson_and_spearman(x_vals, y_vals)
    dot_color = MODEL_COLORS["dinov3"]
    line_color = "#3A3A3A"  # complementary dark grey

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.scatter(x_vals, y_vals, alpha=0.55, s=28, color=dot_color, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=2)

    # OLS regression line with 95% confidence band (t-distribution, df = n-2).
    n = len(x_vals)
    slope, intercept, *_ = stats.linregress(x_vals, y_vals)
    x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_pred = slope * x_grid + intercept

    y_hat = slope * x_vals + intercept
    dof = n - 2
    s_err = np.sqrt(np.sum((y_vals - y_hat) ** 2) / dof)
    x_mean = x_vals.mean()
    ss_x = np.sum((x_vals - x_mean) ** 2)
    se_fit = s_err * np.sqrt(1.0 / n + (x_grid - x_mean) ** 2 / ss_x)
    t_crit = stats.t.ppf(0.975, dof)
    band = t_crit * se_fit

    ax.fill_between(
        x_grid, y_pred - band, y_pred + band,
        color=line_color, alpha=0.2, zorder=3, linewidth=0,
    )
    ax.plot(x_grid, y_pred, color=line_color, linewidth=1.5, zorder=4)

    ax.set_xlabel("DINOv3 frozen IoU@90", fontsize=12)
    ax.set_ylabel("CLIP ΔIoU@90 (Full fine-tuning − Frozen)", fontsize=12)
    ax.tick_params(labelsize=11)

    p_val = corr["pearson_p"]
    p_text = "p < 0.0001" if p_val < 1e-4 else f"p = {p_val:.4f}"
    annotation = (
        f"Pearson r = {corr['pearson_r']:+.3f}\n"
        f"Spearman ρ = {corr['spearman_r']:+.3f}\n"
        f"{p_text}"
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "0.7", "alpha": 0.85},
        zorder=5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"DINOv3-vs-CLIP figure saved to: {output_path}")


def save_figures(
    *,
    clip_delta: dict[str, float],
    frozen_iou_all: dict[str, dict[str, float]],
    delta_matrix: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    scatter_output: Path,
    heatmap_output: Path,
    strategy: str,
    percentile: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figures.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Figure 1: DINOv3 frozen IoU vs CLIP Δ IoU scatter ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        frozen = frozen_iou_all.get(model, {})
        x_vals, y_vals, _ = align_vectors(frozen, clip_delta)
        if len(x_vals) < 3:
            ax.set_visible(False)
            continue

        corr = pearson_and_spearman(x_vals, y_vals)
        color = MODEL_COLORS.get(model, "#888888")

        ax.scatter(x_vals, y_vals, alpha=0.55, s=28, color=color, zorder=3)

        # Regression line
        slope, intercept, *_ = stats.linregress(x_vals, y_vals)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.5,
                alpha=0.8, zorder=4)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=2)
        ax.set_xlabel(f"{model} frozen IoU (p{percentile})", fontsize=9)
        ax.set_ylabel("CLIP Δ IoU (full FT)", fontsize=9)
        ax.set_title(
            f"r={corr['pearson_r']:+.3f}  ρ={corr['spearman_r']:+.3f}"
            f"  p={corr['pearson_p']:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Frozen IoU of each model vs CLIP Δ IoU (full FT, layer 11, p{percentile})\n"
        "Positive r: models that share 'easy images'; negative r: complementary mechanisms",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    scatter_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scatter_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter figure saved to: {scatter_output}")

    # --- Figure 2: Pairwise Δ IoU Pearson r heatmap ---
    n = len(models)
    r_matrix = np.array([
        [delta_matrix[m1][m2]["pearson_r"] for m2 in models]
        for m1 in models
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r_matrix, vmin=-0.5, vmax=1.0, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            r_val = r_matrix[i, j]
            p_val = delta_matrix[models[i]][models[j]]["pearson_p"]
            text = f"{r_val:.2f}"
            if i != j and p_val < 0.05:
                text += "*"
            color = "white" if abs(r_val) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_title(
        f"Pairwise Δ IoU Pearson r across 139 images\n"
        f"Strategy: {strategy}, p{percentile}  (* p < 0.05)",
        fontsize=11,
    )
    fig.tight_layout()
    heatmap_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(heatmap_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap figure saved to: {heatmap_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model image-level correlation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID to analyze. Defaults to active experiment.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="full",
        choices=["linear_probe", "lora", "full"],
        help="Fine-tuning strategy for Δ IoU vectors. Default: full.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=90,
        help="IoU percentile threshold. Default: 90.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        help="Layer for frozen IoU lookup in metrics.db. Default: 11.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <experiment_dir>/model_correlation.json.",
    )
    parser.add_argument(
        "--output-scatter",
        type=Path,
        default=None,
        help="Output scatter figure path. Defaults to <experiment_dir>/model_correlation_scatter.png.",
    )
    parser.add_argument(
        "--output-heatmap",
        type=Path,
        default=None,
        help="Output heatmap figure path. Defaults to <experiment_dir>/model_correlation_heatmap.png.",
    )
    parser.add_argument(
        "--output-dinov3-scatter",
        type=Path,
        default=(
            project_root
            / "docs/final_report/figures/model_correlation_dinov3_vs_clip.png"
        ),
        help=(
            "Output path for the single-panel DINOv3-vs-CLIP report figure. "
            "Defaults to docs/final_report/figures/model_correlation_dinov3_vs_clip.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Q2 metrics analysis...")
    q2_data, experiment_id = load_q2_metrics(args.experiment_id)
    print(f"  Experiment: {experiment_id}")

    # Collect available models from the Q2 JSON
    available_models = [
        m for m in ANALYSIS_MODELS
        if any(
            r["model_name"] == m and r["strategy_id"] == args.strategy
            and r["metric"] == "iou" and r.get("percentile") == args.percentile
            for r in q2_data["rows"]
        )
    ]
    print(f"  Models with full-strategy data: {available_models}")

    # Load Δ IoU per image for each model
    print(f"\nExtracting per-image Δ IoU (strategy={args.strategy}, p{args.percentile})...")
    delta_vectors: dict[str, dict[str, float]] = {}
    for model in available_models:
        delta_vectors[model] = extract_delta_per_image(
            q2_data, model=model, strategy=args.strategy, percentile=args.percentile
        )
        print(f"  {model}: {len(delta_vectors[model])} images")

    # Load frozen IoU per image from SQLite
    print(f"\nLoading frozen IoU per image from metrics.db (layer={args.layer}, p{args.percentile})...")
    frozen_iou_all: dict[str, dict[str, float]] = {}
    for model in available_models:
        try:
            frozen_iou_all[model] = load_frozen_iou_per_image(
                model, layer=args.layer, percentile=args.percentile
            )
            print(f"  {model}: {len(frozen_iou_all[model])} images")
        except Exception as exc:
            print(f"  {model}: failed — {exc}")

    if "clip" not in delta_vectors:
        print("CLIP delta not available — cannot run primary correlation. Exiting.")
        return

    # Analysis 1: frozen IoU of each model vs CLIP Δ IoU
    print("\nComputing: frozen IoU of each model vs CLIP Δ IoU...")
    clip_delta = delta_vectors["clip"]
    frozen_vs_clip_delta: dict[str, dict[str, float]] = {}
    for model in available_models:
        frozen = frozen_iou_all.get(model, {})
        if not frozen:
            continue
        x, y, _ = align_vectors(frozen, clip_delta)
        frozen_vs_clip_delta[model] = pearson_and_spearman(x, y)

    print_frozen_vs_delta_table(frozen_vs_clip_delta, strategy=args.strategy)

    # Analysis 2: pairwise Δ IoU correlation matrix
    print("\nComputing pairwise Δ IoU correlation matrix...")
    delta_matrix = build_pairwise_delta_matrix(delta_vectors)
    print_delta_correlation_matrix(delta_matrix, available_models)

    # Highlight key finding
    if "dinov3" in frozen_vs_clip_delta and "clip" in delta_vectors:
        corr = frozen_vs_clip_delta["dinov3"]
        pr = corr["pearson_r"]
        pp = corr["pearson_p"]
        interpretation = (
            "POSITIVE: high-DINOv3-IoU images also gain more from CLIP FT → shared 'easy' images"
            if pr > 0.15 else
            "NEGATIVE: CLIP FT helps most where DINOv3 already struggles → complementary mechanisms"
            if pr < -0.15 else
            "NEAR ZERO: DINOv3 frozen IoU does not predict CLIP Δ → independent mechanisms"
        )
        print(f"\nKey finding (DINOv3 frozen vs CLIP Δ): r={pr:+.3f}, p={pp:.4f}")
        print(f"  → {interpretation}")

    # Collect per-image data for JSON export
    all_image_ids = sorted(
        set().union(*[set(v.keys()) for v in delta_vectors.values()])
    )
    per_image_export = []
    for image_id in all_image_ids:
        entry: dict[str, Any] = {"image_id": image_id}
        for model in available_models:
            entry[f"{model}_delta"] = delta_vectors[model].get(image_id)
            entry[f"{model}_frozen_iou"] = frozen_iou_all.get(model, {}).get(image_id)
        per_image_export.append(entry)

    # Resolve output paths
    exp_dir = get_experiment_paths(experiment_id).results_dir
    json_path = args.output_json or exp_dir / "model_correlation.json"
    scatter_path = args.output_scatter or exp_dir / "model_correlation_scatter.png"
    heatmap_path = args.output_heatmap or exp_dir / "model_correlation_heatmap.png"

    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "strategy": args.strategy,
        "percentile": args.percentile,
        "layer": args.layer,
        "frozen_vs_clip_delta": frozen_vs_clip_delta,
        "delta_pairwise_correlations": {
            m1: {m2: delta_matrix[m1][m2] for m2 in available_models}
            for m1 in available_models
        },
        "per_image": per_image_export,
    }
    save_json(payload, json_path)

    save_figures(
        clip_delta=clip_delta,
        frozen_iou_all=frozen_iou_all,
        delta_matrix=delta_matrix,
        models=available_models,
        scatter_output=scatter_path,
        heatmap_output=heatmap_path,
        strategy=args.strategy,
        percentile=args.percentile,
    )

    # Single-panel report figure (Fig 7): DINOv3 frozen IoU vs CLIP Δ IoU.
    if "dinov3" in frozen_iou_all and frozen_iou_all["dinov3"]:
        save_dinov3_vs_clip_figure(
            dinov3_frozen=frozen_iou_all["dinov3"],
            clip_delta=clip_delta,
            output_path=args.output_dinov3_scatter,
        )
    else:
        print("DINOv3 frozen IoU unavailable — skipping single-panel report figure.")

    print("\nDone!")


if __name__ == "__main__":
    main()
