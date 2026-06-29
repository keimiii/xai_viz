#!/usr/bin/env python3
"""Per-architectural-style breakdown of Δ IoU across model × strategy pairs.

Reads per-image deltas from an existing Q2 metrics analysis artifact and
cross-references them against building_parts.json style labels to produce
per-style statistics — no new model runs required.

Outputs:
  - Console table: per-style (model × strategy) Δ IoU
  - JSON artifact: full per-style statistics with Kruskal-Wallis test
  - PNG figure: grouped bar chart of per-style Δ IoU (full strategy)

Usage:
  python experiments/scripts/analyze_style_breakdown.py
  python experiments/scripts/analyze_style_breakdown.py --experiment-id fine_tuning_primary_20260327
  python experiments/scripts/analyze_style_breakdown.py --strategy full --metric iou --percentile 90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import (  # noqa: E402
    ANNOTATIONS_PATH,
    STYLE_MAPPING,
    STYLE_NAMES,
)
from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    get_experiment_paths,
    load_active_experiment,
)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

STYLE_ORDER: tuple[str, ...] = STYLE_NAMES  # ("Romanesque", "Gothic", "Renaissance", "Baroque")
STYLE_BY_ID: dict[str, str] = {qid: STYLE_NAMES[idx] for qid, idx in STYLE_MAPPING.items()}

# Model display order for tables and figures
MODEL_ORDER = ("clip", "siglip", "siglip2", "mae", "dinov2", "dinov3")
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


def load_image_styles() -> dict[str, list[str]]:
    """Return {image_filename: [style_name, ...]} from building_parts.json."""
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Annotations not found at {ANNOTATIONS_PATH}")

    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        ann = json.load(f)

    return {
        img_id: [
            STYLE_BY_ID[s]
            for s in img_data.get("styles", [])
            if s in STYLE_BY_ID
        ]
        for img_id, img_data in ann["annotations"].items()
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def group_deltas_by_style(
    per_image_deltas: dict[str, float],
    image_styles: dict[str, list[str]],
) -> dict[str, list[float]]:
    """Map per-image deltas to a dict of {style_name: [delta, ...]}."""
    style_deltas: dict[str, list[float]] = {s: [] for s in STYLE_ORDER}
    for image_id, delta in per_image_deltas.items():
        for style in image_styles.get(image_id, []):
            if style in style_deltas:
                style_deltas[style].append(delta)
    return style_deltas


def style_stats(deltas: list[float]) -> dict[str, float | int]:
    """Compute summary statistics for one style's delta list."""
    if not deltas:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}
    arr = np.array(deltas)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    # 95% bootstrap CI
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(10_000, n), replace=True).mean(axis=1)
    ci_lower = float(np.percentile(boots, 2.5))
    ci_upper = float(np.percentile(boots, 97.5))
    return {"n": n, "mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper}


def kruskal_wallis_test(style_deltas: dict[str, list[float]]) -> dict[str, Any]:
    """Kruskal-Wallis H-test: does Δ IoU differ significantly across styles?"""
    groups = [style_deltas[s] for s in STYLE_ORDER if len(style_deltas[s]) >= 2]
    if len(groups) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    h_stat, p_value = stats.kruskal(*groups)
    return {
        "statistic": float(h_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def analyze_rows(
    q2_data: dict[str, Any],
    image_styles: dict[str, list[str]],
    *,
    strategy: str,
    metric: str,
    percentile: int,
) -> list[dict[str, Any]]:
    """Compute per-style stats for every model matching the given filters."""
    results = []
    target_percentile: int | None = percentile if metric == "iou" else None

    for row in q2_data["rows"]:
        if row["strategy_id"] != strategy:
            continue
        if row["metric"] != metric:
            continue
        if row.get("percentile") != target_percentile:
            continue

        per_image = row.get("per_image_deltas", {})
        style_deltas = group_deltas_by_style(per_image, image_styles)

        per_style = {
            style: style_stats(style_deltas[style]) for style in STYLE_ORDER
        }
        kw = kruskal_wallis_test(style_deltas)

        results.append({
            "model_name": row["model_name"],
            "strategy_id": row["strategy_id"],
            "metric": row["metric"],
            "percentile": row.get("percentile"),
            "aggregate_frozen_mean": row["frozen_mean"],
            "aggregate_finetuned_mean": row["finetuned_mean"],
            "aggregate_delta": row["mean_delta"],
            "aggregate_cohens_d": row["cohens_d"],
            "aggregate_significant": row["significant"],
            "per_style": per_style,
            "kruskal_wallis": kw,
        })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(
    results: list[dict[str, Any]],
    *,
    strategy: str,
    metric: str,
    percentile: int,
) -> None:
    metric_label = f"{metric.upper()} p{percentile}" if metric == "iou" else metric.upper()
    print(f"\n{'=' * 100}")
    print(f"Per-Style Δ {metric_label} — strategy: {strategy}")
    print(f"{'=' * 100}")

    header = f"{'Model':<10}  {'Aggregate':>10}"
    for style in STYLE_ORDER:
        header += f"  {style[:13]:>13}"
    header += f"  {'KW p':>8}"
    print(header)
    print("-" * 100)

    ordered = sorted(
        results,
        key=lambda r: r["aggregate_delta"],
        reverse=True,
    )
    for r in ordered:
        row_str = f"{r['model_name']:<10}  {r['aggregate_delta']:>+10.4f}"
        for style in STYLE_ORDER:
            ps = r["per_style"][style]
            mean_str = f"{ps['mean']:+.3f}" if ps["n"] > 0 else "  n/a"
            row_str += f"  {mean_str:>13}"
        kw_p = r["kruskal_wallis"]["p_value"]
        sig = "*" if r["kruskal_wallis"]["significant"] else ""
        kw_str = f"{kw_p:.3f}{sig}" if not np.isnan(kw_p) else "  n/a"
        row_str += f"  {kw_str:>8}"
        print(row_str)

    print()
    print("Kruskal-Wallis * = p < 0.05 (style significantly moderates Δ within this model)")
    if results:
        counts = {s: results[0]["per_style"][s]["n"] for s in STYLE_ORDER}
        print("Style image counts: " + ", ".join(f"{s}={counts[s]}" for s in STYLE_ORDER))


def save_json(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"JSON saved to: {output_path}")


def save_figure(
    results: list[dict[str, Any]],
    output_path: Path,
    *,
    strategy: str,
    metric: str,
    percentile: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # Build data matrix: models × styles
    ordered_models = [r["model_name"] for r in sorted(
        results, key=lambda r: r["aggregate_delta"], reverse=True
    )]

    n_models = len(ordered_models)
    n_styles = len(STYLE_ORDER)
    bar_width = 0.12
    group_gap = 0.05
    x = np.arange(n_styles)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, model in enumerate(ordered_models):
        model_result = next(r for r in results if r["model_name"] == model)
        means = [model_result["per_style"][s]["mean"] for s in STYLE_ORDER]
        cis_lower = [model_result["per_style"][s]["ci_lower"] for s in STYLE_ORDER]
        cis_upper = [model_result["per_style"][s]["ci_upper"] for s in STYLE_ORDER]

        yerr_lower = [
            max(0.0, m - cl) if not (np.isnan(m) or np.isnan(cl)) else 0.0
            for m, cl in zip(means, cis_lower, strict=True)
        ]
        yerr_upper = [
            max(0.0, cu - m) if not (np.isnan(m) or np.isnan(cu)) else 0.0
            for m, cu in zip(means, cis_upper, strict=True)
        ]

        offset = (i - n_models / 2 + 0.5) * (bar_width + group_gap / n_models)
        color = MODEL_COLORS.get(model, "#888888")

        ax.bar(
            x + offset,
            [m if not np.isnan(m) else 0.0 for m in means],
            width=bar_width,
            color=color,
            alpha=0.85,
            label=model,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            [m if not np.isnan(m) else 0.0 for m in means],
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=0.8,
            zorder=4,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(STYLE_ORDER, fontsize=11)
    ax.set_ylabel(f"Δ {metric.upper()} (fine-tuned − frozen)", fontsize=11)
    metric_label = f"{metric.upper()} p{percentile}" if metric == "iou" else metric.upper()
    ax.set_title(
        f"Per-Style Δ {metric_label} — strategy: {strategy}\n"
        "Error bars: 95% bootstrap CI",
        fontsize=12,
    )
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        ncol=2,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.3f"))

    # Annotate image counts per style
    style_counts = {s: results[0]["per_style"][s]["n"] for s in STYLE_ORDER} if results else {}
    for i, style in enumerate(STYLE_ORDER):
        ax.text(
            i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
            f"n={style_counts.get(style, '?')}",
            ha="center",
            va="top",
            fontsize=9,
            color="gray",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-style Δ IoU breakdown from existing Q2 analysis",
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
        help="Fine-tuning strategy to report. Default: full.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "coverage", "mse", "kl", "emd"],
        help="Metric to break down by style. Default: iou.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=90,
        help="IoU percentile threshold to use (ignored for non-IoU metrics). Default: 90.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <experiment_dir>/style_breakdown.json.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <experiment_dir>/style_breakdown.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Q2 metrics analysis...")
    q2_data, experiment_id = load_q2_metrics(args.experiment_id)
    print(f"  Experiment: {experiment_id}")
    print(f"  Images: {q2_data.get('evaluation_image_count', '?')}")
    print(f"  Analyzed layer: {q2_data.get('analyzed_layer', '?')}")

    print("Loading style labels from building_parts.json...")
    image_styles = load_image_styles()
    print(f"  {len(image_styles)} annotated images with style labels")

    print(f"\nComputing per-style breakdown: strategy={args.strategy}, "
          f"metric={args.metric}, percentile={args.percentile}...")
    results = analyze_rows(
        q2_data,
        image_styles,
        strategy=args.strategy,
        metric=args.metric,
        percentile=args.percentile,
    )

    if not results:
        print(f"No rows found for strategy={args.strategy}, metric={args.metric}. "
              "Check that the Q2 analysis includes these combinations.")
        return

    print_table(results, strategy=args.strategy, metric=args.metric, percentile=args.percentile)

    # Resolve output paths
    exp_dir = get_experiment_paths(experiment_id).results_dir
    json_path = args.output_json or exp_dir / "style_breakdown.json"
    fig_path = args.output_figure or exp_dir / "style_breakdown.png"

    save_json(results, json_path)
    save_figure(results, fig_path, strategy=args.strategy, metric=args.metric, percentile=args.percentile)

    print("\nDone!")


if __name__ == "__main__":
    main()
