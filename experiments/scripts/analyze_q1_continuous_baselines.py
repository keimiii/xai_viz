#!/usr/bin/env python3
"""Generate Q1 continuous-metric baseline comparison artifacts.

This script compares the frozen dashboard models' best default-method layer
scores against the documented continuous-metric baseline references for MSE,
KL divergence, and EMD. It writes both a machine-readable JSON artifact and a
short Markdown summary tailored for report writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.config import AVAILABLE_MODELS, METRICS_DB_PATH  # noqa: E402
from app.backend.services.metrics_service import MetricsService  # noqa: E402

ContinuousMetric = Literal["mse", "kl", "emd"]
BaselineName = Literal["random", "center_gaussian", "saliency_prior", "sobel_edge"]

METRIC_ORDER: tuple[ContinuousMetric, ...] = ("mse", "kl", "emd")
BASELINE_ORDER: tuple[BaselineName, ...] = (
    "random",
    "center_gaussian",
    "saliency_prior",
    "sobel_edge",
)

METRIC_LABELS: dict[ContinuousMetric, str] = {
    "mse": "MSE",
    "kl": "KL",
    "emd": "EMD",
}

BASELINE_LABELS: dict[BaselineName, str] = {
    "random": "Random",
    "center_gaussian": "Center Gaussian",
    "saliency_prior": "Saliency Prior",
    "sobel_edge": "Sobel Edge",
}

BASELINE_REFERENCES: dict[BaselineName, dict[ContinuousMetric, float]] = {
    "random": {"mse": 0.3192, "kl": 3.3627, "emd": 0.3468},
    "center_gaussian": {"mse": 0.1770, "kl": 2.6317, "emd": 0.2836},
    "saliency_prior": {"mse": 0.0957, "kl": 2.6111, "emd": 0.2654},
    "sobel_edge": {"mse": 0.0376, "kl": 3.2237, "emd": 0.3137},
}

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_JSON_FILENAME = "q1_continuous_baseline_comparison.json"
DEFAULT_MARKDOWN_FILENAME = "q1_continuous_baseline_summary.md"


class LeaderboardRow(TypedDict):
    """Subset of leaderboard fields used by this script."""

    rank: int
    model: str
    metric: str
    score: float
    best_layer: str
    method_used: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-model continuous metrics against the documented "
            "baseline references and write JSON + Markdown artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated JSON and Markdown artifacts.",
    )
    parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_FILENAME,
        help="Filename for the JSON artifact.",
    )
    parser.add_argument(
        "--markdown-name",
        default=DEFAULT_MARKDOWN_FILENAME,
        help="Filename for the Markdown summary artifact.",
    )
    return parser.parse_args()


def collect_leaderboards(service: MetricsService) -> dict[ContinuousMetric, list[LeaderboardRow]]:
    """Load best-layer leaderboard rows for the three continuous metrics."""
    leaderboards: dict[ContinuousMetric, list[LeaderboardRow]] = {}
    for metric in METRIC_ORDER:
        rows = service.get_leaderboard(metric=metric, ranking_mode="default_method")
        leaderboards[metric] = [cast(LeaderboardRow, row) for row in rows]
    return leaderboards


def summarize_metric_row(row: LeaderboardRow, metric: ContinuousMetric) -> dict[str, Any]:
    """Compare one best-layer score against the documented baseline values."""
    score = float(row["score"])
    passes_baselines = {
        baseline_name: score <= BASELINE_REFERENCES[baseline_name][metric]
        for baseline_name in BASELINE_ORDER
    }
    passed_baselines = [
        baseline_name for baseline_name in BASELINE_ORDER if passes_baselines[baseline_name]
    ]

    surprises: list[str] = []
    if not passes_baselines["random"]:
        surprises.append("worse_than_random")

    beats_all_baselines = len(passed_baselines) == len(BASELINE_ORDER)
    if beats_all_baselines:
        surprises.append("beats_all_baselines")
    elif passed_baselines == ["random"]:
        surprises.append("beats_only_random")
    elif passes_baselines["random"]:
        surprises.append("beats_random_but_not_stronger_priors")

    return {
        "rank": int(row["rank"]),
        "score": score,
        "best_layer": row["best_layer"],
        "method_used": row["method_used"],
        "passes_baselines": passes_baselines,
        "passed_baseline_count": len(passed_baselines),
        "passed_baselines": passed_baselines,
        "beats_all_baselines": beats_all_baselines,
        "surprises": surprises,
    }


def build_model_records(
    leaderboards: dict[ContinuousMetric, list[LeaderboardRow]],
    *,
    model_order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Build per-model continuous-metric comparison records."""
    models: dict[str, dict[str, Any]] = {}
    for model_name in model_order:
        metrics: dict[str, Any] = {}
        for metric in METRIC_ORDER:
            row = next((entry for entry in leaderboards[metric] if entry["model"] == model_name), None)
            if row is None:
                continue
            metrics[metric] = summarize_metric_row(row, metric)

        if not metrics:
            continue

        models[model_name] = {
            "model": model_name,
            "metrics": metrics,
        }

    return models


def describe_metric_strength(metric_data: dict[str, Any], metric: ContinuousMetric) -> str:
    """Return a short evidence-only summary for one metric."""
    passed = cast(list[BaselineName], metric_data["passed_baselines"])
    if not passed:
        return f"{METRIC_LABELS[metric]} beats no baselines"
    if metric_data["beats_all_baselines"]:
        return f"{METRIC_LABELS[metric]} beats all four baselines"

    labels = ", ".join(BASELINE_LABELS[name] for name in passed)
    return f"{METRIC_LABELS[metric]} beats {labels}"


def build_headline_findings(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate report-ready, evidence-only headline findings."""
    findings: list[dict[str, Any]] = []

    for metric in METRIC_ORDER:
        metric_ranked_models = sorted(
            models.items(),
            key=lambda item: cast(int, item[1]["metrics"][metric]["rank"]),
        )
        beat_all_models = [
            model_name
            for model_name, model_data in metric_ranked_models
            if model_data["metrics"][metric]["beats_all_baselines"]
        ]
        worse_than_random_models = [
            model_name
            for model_name, model_data in metric_ranked_models
            if "worse_than_random" in model_data["metrics"][metric]["surprises"]
        ]

        findings.append(
            {
                "type": "metric_clearance",
                "metric": metric,
                "models": beat_all_models,
                "summary": (
                    f"{METRIC_LABELS[metric]}: "
                    f"{', '.join(beat_all_models) if beat_all_models else 'no models'} "
                    "beat all four baselines."
                ),
            }
        )
        findings.append(
            {
                "type": "metric_worse_than_random",
                "metric": metric,
                "models": worse_than_random_models,
                "summary": (
                    f"{METRIC_LABELS[metric]}: "
                    f"{', '.join(worse_than_random_models) if worse_than_random_models else 'no models'} "
                    "score worse than the random baseline."
                ),
            }
        )

    all_metric_clear_models = [
        model_name
        for model_name, model_data in models.items()
        if all(model_data["metrics"][metric]["beats_all_baselines"] for metric in METRIC_ORDER)
    ]
    findings.append(
        {
            "type": "all_metric_clearance",
            "models": all_metric_clear_models,
            "summary": (
                "Across MSE, KL, and EMD: "
                f"{', '.join(all_metric_clear_models) if all_metric_clear_models else 'no models'} "
                "beat all four baselines on every continuous metric."
            ),
        }
    )

    return findings


def build_cross_metric_findings(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate per-model cross-metric synthesis findings."""
    findings: list[dict[str, Any]] = []

    for model_name, model_data in models.items():
        metrics = cast(dict[ContinuousMetric, dict[str, Any]], model_data["metrics"])
        pass_counts = {
            metric: int(metric_data["passed_baseline_count"])
            for metric, metric_data in metrics.items()
        }
        full_clear_metrics = [
            metric for metric, metric_data in metrics.items() if metric_data["beats_all_baselines"]
        ]
        worse_than_random_metrics = [
            metric for metric, metric_data in metrics.items() if "worse_than_random" in metric_data["surprises"]
        ]
        has_specific_gap_finding = False

        if len(full_clear_metrics) == len(METRIC_ORDER):
            findings.append(
                {
                    "type": "consistent_strength",
                    "model": model_name,
                    "summary": (
                        f"{model_name} beats all four baselines on MSE, KL, and EMD at its best "
                        "default-method layer."
                    ),
                }
            )
            continue

        if "mse" in full_clear_metrics and (
            "kl" not in full_clear_metrics or "emd" not in full_clear_metrics
        ):
            weaker_metrics: list[str] = []
            for weaker_metric in ("kl", "emd"):
                metric_data = metrics[weaker_metric]
                if metric_data["beats_all_baselines"]:
                    continue
                weaker_metrics.append(describe_metric_strength(metric_data, weaker_metric))

            findings.append(
                {
                    "type": "mse_vs_distribution_gap",
                    "model": model_name,
                    "summary": (
                        f"{model_name} beats all four baselines on MSE but has a weaker "
                        f"distribution-level story: {'; '.join(weaker_metrics)}."
                    ),
                }
            )
            has_specific_gap_finding = True

        strongest_metric = max(
            METRIC_ORDER,
            key=lambda metric: (pass_counts[metric], -METRIC_ORDER.index(metric)),
        )
        weakest_metric = min(
            METRIC_ORDER,
            key=lambda metric: (pass_counts[metric], METRIC_ORDER.index(metric)),
        )

        if not has_specific_gap_finding and pass_counts[strongest_metric] - pass_counts[weakest_metric] >= 2:
            findings.append(
                {
                    "type": "strongest_vs_weakest_conflict",
                    "model": model_name,
                    "summary": (
                        f"{model_name} shows a cross-metric spread: "
                        f"{describe_metric_strength(metrics[strongest_metric], strongest_metric)} "
                        f"but {describe_metric_strength(metrics[weakest_metric], weakest_metric).lower()}."
                    ),
                }
            )

        if worse_than_random_metrics:
            findings.append(
                {
                    "type": "worse_than_random_cross_metric",
                    "model": model_name,
                    "summary": (
                        f"{model_name} falls below the random baseline on "
                        f"{', '.join(METRIC_LABELS[metric] for metric in worse_than_random_metrics)} "
                        f"even though its strongest metric is {METRIC_LABELS[strongest_metric]}."
                    ),
                }
            )

    return findings


def build_comparison_payload(
    leaderboards: dict[ContinuousMetric, list[LeaderboardRow]],
    *,
    model_order: Sequence[str] = AVAILABLE_MODELS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-ready comparison payload."""
    timestamp = generated_at or datetime.now(UTC).isoformat()
    models = build_model_records(leaderboards, model_order=model_order)
    headline_findings = build_headline_findings(models)
    cross_metric_findings = build_cross_metric_findings(models)

    return {
        "generated_at": timestamp,
        "scope": {
            "models": list(model_order),
            "ranking_mode": "default_method",
            "metrics": list(METRIC_ORDER),
            "source_table": "aggregate_metrics",
            "score_selection": "best default-method layer per model and metric",
            "metrics_db_path": str(METRICS_DB_PATH),
        },
        "baseline_source": {
            "kind": "documented_constants",
            "source_doc": "docs/reference/metrics_methodology.md",
            "lower_is_better": True,
        },
        "baselines": BASELINE_REFERENCES,
        "models": models,
        "headline_findings": headline_findings,
        "cross_metric_findings": cross_metric_findings,
    }


def format_metric_value(metric: ContinuousMetric, value: float) -> str:
    """Format a metric value for Markdown output."""
    _ = metric
    return f"{value:.4f}"


def format_passed_baselines(passed_baselines: Sequence[str]) -> str:
    """Format passed baseline labels for Markdown tables."""
    if not passed_baselines:
        return "None"
    return ", ".join(BASELINE_LABELS[cast(BaselineName, name)] for name in passed_baselines)


def render_metric_table(payload: dict[str, Any], metric: ContinuousMetric) -> list[str]:
    """Render one Markdown table for a metric."""
    baseline_summary = ", ".join(
        f"{BASELINE_LABELS[name]} {format_metric_value(metric, values[metric])}"
        for name, values in BASELINE_REFERENCES.items()
    )
    lines = [
        f"### {METRIC_LABELS[metric]}",
        "",
        f"Baseline references: {baseline_summary}",
        "",
        "| Rank | Model | Score | Best layer | Method | Beats |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    metric_rows: list[tuple[str, dict[str, Any]]] = []
    for model_name, model_data in cast(dict[str, dict[str, Any]], payload["models"]).items():
        metric_rows.append((model_name, model_data["metrics"][metric]))

    for model_name, metric_data in sorted(metric_rows, key=lambda item: item[1]["rank"]):
        lines.append(
            "| "
            f"{metric_data['rank']} | "
            f"{model_name} | "
            f"{format_metric_value(metric, metric_data['score'])} | "
            f"{metric_data['best_layer']} | "
            f"{metric_data['method_used']} | "
            f"{format_passed_baselines(metric_data['passed_baselines'])} |"
        )

    lines.append("")
    return lines


def render_model_wrapups(payload: dict[str, Any]) -> list[str]:
    """Render compact per-model wrap-ups for quick report drafting."""
    lines = ["## Per-model Wrap-up", ""]

    for model_name, model_data in cast(dict[str, dict[str, Any]], payload["models"]).items():
        summaries = [
            describe_metric_strength(
                cast(dict[str, Any], model_data["metrics"][metric]),
                metric,
            )
            for metric in METRIC_ORDER
        ]
        lines.append(f"- {model_name}: " + "; ".join(summaries) + ".")

    lines.append("")
    return lines


def render_markdown_summary(payload: dict[str, Any]) -> str:
    """Render the report-friendly Markdown summary artifact."""
    lines = [
        "# Q1 Continuous Baseline Comparison",
        "",
        (
            "Scope: frozen dashboard models, scored on MSE, KL, and EMD using each model's "
            "best default-method layer from the metrics database. Lower is better for all "
            "three metrics."
        ),
        "",
        (
            "Baseline source: documented constants from "
            "`docs/reference/metrics_methodology.md`."
        ),
        "",
        "## Headline Findings",
        "",
    ]

    for finding in cast(list[dict[str, Any]], payload["headline_findings"]):
        lines.append(f"- {finding['summary']}")

    lines.extend(["", "## Cross-metric Divergences", ""])
    cross_metric_findings = cast(list[dict[str, Any]], payload["cross_metric_findings"])
    if cross_metric_findings:
        for finding in cross_metric_findings:
            lines.append(f"- {finding['summary']}")
    else:
        lines.append("- No cross-metric divergences met the reporting thresholds.")

    lines.append("")
    lines.append("## Per-metric Comparison")
    lines.append("")
    for metric in METRIC_ORDER:
        lines.extend(render_metric_table(payload, metric))

    lines.extend(render_model_wrapups(payload))
    return "\n".join(lines).strip() + "\n"


def save_artifacts(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    json_name: str,
    markdown_name: str,
) -> tuple[Path, Path]:
    """Write the JSON + Markdown artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    markdown_path = output_dir / markdown_name

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(payload), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    service = MetricsService()
    leaderboards = collect_leaderboards(service)
    payload = build_comparison_payload(leaderboards)
    json_path, markdown_path = save_artifacts(
        payload,
        output_dir=args.output_dir,
        json_name=args.json_name,
        markdown_name=args.markdown_name,
    )

    print(f"Wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {markdown_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
