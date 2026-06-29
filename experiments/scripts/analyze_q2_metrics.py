#!/usr/bin/env python3
"""Analyze Q2 attention shift metrics across fine-tuning strategies.

This script compares frozen and fine-tuned model attention alignment for
(model, strategy) pairs across the metrics integrated with the frozen pipeline:
- IoU
- Coverage
- MSE
- KL divergence
- EMD

Outputs:
- per-image deltas
- per-(model, strategy, metric[, percentile]) statistics
- cross-strategy paired comparisons within each model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.attention.cls_attention import (  # noqa: E402
    HeadFusion,
    attention_to_heatmap,
    extract_cls_attention,
    extract_mean_attention,
)
from ssl_attention.config import (  # noqa: E402
    DATASET_PATH,
    DEFAULT_METHOD,
    FINETUNE_MODELS,
    FINETUNE_STRATEGIES,
    MODELS,
    AttentionMethod,
)
from ssl_attention.data.wikichurches import AnnotatedSubset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    RESULTS_PATH,
    FineTunableModel,
    get_checkpoint_candidates,
    load_finetuned_model,
)
from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    ANALYSIS_GIT_COMMIT_SHA_FIELD,
    get_experiment_paths,
    get_git_commit_sha,
    load_active_experiment,
    load_run_matrix,
    normalize_q2_analysis_payload,
    project_path,
    refresh_experiment_training_provenance,
    repo_relative_path,
    save_run_matrix,
    write_active_experiment,
)
from ssl_attention.metrics import (  # noqa: E402
    compute_coverage,
    compute_image_emd,
    compute_image_iou,
    compute_image_kl,
    compute_image_mse,
)
from ssl_attention.metrics.statistics import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    multiple_comparison_correction,
    paired_comparison,
)
from ssl_attention.models import create_model  # noqa: E402
from ssl_attention.utils.device import clear_memory  # noqa: E402

if TYPE_CHECKING:
    from ssl_attention.models.base import BaseVisionModel


AnalysisMetric = Literal["iou", "coverage", "mse", "kl", "emd"]
MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricConfig:
    key: AnalysisMetric
    label: str
    direction: MetricDirection
    percentile_dependent: bool


METRIC_CONFIGS: dict[AnalysisMetric, MetricConfig] = {
    "iou": MetricConfig(
        key="iou",
        label="IoU",
        direction="higher",
        percentile_dependent=True,
    ),
    "coverage": MetricConfig(
        key="coverage",
        label="Coverage",
        direction="higher",
        percentile_dependent=False,
    ),
    "mse": MetricConfig(
        key="mse",
        label="MSE",
        direction="lower",
        percentile_dependent=False,
    ),
    "kl": MetricConfig(
        key="kl",
        label="KL Divergence",
        direction="lower",
        percentile_dependent=False,
    ),
    "emd": MetricConfig(
        key="emd",
        label="EMD",
        direction="lower",
        percentile_dependent=False,
    ),
}


@dataclass
class PerImageMetricDelta:
    image_id: str
    frozen_value: float
    finetuned_value: float
    delta: float


@dataclass
class StrategyMetricSummary:
    model_name: str
    strategy_id: str
    metric: AnalysisMetric
    label: str
    direction: MetricDirection
    percentile_dependent: bool
    percentile: int | None
    method: str
    frozen_mean: float
    finetuned_mean: float
    mean_delta: float
    std_delta: float
    delta_ci_lower: float
    delta_ci_upper: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None
    significant: bool
    test_name: str
    correction_method: str | None = None
    correction_family_id: str | None = None
    correction_family_size: int | None = None
    per_image: list[PerImageMetricDelta] = field(default_factory=list)
    num_images: int = 0


@dataclass
class StrategyPairMetricComparison:
    model_name: str
    metric: AnalysisMetric
    percentile: int | None
    strategy_a: str
    strategy_b: str
    mean_delta_difference: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None
    significant: bool
    test_name: str


@dataclass
class AnalysisResults:
    experiment_id: str | None
    split_id: str | None
    analysis_git_commit_sha: str | None
    percentiles: list[int]
    analyzed_layer: int
    evaluation_image_count: int
    checkpoint_selection_rule: str
    result_set_scope: str
    metrics: list[MetricConfig]
    rows: list[StrategyMetricSummary]
    strategy_comparisons: list[StrategyPairMetricComparison]
    timestamp: str = ""


MetricBuckets = dict[AnalysisMetric, dict[int | None, list[tuple[str, float]]]]
StrategyMetricResults = dict[str, dict[AnalysisMetric, dict[int | None, StrategyMetricSummary]]]


def build_correction_family_id(metric: AnalysisMetric, percentile: int | None) -> str:
    """Build a stable identifier for one cross-model correction bucket."""
    percentile_label = f"p{percentile}" if percentile is not None else "threshold_free"
    return f"cross_model_summary:{metric}:{percentile_label}"


def resolve_layer_index(model_name: str, layer: int) -> int:
    """Resolve a requested layer index into an explicit non-negative layer."""
    if layer >= 0:
        return layer
    return MODELS[model_name].num_layers - 1


def parse_strategy_from_checkpoint_name(checkpoint_path: Path, model_name: str) -> str:
    """Infer strategy from checkpoint filename."""
    stem = checkpoint_path.stem
    for strategy in (s.value for s in FINETUNE_STRATEGIES):
        if stem == f"{model_name}_{strategy}_finetuned":
            return strategy
    if stem == f"{model_name}_finetuned":
        return "full"
    return "unknown"


def discover_strategy_checkpoints(
    model_names: list[str],
    strategy_ids: list[str],
    *,
    experiment_id: str | None,
    include_exploratory: bool,
) -> tuple[dict[str, dict[str, Path]], str | None]:
    """Discover checkpoint paths for requested model/strategy pairs.

    Prefers the explicit experiment run matrix and falls back to empty results if
    no active or requested experiment is available.
    """
    if experiment_id is None:
        discovered_checkpoints: dict[str, dict[str, Path]] = {}
        for model_name in model_names:
            per_model: dict[str, Path] = {}
            for strategy_id in strategy_ids:
                candidates = get_checkpoint_candidates(model_name, strategy_id=strategy_id)
                checkpoint = next((path for path in candidates if path.exists()), None)
                if checkpoint is not None:
                    per_model[strategy_id] = checkpoint
            if per_model:
                discovered_checkpoints[model_name] = per_model
        return discovered_checkpoints, None

    run_matrix = load_run_matrix(experiment_id)
    split_id = run_matrix.get("split_id")
    discovered: dict[str, dict[str, Path]] = {}

    for run in run_matrix.get("runs", {}).values():
        model_name = run.get("model")
        strategy_id = run.get("strategy")
        if model_name not in model_names or strategy_id not in strategy_ids:
            continue
        if not include_exploratory and run.get("run_scope") != "primary":
            continue
        checkpoint_path = project_path(run.get("checkpoint_path"))
        if checkpoint_path is None or not checkpoint_path.exists():
            continue
        discovered.setdefault(model_name, {})[strategy_id] = checkpoint_path

    return discovered, split_id


def extract_attention_heatmap(
    model: BaseVisionModel | FineTunableModel,
    pixel_values: torch.Tensor,
    model_name: str,
    layer: int = -1,
) -> torch.Tensor:
    """Extract attention heatmap from frozen or fine-tuned model."""
    config = MODELS[model_name]
    method = DEFAULT_METHOD[model_name]

    output = model.extract_attention(pixel_values) if isinstance(model, FineTunableModel) else model(pixel_values)
    attention_weights = output.attention_weights

    if method == AttentionMethod.CLS:
        attn = extract_cls_attention(
            attention_weights,
            layer=layer,
            num_registers=config.num_registers,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.MEAN:
        attn = extract_mean_attention(
            attention_weights,
            layer=layer,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.GRADCAM:
        raise NotImplementedError(f"Grad-CAM not supported in this script. Model: {model_name}")
    else:
        raise ValueError(f"Unknown attention method: {method}")

    heatmap = attention_to_heatmap(attn, image_size=224, normalize=True)
    return heatmap.squeeze(0)


def compute_image_metric_values(
    *,
    attention: torch.Tensor,
    annotation: Any,
    image_id: str,
    percentiles: list[int],
) -> dict[AnalysisMetric, dict[int | None, float]]:
    """Compute the full metric set for one image heatmap."""
    height, width = attention.shape[-2:]
    union_mask = annotation.get_union_mask(height, width).to(attention.device)

    metric_values: dict[AnalysisMetric, dict[int | None, float]] = {
        "iou": {},
        "coverage": {None: compute_coverage(attention, union_mask)},
        "mse": {None: compute_image_mse(attention=attention, annotation=annotation)},
        "kl": {None: compute_image_kl(attention=attention, annotation=annotation)},
        "emd": {None: compute_image_emd(attention=attention, annotation=annotation)},
    }

    for percentile in percentiles:
        metric_values["iou"][percentile] = compute_image_iou(
            attention=attention,
            annotation=annotation,
            image_id=image_id,
            percentile=percentile,
        ).iou

    return metric_values


def initialize_metric_buckets(percentiles: list[int]) -> MetricBuckets:
    """Create empty metric buckets for one model."""
    buckets: MetricBuckets = {}
    for metric_name, config in METRIC_CONFIGS.items():
        bucket_keys = tuple(percentiles) if config.percentile_dependent else (None,)
        buckets[metric_name] = {bucket_key: [] for bucket_key in bucket_keys}
    return buckets


def compute_model_metric_values(
    model: BaseVisionModel | FineTunableModel,
    dataset: AnnotatedSubset,
    model_name: str,
    percentiles: list[int],
    layer: int = -1,
) -> MetricBuckets:
    """Compute all supported metrics for one model across the evaluation dataset."""
    results = initialize_metric_buckets(percentiles)

    processor = model.processor
    device = model.device
    dtype = model.dtype

    for sample in tqdm(dataset, desc=f"Processing {model_name}", leave=False):
        image_id = sample["image_id"]
        image = sample["image"]
        annotation = sample["annotation"]

        processed = processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)

        with torch.no_grad():
            heatmap = extract_attention_heatmap(model, pixel_values, model_name, layer=layer)

        image_metric_values = compute_image_metric_values(
            attention=heatmap,
            annotation=annotation,
            image_id=image_id,
            percentiles=percentiles,
        )

        for metric_name, bucket_map in image_metric_values.items():
            for bucket_key, value in bucket_map.items():
                results[metric_name][bucket_key].append((image_id, value))

    return results


def summarize_metric_delta(
    *,
    model_name: str,
    strategy_id: str,
    metric: AnalysisMetric,
    percentile: int | None,
    method: str,
    frozen_values: dict[str, float],
    finetuned_values: dict[str, float],
) -> StrategyMetricSummary:
    """Build summary stats for one model/strategy/metric bucket."""
    config = METRIC_CONFIGS[metric]
    per_image_results: list[PerImageMetricDelta] = []
    frozen_arr: list[float] = []
    finetuned_arr: list[float] = []

    for image_id, frozen_value in frozen_values.items():
        finetuned_value = finetuned_values[image_id]
        delta = finetuned_value - frozen_value
        per_image_results.append(
            PerImageMetricDelta(
                image_id=image_id,
                frozen_value=frozen_value,
                finetuned_value=finetuned_value,
                delta=delta,
            )
        )
        frozen_arr.append(frozen_value)
        finetuned_arr.append(finetuned_value)

    frozen_np = np.array(frozen_arr)
    finetuned_np = np.array(finetuned_arr)
    delta_np = finetuned_np - frozen_np

    _, ci_lower, ci_upper = bootstrap_ci(delta_np, statistic="mean")
    comparison = paired_comparison(
        finetuned_np,
        frozen_np,
        model_a_name="finetuned",
        model_b_name="frozen",
        test="auto",
    )

    return StrategyMetricSummary(
        model_name=model_name,
        strategy_id=strategy_id,
        metric=metric,
        label=config.label,
        direction=config.direction,
        percentile_dependent=config.percentile_dependent,
        percentile=percentile,
        method=method,
        frozen_mean=float(np.mean(frozen_np)),
        finetuned_mean=float(np.mean(finetuned_np)),
        mean_delta=float(np.mean(delta_np)),
        std_delta=float(np.std(delta_np, ddof=1)),
        delta_ci_lower=ci_lower,
        delta_ci_upper=ci_upper,
        cohens_d=cohens_d(finetuned_np, frozen_np, paired=True),
        p_value=comparison.p_value,
        corrected_p_value=None,
        significant=comparison.significant,
        test_name=comparison.test_name,
        per_image=per_image_results,
        num_images=len(per_image_results),
    )


def analyze_model_strategy(
    *,
    model_name: str,
    strategy_id: str,
    checkpoint_path: Path,
    dataset: AnnotatedSubset,
    frozen_metrics: MetricBuckets,
    percentiles: list[int],
    layer: int,
) -> dict[AnalysisMetric, dict[int | None, StrategyMetricSummary]]:
    """Analyze one (model, strategy) checkpoint against the frozen baseline."""
    method = DEFAULT_METHOD[model_name].value

    print(f"  Loading fine-tuned {model_name} [{strategy_id}] from {checkpoint_path}...")
    finetuned_model = load_finetuned_model(
        model_name,
        checkpoint_path=checkpoint_path,
        strategy_id=strategy_id,
    )

    print("  Computing fine-tuned metrics...")
    finetuned_metrics = compute_model_metric_values(
        finetuned_model,
        dataset,
        model_name,
        percentiles,
        layer,
    )

    del finetuned_model
    clear_memory()

    results: dict[AnalysisMetric, dict[int | None, StrategyMetricSummary]] = {}
    for metric_name, config in METRIC_CONFIGS.items():
        bucket_keys = tuple(percentiles) if config.percentile_dependent else (None,)
        results[metric_name] = {}
        for bucket_key in bucket_keys:
            frozen_dict = dict(frozen_metrics[metric_name][bucket_key])
            finetuned_dict = dict(finetuned_metrics[metric_name][bucket_key])
            results[metric_name][bucket_key] = summarize_metric_delta(
                model_name=model_name,
                strategy_id=strategy_id,
                metric=metric_name,
                percentile=bucket_key,
                method=method,
                frozen_values=frozen_dict,
                finetuned_values=finetuned_dict,
            )

    return results


def apply_holm_correction(
    model_results: dict[str, StrategyMetricResults],
    *,
    metric: AnalysisMetric,
    percentile: int | None,
    alpha: float = 0.05,
) -> None:
    """Apply Holm correction across all discovered rows for one metric bucket."""
    keys: list[tuple[str, str]] = []
    p_values: list[float] = []
    family_id = build_correction_family_id(metric, percentile)

    for model_name, strategy_results in model_results.items():
        for strategy_id, metric_results in strategy_results.items():
            row = metric_results.get(metric, {}).get(percentile)
            if row is None:
                continue
            keys.append((model_name, strategy_id))
            p_values.append(row.p_value)

    if not p_values:
        return

    family_size = len(p_values)
    corrected = multiple_comparison_correction(p_values, method="holm", alpha=alpha)
    for (model_name, strategy_id), (corrected_p, significant) in zip(keys, corrected, strict=True):
        row = model_results[model_name][strategy_id][metric][percentile]
        row.corrected_p_value = corrected_p
        row.significant = significant
        row.correction_method = "holm"
        row.correction_family_id = family_id
        row.correction_family_size = family_size


def compare_strategies_within_model(
    *,
    model_name: str,
    strategy_results: StrategyMetricResults,
    percentiles: list[int],
    alpha: float = 0.05,
) -> list[StrategyPairMetricComparison]:
    """Compute paired strategy comparisons using per-image deltas."""
    rows: list[StrategyPairMetricComparison] = []

    for metric_name, config in METRIC_CONFIGS.items():
        bucket_keys = tuple(percentiles) if config.percentile_dependent else (None,)

        for bucket_key in bucket_keys:
            available_strategies = [
                strategy_id
                for strategy_id, metric_map in strategy_results.items()
                if bucket_key in metric_map.get(metric_name, {})
            ]

            bucket_rows: list[StrategyPairMetricComparison] = []
            for strategy_a, strategy_b in combinations(sorted(available_strategies), 2):
                a_result = strategy_results[strategy_a][metric_name][bucket_key]
                b_result = strategy_results[strategy_b][metric_name][bucket_key]

                deltas_a = np.array([row.delta for row in a_result.per_image])
                deltas_b = np.array([row.delta for row in b_result.per_image])

                comparison = paired_comparison(
                    deltas_a,
                    deltas_b,
                    model_a_name=strategy_a,
                    model_b_name=strategy_b,
                    test="auto",
                )

                bucket_rows.append(
                    StrategyPairMetricComparison(
                        model_name=model_name,
                        metric=metric_name,
                        percentile=bucket_key,
                        strategy_a=strategy_a,
                        strategy_b=strategy_b,
                        mean_delta_difference=float(np.mean(deltas_a - deltas_b)),
                        cohens_d=cohens_d(deltas_a, deltas_b, paired=True),
                        p_value=comparison.p_value,
                        corrected_p_value=None,
                        significant=comparison.significant,
                        test_name=comparison.test_name,
                    )
                )

            if bucket_rows:
                corrected = multiple_comparison_correction(
                    [row.p_value for row in bucket_rows],
                    method="holm",
                    alpha=alpha,
                )
                for row, (corrected_p, significant) in zip(bucket_rows, corrected, strict=True):
                    row.corrected_p_value = corrected_p
                    row.significant = significant
                rows.extend(bucket_rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Q2 metrics between frozen and fine-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Specific models to analyze. Default: all fine-tunable models with checkpoints.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[s.value for s in FINETUNE_STRATEGIES],
        choices=[s.value for s in FINETUNE_STRATEGIES],
        help="Fine-tuning strategies to analyze.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=None,
        help="Single percentile to analyze for IoU. Default: [90, 80, 70, 60, 50].",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to extract attention from. Default: -1 (last layer).",
    )
    parser.add_argument(
        "--include-resnet",
        action="store_true",
        help="Include ResNet-50 (unsupported in this script).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Experiment batch identifier to analyze. Defaults to the active "
            "experiment pointer when available."
        ),
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help=(
            "Include exploratory runs such as annotated-eval validation in the "
            "analysis set. The default primary path excludes them."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to the experiment-scoped "
            "q2_metrics_analysis.json for the selected experiment."
        ),
    )

    return parser.parse_args()


def print_summary_table(
    rows: list[StrategyMetricSummary],
    *,
    metric: AnalysisMetric,
    percentile: int | None,
) -> None:
    """Print a compact console summary for one metric bucket."""
    config = METRIC_CONFIGS[metric]
    percentile_label = f"Percentile {percentile}" if percentile is not None else "Threshold-free"
    print(f"\n{'=' * 108}")
    print(f"Q2 {config.label} Results ({percentile_label})")
    print(f"{'=' * 108}")
    print(
        f"{'Model':<10} {'Strategy':<13} {'Frozen':>10} {'Fine':>10} {'Delta':>10} "
        f"{'95% CI':>22} {'d':>8} {'p(Holm)':>10} {'Sig':>5}"
    )
    print("-" * 108)

    sorted_rows = sorted(
        rows,
        key=lambda row: row.mean_delta,
        reverse=(config.direction == "higher"),
    )
    for row in sorted_rows:
        ci = f"[{row.delta_ci_lower:+.3f}, {row.delta_ci_upper:+.3f}]"
        p_disp = row.corrected_p_value if row.corrected_p_value is not None else row.p_value
        sig = "***" if row.significant else ""
        print(
            f"{row.model_name:<10} {row.strategy_id:<13} "
            f"{row.frozen_mean:>10.3f} {row.finetuned_mean:>10.3f} {row.mean_delta:>+10.3f} "
            f"{ci:>22} {row.cohens_d:>+8.2f} {p_disp:>10.4f} {sig:>5}"
        )


def save_results(results: AnalysisResults, output_path: Path) -> None:
    """Persist the metric-generic Q2 analysis artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "experiment_id": results.experiment_id,
        "split_id": results.split_id,
        ANALYSIS_GIT_COMMIT_SHA_FIELD: results.analysis_git_commit_sha,
        "percentiles": results.percentiles,
        "analyzed_layer": results.analyzed_layer,
        "evaluation_image_count": results.evaluation_image_count,
        "checkpoint_selection_rule": results.checkpoint_selection_rule,
        "result_set_scope": results.result_set_scope,
        "timestamp": results.timestamp,
        "metrics": [asdict(metric) for metric in results.metrics],
        "rows": [],
        "strategy_comparisons": [asdict(row) for row in results.strategy_comparisons],
    }

    for row in results.rows:
        row_data = asdict(row)
        row_data["per_image_deltas"] = {
            item["image_id"]: item["delta"] for item in row_data.pop("per_image")
        }
        payload["rows"].append(row_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalize_q2_analysis_payload(payload, drop_legacy=True), f, indent=2)

    print(f"\nResults saved to: {output_path}")


def update_experiment_analysis_outputs(
    *,
    experiment_id: str,
    split_id: str | None,
    output_path: Path,
    compatibility_output: Path,
    include_exploratory: bool,
    analysis_git_commit_sha: str | None,
) -> None:
    """Update the run matrix and active pointer after saving Q2 analysis."""
    experiment_paths = get_experiment_paths(experiment_id)
    run_matrix = load_run_matrix(experiment_id)
    runs = dict(run_matrix.get("runs", {}))
    for run_entry in runs.values():
        if run_entry.get("run_scope") != "primary" and not include_exploratory:
            continue
        run_entry["analysis_artifact_paths"] = {
            "q2_metrics": repo_relative_path(output_path),
            "q2_delta_iou": repo_relative_path(compatibility_output),
        }
        run_entry[ANALYSIS_GIT_COMMIT_SHA_FIELD] = analysis_git_commit_sha
    save_run_matrix(
        experiment_id,
        {
            **run_matrix,
            "split_id": split_id,
            "runs": runs,
        },
    )
    write_active_experiment(
        experiment_id,
        split_id=split_id,
        run_matrix_path=experiment_paths.run_matrix_path,
        q2_metrics_path=output_path,
        q2_delta_iou_path=compatibility_output,
    )


def main() -> None:
    from datetime import datetime

    args = parse_args()
    percentiles = [args.percentile] if args.percentile else [90, 80, 70, 60, 50]
    active_experiment = load_active_experiment()
    experiment_id = args.experiment_id or (active_experiment or {}).get("experiment_id")
    if args.experiment_id is not None and experiment_id is not None:
        refreshed = refresh_experiment_training_provenance(experiment_id)
        print(
            "Refreshed training provenance fields: "
            f"{refreshed['manifests']} manifests, "
            f"{refreshed['fine_tuning_results']} fine_tuning_results.json, "
            f"{refreshed['run_matrix']} run_matrix.json"
        )

    if args.models:
        requested_models = [model for model in args.models if model in FINETUNE_MODELS]
        invalid_models = [model for model in args.models if model not in FINETUNE_MODELS]
        if invalid_models:
            print(f"Skipping non-fine-tunable models: {invalid_models}")
    else:
        requested_models = sorted(FINETUNE_MODELS)

    if args.include_resnet:
        print("ResNet-50 uses Grad-CAM and is not supported by this script. Skipping.")

    discovered, split_id = discover_strategy_checkpoints(
        requested_models,
        args.strategies,
        experiment_id=experiment_id,
        include_exploratory=args.include_exploratory,
    )
    if not discovered:
        print("No strategy checkpoints found. Exiting.")
        return

    result_set_scope = "exploratory" if args.include_exploratory else "primary"
    print(f"Models to analyze: {sorted(discovered.keys())}")
    print(f"Strategies: {args.strategies}")
    print(f"Percentiles (IoU only): {percentiles}")
    print(f"Layer: {args.layer}")
    if experiment_id:
        print(f"Experiment ID: {experiment_id}")
        print(f"Result set scope: {result_set_scope}")

    print("\nLoading annotated dataset...")
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"  {len(dataset)} images with expert bounding boxes")

    all_results: dict[str, StrategyMetricResults] = {}

    for model_name, strategy_paths in discovered.items():
        print(f"\n{'=' * 64}")
        print(f"Analyzing {model_name.upper()}")
        print(f"{'=' * 64}")

        print(f"  Loading frozen {model_name}...")
        frozen_model = create_model(model_name)

        print("  Computing frozen metrics...")
        frozen_metrics = compute_model_metric_values(
            frozen_model,
            dataset,
            model_name,
            percentiles,
            args.layer,
        )
        del frozen_model
        clear_memory()

        strategy_results: StrategyMetricResults = {}
        for strategy_id in sorted(strategy_paths):
            checkpoint_path = strategy_paths[strategy_id]
            parsed_strategy = parse_strategy_from_checkpoint_name(checkpoint_path, model_name)
            effective_strategy = strategy_id if parsed_strategy == "unknown" else parsed_strategy
            try:
                strategy_results[strategy_id] = analyze_model_strategy(
                    model_name=model_name,
                    strategy_id=effective_strategy,
                    checkpoint_path=checkpoint_path,
                    dataset=dataset,
                    frozen_metrics=frozen_metrics,
                    percentiles=percentiles,
                    layer=args.layer,
                )
            except Exception as exc:
                print(f"  Error analyzing {model_name}/{strategy_id}: {exc}")

        if strategy_results:
            all_results[model_name] = strategy_results

    if not all_results:
        print("\nNo model/strategy pairs were successfully analyzed.")
        return

    print("\nApplying Holm correction across model/strategy rows...")
    for metric_name, config in METRIC_CONFIGS.items():
        bucket_keys = tuple(percentiles) if config.percentile_dependent else (None,)
        for bucket_key in bucket_keys:
            apply_holm_correction(
                all_results,
                metric=metric_name,
                percentile=bucket_key,
            )

    strategy_comparisons: list[StrategyPairMetricComparison] = []
    print("Computing within-model cross-strategy comparisons...")
    for model_name, strategy_results in all_results.items():
        strategy_comparisons.extend(
            compare_strategies_within_model(
                model_name=model_name,
                strategy_results=strategy_results,
                percentiles=percentiles,
            )
        )

    flat_rows = [
        row
        for strategy_results in all_results.values()
        for metric_results in strategy_results.values()
        for bucket_results in metric_results.values()
        for row in bucket_results.values()
    ]

    for metric_name, config in METRIC_CONFIGS.items():
        summary_bucket_keys = tuple(percentiles) if config.percentile_dependent else (None,)
        for bucket_key in summary_bucket_keys:
            bucket_rows = [
                row
                for row in flat_rows
                if row.metric == metric_name and row.percentile == bucket_key
            ]
            if bucket_rows:
                print_summary_table(bucket_rows, metric=metric_name, percentile=bucket_key)

    analyzed_layer = resolve_layer_index(next(iter(discovered.keys())), args.layer)
    results = AnalysisResults(
        percentiles=percentiles,
        analyzed_layer=analyzed_layer,
        experiment_id=experiment_id,
        split_id=split_id,
        analysis_git_commit_sha=get_git_commit_sha(),
        evaluation_image_count=len(dataset),
        checkpoint_selection_rule=(
            "best classification validation accuracy on shared non-annotated validation split"
        ),
        result_set_scope=result_set_scope,
        metrics=list(METRIC_CONFIGS.values()),
        rows=flat_rows,
        strategy_comparisons=strategy_comparisons,
        timestamp=datetime.now().isoformat(),
    )

    if args.output is not None:
        output_path = args.output
    elif experiment_id:
        output_path = get_experiment_paths(experiment_id).q2_metrics_path
    else:
        output_path = RESULTS_PATH / "q2_metrics_analysis.json"
    save_results(results, output_path)

    if experiment_id:
        experiment_paths = get_experiment_paths(experiment_id)
        compatibility_output = experiment_paths.q2_delta_iou_path
        if output_path != compatibility_output:
            save_results(results, compatibility_output)

        update_experiment_analysis_outputs(
            experiment_id=experiment_id,
            split_id=split_id,
            output_path=output_path,
            compatibility_output=compatibility_output,
            include_exploratory=args.include_exploratory,
            analysis_git_commit_sha=results.analysis_git_commit_sha,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
