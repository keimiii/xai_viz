"""Unit tests for strategy-aware Q2 metric analysis helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "analyze_q2_metrics.py"
    spec = importlib.util.spec_from_file_location("analyze_q2_metrics", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_metric_delta_keeps_metric_metadata() -> None:
    module = _load_module()

    result = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="lora",
        metric="mse",
        percentile=None,
        method="cls",
        frozen_values={"a": 0.20, "b": 0.40},
        finetuned_values={"a": 0.18, "b": 0.36},
    )

    assert result.metric == "mse"
    assert result.direction == "lower"
    assert result.percentile is None
    assert result.mean_delta < 0


def test_compare_strategies_within_model_returns_pairwise_rows() -> None:
    module = _load_module()

    baseline = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="linear_probe",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.41, "b": 0.39},
    )
    lora = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="lora",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )

    output = module.compare_strategies_within_model(
        model_name="dinov2",
        strategy_results={
            "linear_probe": {"iou": {90: baseline}},
            "lora": {"iou": {90: lora}},
        },
        percentiles=[90],
    )

    assert len(output) == 1
    row = output[0]
    assert row.metric == "iou"
    assert row.percentile == 90
    assert row.strategy_a == "linear_probe"
    assert row.strategy_b == "lora"
    assert row.corrected_p_value is not None


def test_apply_holm_correction_includes_all_strategies_in_same_family() -> None:
    module = _load_module()

    linear_probe = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="linear_probe",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.4, "b": 0.4},
    )
    lora = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="lora",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )
    full = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="full",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.47, "b": 0.46},
    )

    linear_probe.p_value = 0.001
    lora.p_value = 0.01
    full.p_value = 0.02

    model_results = {
        "clip": {
            "linear_probe": {"iou": {90: linear_probe}},
            "lora": {"iou": {90: lora}},
        },
        "dinov2": {
            "full": {"iou": {90: full}},
        },
    }

    module.apply_holm_correction(model_results, metric="iou", percentile=90)

    expected = module.multiple_comparison_correction([0.001, 0.01, 0.02], method="holm", alpha=0.05)
    family_id = module.build_correction_family_id("iou", 90)

    assert linear_probe.corrected_p_value == expected[0][0]
    assert linear_probe.significant is expected[0][1]
    assert linear_probe.correction_method == "holm"
    assert linear_probe.correction_family_id == family_id
    assert linear_probe.correction_family_size == 3

    assert lora.corrected_p_value == expected[1][0]
    assert lora.significant is expected[1][1]
    assert lora.correction_method == "holm"
    assert lora.correction_family_id == family_id
    assert lora.correction_family_size == 3

    assert full.corrected_p_value == expected[2][0]
    assert full.significant is expected[2][1]
    assert full.correction_method == "holm"
    assert full.correction_family_id == family_id
    assert full.correction_family_size == 3


def test_save_results_serializes_correction_family_metadata(tmp_path: Path) -> None:
    module = _load_module()

    linear_probe = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="linear_probe",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.4, "b": 0.4},
    )
    lora = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="lora",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )

    linear_probe.p_value = 0.001
    lora.p_value = 0.01

    model_results = {
        "clip": {
            "linear_probe": {"iou": {90: linear_probe}},
            "lora": {"iou": {90: lora}},
        },
    }

    module.apply_holm_correction(model_results, metric="iou", percentile=90)

    output_path = tmp_path / "q2_metrics_analysis.json"
    results = module.AnalysisResults(
        experiment_id="exp",
        split_id="split",
        analysis_git_commit_sha="abc123",
        percentiles=[90],
        analyzed_layer=11,
        evaluation_image_count=2,
        checkpoint_selection_rule="best validation accuracy",
        result_set_scope="primary",
        metrics=list(module.METRIC_CONFIGS.values()),
        rows=[linear_probe, lora],
        strategy_comparisons=[],
        timestamp="2026-03-28T00:00:00+08:00",
    )

    module.save_results(results, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = next(item for item in payload["rows"] if item["strategy_id"] == "linear_probe")

    assert payload["analysis_git_commit_sha"] == "abc123"
    assert "git_commit_sha" not in payload
    assert row["correction_method"] == "holm"
    assert row["correction_family_id"] == "cross_model_summary:iou:p90"
    assert row["correction_family_size"] == 2
    assert set(row["per_image_deltas"]) == {"a", "b"}


def test_update_experiment_analysis_outputs_writes_analysis_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    experiment_paths = SimpleNamespace(
        run_matrix_path=tmp_path / "run_matrix.json",
        q2_delta_iou_path=tmp_path / "q2_delta_iou_analysis.json",
    )
    saved: dict[str, object] = {}
    active_experiment_updates: dict[str, object] = {}

    monkeypatch.setattr(module, "get_experiment_paths", lambda experiment_id: experiment_paths)
    monkeypatch.setattr(
        module,
        "load_run_matrix",
        lambda experiment_id: {
            "experiment_id": experiment_id,
            "runs": {
                "primary_run": {"run_scope": "primary", "analysis_artifact_paths": {}, "training_git_commit_sha": "trainsha"},
                "exploratory_run": {
                    "run_scope": "exploratory",
                    "analysis_artifact_paths": {},
                    "training_git_commit_sha": "trainsha",
                },
            },
        },
    )
    monkeypatch.setattr(
        module,
        "save_run_matrix",
        lambda experiment_id, payload: saved.update({"experiment_id": experiment_id, "payload": payload}),
    )
    monkeypatch.setattr(
        module,
        "write_active_experiment",
        lambda experiment_id, **kwargs: active_experiment_updates.update(
            {"experiment_id": experiment_id, **kwargs}
        ),
    )
    monkeypatch.setattr(module, "repo_relative_path", lambda path: path.name)

    output_path = tmp_path / "q2_metrics_analysis.json"
    compatibility_output = tmp_path / "q2_delta_iou_analysis.json"
    module.update_experiment_analysis_outputs(
        experiment_id="exp",
        split_id="split",
        output_path=output_path,
        compatibility_output=compatibility_output,
        include_exploratory=False,
        analysis_git_commit_sha="analysissha",
    )

    assert saved["experiment_id"] == "exp"
    saved_payload = saved["payload"]
    assert isinstance(saved_payload, dict)
    run_entries = saved_payload["runs"]
    assert run_entries["primary_run"]["analysis_git_commit_sha"] == "analysissha"
    assert run_entries["primary_run"]["analysis_artifact_paths"] == {
        "q2_metrics": output_path.name,
        "q2_delta_iou": compatibility_output.name,
    }
    assert "analysis_git_commit_sha" not in run_entries["exploratory_run"]
    assert active_experiment_updates["experiment_id"] == "exp"
    assert active_experiment_updates["q2_metrics_path"] == output_path
    assert active_experiment_updates["q2_delta_iou_path"] == compatibility_output
