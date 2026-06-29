"""Tests for the Q1 continuous baseline comparison script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "scripts"
        / "analyze_q1_continuous_baselines.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_q1_continuous_baselines", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_leaderboards(module: ModuleType) -> dict[str, list[dict[str, object]]]:
    return {
        "mse": [
            {
                "rank": 1,
                "model": "clip",
                "metric": "mse",
                "score": 0.0211,
                "best_layer": "layer6",
                "method_used": "cls",
            },
            {
                "rank": 2,
                "model": "dinov3",
                "metric": "mse",
                "score": 0.0270,
                "best_layer": "layer0",
                "method_used": "cls",
            },
            {
                "rank": 3,
                "model": "siglip",
                "metric": "mse",
                "score": 0.0175,
                "best_layer": "layer6",
                "method_used": "mean",
            },
        ],
        "kl": [
            {
                "rank": 1,
                "model": "dinov3",
                "metric": "kl",
                "score": 2.3247,
                "best_layer": "layer11",
                "method_used": "cls",
            },
            {
                "rank": 2,
                "model": "clip",
                "metric": "kl",
                "score": 2.9122,
                "best_layer": "layer0",
                "method_used": "cls",
            },
            {
                "rank": 3,
                "model": "siglip",
                "metric": "kl",
                "score": 3.5000,
                "best_layer": "layer4",
                "method_used": "mean",
            },
        ],
        "emd": [
            {
                "rank": 1,
                "model": "dinov3",
                "metric": "emd",
                "score": 0.2600,
                "best_layer": "layer11",
                "method_used": "cls",
            },
            {
                "rank": 2,
                "model": "clip",
                "metric": "emd",
                "score": 0.3261,
                "best_layer": "layer0",
                "method_used": "cls",
            },
            {
                "rank": 3,
                "model": "siglip",
                "metric": "emd",
                "score": 0.3538,
                "best_layer": "layer4",
                "method_used": "mean",
            },
        ],
    }


def test_collect_leaderboards_uses_default_method_ranking_mode() -> None:
    module = _load_module()

    class FakeService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_leaderboard(self, *, metric: str, ranking_mode: str) -> list[dict[str, object]]:
            self.calls.append((metric, ranking_mode))
            return []

    service = FakeService()
    leaderboards = module.collect_leaderboards(service)

    assert leaderboards == {"mse": [], "kl": [], "emd": []}
    assert service.calls == [
        ("mse", "default_method"),
        ("kl", "default_method"),
        ("emd", "default_method"),
    ]


def test_build_payload_captures_surprises_and_cross_metric_findings() -> None:
    module = _load_module()
    leaderboards = _make_leaderboards(module)

    payload = module.build_comparison_payload(
        leaderboards,
        model_order=["dinov3", "clip", "siglip"],
        generated_at="2026-04-08T12:00:00+00:00",
    )

    dinov3_metrics = payload["models"]["dinov3"]["metrics"]
    assert dinov3_metrics["mse"]["beats_all_baselines"] is True
    assert dinov3_metrics["kl"]["beats_all_baselines"] is True
    assert dinov3_metrics["emd"]["beats_all_baselines"] is True

    clip_metrics = payload["models"]["clip"]["metrics"]
    assert clip_metrics["mse"]["surprises"] == ["beats_all_baselines"]
    assert clip_metrics["kl"]["surprises"] == ["beats_random_but_not_stronger_priors"]
    assert clip_metrics["emd"]["surprises"] == ["beats_only_random"]
    assert clip_metrics["emd"]["passes_baselines"] == {
        "random": True,
        "center_gaussian": False,
        "saliency_prior": False,
        "sobel_edge": False,
    }

    siglip_metrics = payload["models"]["siglip"]["metrics"]
    assert siglip_metrics["mse"]["beats_all_baselines"] is True
    assert siglip_metrics["kl"]["surprises"] == ["worse_than_random"]
    assert siglip_metrics["emd"]["surprises"] == ["worse_than_random"]

    finding_types_by_model = {
        (finding.get("model"), finding["type"])
        for finding in payload["cross_metric_findings"]
    }
    assert ("dinov3", "consistent_strength") in finding_types_by_model
    assert ("clip", "mse_vs_distribution_gap") in finding_types_by_model
    assert ("siglip", "worse_than_random_cross_metric") in finding_types_by_model

    summaries = [finding["summary"] for finding in payload["headline_findings"]]
    assert any("MSE: clip, dinov3, siglip beat all four baselines." in summary for summary in summaries)
    assert any("KL: siglip score worse than the random baseline." in summary for summary in summaries)


def test_save_artifacts_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_module()
    leaderboards = _make_leaderboards(module)
    payload = module.build_comparison_payload(
        leaderboards,
        model_order=["dinov3", "clip", "siglip"],
        generated_at="2026-04-08T12:00:00+00:00",
    )

    json_path, markdown_path = module.save_artifacts(
        payload,
        output_dir=tmp_path,
        json_name="q1_continuous_baseline_comparison.json",
        markdown_name="q1_continuous_baseline_summary.md",
    )

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown_text = markdown_path.read_text(encoding="utf-8")

    assert json_payload["generated_at"] == "2026-04-08T12:00:00+00:00"
    assert "cross_metric_findings" in json_payload
    assert "## Cross-metric Divergences" in markdown_text
    assert "clip beats all four baselines on MSE but has a weaker distribution-level story" in markdown_text
    assert "| Rank | Model | Score | Best layer | Method | Beats |" in markdown_text
