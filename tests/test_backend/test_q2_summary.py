"""Tests for /api/metrics/q2_summary endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_q2_summary_success() -> None:
    payload = {
        "metric": "mse",
        "label": "MSE",
        "direction": "lower",
        "percentile_dependent": False,
        "selected_percentile": None,
        "analysis_git_commit_sha": "analysissha",
        "analyzed_layer": 11,
        "timestamp": "2026-03-06T00:00:00",
        "rows": [
            {
                "model_name": "dinov2",
                "strategy_id": "lora",
                "metric": "mse",
                "label": "MSE",
                "direction": "lower",
                "percentile_dependent": False,
                "percentile": None,
                "method": "cls",
                "frozen_mean": 0.30,
                "finetuned_mean": 0.28,
                "mean_delta": -0.02,
                "std_delta": 0.01,
                "delta_ci_lower": -0.03,
                "delta_ci_upper": -0.01,
                "cohens_d": -0.5,
                "p_value": 0.01,
                "corrected_p_value": 0.02,
                "significant": True,
                "test_name": "paired_ttest",
                "num_images": 139,
            }
        ],
        "strategy_comparisons": [
            {
                "model_name": "dinov2",
                "metric": "mse",
                "percentile": None,
                "strategy_a": "linear_probe",
                "strategy_b": "lora",
                "mean_delta_difference": -0.04,
                "cohens_d": -0.8,
                "p_value": 0.01,
                "corrected_p_value": 0.02,
                "significant": True,
                "test_name": "paired_ttest",
            }
        ],
    }

    with patch("app.backend.routers.metrics.metrics_service.get_q2_summary", return_value=payload):
        response = client.get("/api/metrics/q2_summary", params={"metric": "mse", "percentile": 90})

    assert response.status_code == 200
    body = response.json()
    assert body["metric"] == "mse"
    assert body["selected_percentile"] is None
    assert body["analysis_git_commit_sha"] == "analysissha"
    assert body["rows"][0]["strategy_id"] == "lora"


def test_q2_summary_unavailable_returns_503() -> None:
    with patch("app.backend.routers.metrics.metrics_service.get_q2_summary", return_value=None):
        response = client.get("/api/metrics/q2_summary")

    assert response.status_code == 503
