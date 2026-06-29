"""Backend API tests for MSE-enabled metrics responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.backend.main import app
from ssl_attention.data.annotations import BoundingBox, ImageAnnotation

client = TestClient(app)

IMAGE_ID = "Q1234_test"


def _image_metrics_payload(model: str) -> dict:
    return {
        "image_id": IMAGE_ID,
        "model": model,
        "layer": "layer0",
        "percentile": 90,
        "iou": 0.5,
        "coverage": 0.6,
        "mse": 0.0125,
        "kl": 0.034,
        "emd": 0.056,
        "attention_area": 0.4,
        "annotation_area": 0.3,
        "method": "cls",
    }


class TestMetricsEndpointsExposeMse:
    """Metrics endpoints should surface MSE alongside existing fields."""

    def test_image_metrics_response_includes_mse(self):
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics:
            mock_metrics.db_exists = True
            mock_metrics.get_image_metrics.return_value = _image_metrics_payload("dinov2")

            response = client.get(
                f"/api/metrics/{IMAGE_ID}",
                params={"model": "dinov2", "layer": 0, "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 200
        assert response.json()["mse"] == 0.0125
        assert response.json()["kl"] == 0.034
        assert response.json()["emd"] == 0.056

    def test_image_metrics_accepts_finetuned_model(self):
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics:
            mock_metrics.db_exists = True
            mock_metrics.get_image_metrics.return_value = _image_metrics_payload("dinov2_finetuned")

            response = client.get(
                f"/api/metrics/{IMAGE_ID}",
                params={"model": "dinov2_finetuned", "layer": 0, "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 200
        assert response.json()["model"] == "dinov2_finetuned"
        mock_metrics.get_image_metrics.assert_called_once_with(
            IMAGE_ID,
            "dinov2_finetuned",
            "layer0",
            90,
            method="cls",
        )

    def test_bbox_metrics_response_includes_computed_mse(self):
        annotation = ImageAnnotation(
            image_id=IMAGE_ID,
            styles=(),
            bboxes=(
                BoundingBox(left=0.2, top=0.2, width=0.3, height=0.3, label=0, group_label=0),
            ),
        )

        with (
            patch("app.backend.routers.metrics.image_service") as mock_image_service,
            patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = annotation
            mock_metrics_service.get_bbox_metrics.return_value = _image_metrics_payload("dinov2")

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/bbox/0",
                params={"model": "dinov2", "layer": 0, "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert "mse" in payload
        assert "kl" in payload
        assert "emd" in payload
        assert isinstance(payload["mse"], float)
        assert isinstance(payload["kl"], float)
        assert isinstance(payload["emd"], float)


class TestComparisonEndpointsExposeMse:
    """Comparison endpoints should include MSE-related fields."""

    def test_compare_models_results_include_mse(self):
        annotation = MagicMock()
        annotation.bboxes = []

        with (
            patch("app.backend.routers.comparison.image_service") as mock_image_service,
            patch("app.backend.routers.comparison.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = annotation
            mock_image_service.heatmap_exists.return_value = False
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_image_metrics.side_effect = [
                _image_metrics_payload("dinov2"),
                _image_metrics_payload("clip"),
            ]

            response = client.get(
                "/api/compare/models",
                params={
                    "image_id": IMAGE_ID,
                    "models": ["dinov2", "clip"],
                    "layer": 0,
                    "percentile": 90,
                    "method": "cls",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["selection"] == {
            "mode": "union",
            "bbox_index": None,
            "bbox_label": None,
        }
        assert payload["unavailable_models"] == {}
        assert all("mse" in result for result in payload["results"])
        assert all("kl" in result for result in payload["results"])
        assert all("emd" in result for result in payload["results"])

    def test_all_models_summary_uses_metric_and_best_score_fields(self):
        with patch("app.backend.routers.comparison.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_leaderboard.return_value = [
                {
                    "rank": 1,
                    "model": "dinov2",
                    "metric": "emd",
                    "score": 0.08,
                    "best_layer": "layer1",
                    "method_used": "cls",
                }
            ]
            mock_metrics_service.get_layer_progression.return_value = {
                "model": "dinov2",
                "metric": "emd",
                "percentile": 90,
                "method": "cls",
                "layers": ["layer0", "layer1"],
                "scores": [0.12, 0.08],
                "best_layer": "layer1",
                "best_score": 0.08,
            }

            response = client.get("/api/compare/all_models_summary", params={"percentile": 90, "metric": "emd"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["metric"] == "emd"
        assert payload["ranking_mode"] == "default_method"
        assert payload["method"] is None
        assert payload["excluded_models"] == []
        assert payload["leaderboard"][0]["score"] == 0.08
        assert payload["leaderboard"][0]["method_used"] == "cls"
        assert payload["models"]["dinov2"]["best_score"] == 0.08
        assert payload["models"]["dinov2"]["method_used"] == "cls"
        assert "best_iou" not in payload["models"]["dinov2"]
