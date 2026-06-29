"""Backend API tests for the image-detail layer progression endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)

IMAGE_ID = "Q1234_test.jpg"


def _metric_descriptors() -> list[dict]:
    return [
        {
            "key": "iou",
            "label": "IoU Score",
            "direction": "higher",
            "default_enabled": True,
            "percentile_dependent": True,
        },
        {
            "key": "coverage",
            "label": "Coverage",
            "direction": "higher",
            "default_enabled": True,
            "percentile_dependent": False,
        },
        {
            "key": "mse",
            "label": "MSE",
            "direction": "lower",
            "default_enabled": True,
            "percentile_dependent": False,
        },
        {
            "key": "kl",
            "label": "KL Divergence",
            "direction": "lower",
            "default_enabled": True,
            "percentile_dependent": False,
        },
        {
            "key": "emd",
            "label": "EMD",
            "direction": "lower",
            "default_enabled": True,
            "percentile_dependent": False,
        },
    ]


def _union_progression_payload() -> dict:
    return {
        "image_id": IMAGE_ID,
        "model": "dinov2",
        "method": "cls",
        "percentile": 90,
        "selection": {
            "mode": "union",
            "bbox_index": None,
            "bbox_label": None,
        },
        "metrics": _metric_descriptors(),
        "layers": [
            {
                "layer": 0,
                "layer_key": "layer0",
                "values": {"iou": 0.12, "coverage": 0.44, "mse": 0.08, "kl": 0.11, "emd": 0.09},
            },
            {
                "layer": 1,
                "layer_key": "layer1",
                "values": {"iou": 0.18, "coverage": 0.46, "mse": 0.06, "kl": 0.08, "emd": 0.07},
            },
        ],
    }


def _bbox_progression_payload() -> dict:
    return {
        "image_id": IMAGE_ID,
        "model": "resnet50",
        "method": "gradcam",
        "percentile": 90,
        "selection": {
            "mode": "bbox",
            "bbox_index": 0,
            "bbox_label": "Window",
        },
        "metrics": _metric_descriptors(),
        "layers": [
            {
                "layer": 0,
                "layer_key": "layer0",
                "values": {"iou": 0.22, "coverage": 0.51, "mse": 0.03, "kl": 0.05, "emd": 0.04},
            },
            {
                "layer": 1,
                "layer_key": "layer1",
                "values": {"iou": 0.24, "coverage": 0.52, "mse": 0.028, "kl": 0.046, "emd": 0.038},
            },
            {
                "layer": 2,
                "layer_key": "layer2",
                "values": {"iou": 0.27, "coverage": 0.55, "mse": 0.024, "kl": 0.039, "emd": 0.032},
            },
            {
                "layer": 3,
                "layer_key": "layer3",
                "values": {"iou": 0.29, "coverage": 0.57, "mse": 0.022, "kl": 0.034, "emd": 0.029},
            },
        ],
    }


class TestImageDetailProgressionEndpoint:
    """Verify API behavior for extensible image-detail layer progression."""

    def test_union_progression_returns_descriptor_driven_payload(self):
        with (
            patch("app.backend.routers.metrics.image_service") as mock_image_service,
            patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = object()
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_image_layer_progression.return_value = _union_progression_payload()

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/progression",
                params={"model": "dinov2", "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["selection"]["mode"] == "union"
        assert [metric["key"] for metric in payload["metrics"]] == ["iou", "coverage", "mse", "kl", "emd"]
        assert payload["layers"][0]["values"]["coverage"] == 0.44

    def test_bbox_progression_returns_bbox_selection_and_short_model_layers(self):
        with (
            patch("app.backend.routers.metrics.image_service") as mock_image_service,
            patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = object()
            mock_metrics_service.get_bbox_layer_progression.return_value = _bbox_progression_payload()

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/progression",
                params={"model": "resnet50", "percentile": 90, "method": "gradcam", "bbox_index": 0},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["selection"] == {"mode": "bbox", "bbox_index": 0, "bbox_label": "Window"}
        assert [layer["layer"] for layer in payload["layers"]] == [0, 1, 2, 3]

    def test_bbox_progression_invalid_index_returns_400(self):
        with (
            patch("app.backend.routers.metrics.image_service") as mock_image_service,
            patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = object()
            mock_metrics_service.get_bbox_layer_progression.side_effect = ValueError("bbox_index 3 out of range")

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/progression",
                params={"model": "dinov2", "percentile": 90, "method": "cls", "bbox_index": 3},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "bbox_index 3 out of range"

    def test_union_progression_requires_metrics_db(self):
        with (
            patch("app.backend.routers.metrics.image_service") as mock_image_service,
            patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service,
        ):
            mock_image_service.get_annotation.return_value = object()
            mock_metrics_service.db_exists = False

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/progression",
                params={"model": "dinov2", "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "Metrics database not available."

    def test_progression_returns_404_when_annotation_missing(self):
        with patch("app.backend.routers.metrics.image_service") as mock_image_service:
            mock_image_service.get_annotation.return_value = None

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/progression",
                params={"model": "dinov2", "percentile": 90, "method": "cls"},
            )

        assert response.status_code == 404
        assert response.json()["detail"] == f"Annotation not found for {IMAGE_ID}"
