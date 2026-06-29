"""Backend API tests for Q3 per-head endpoints and raw-attention head validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)

IMAGE_ID = "Q1234_test"


class TestRawAttentionHeadValidation:
    """The raw attention endpoint should validate per-head requests cleanly."""

    def test_models_endpoint_reports_models_with_per_head_cache(self) -> None:
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.list_models_with_per_head_cache.return_value = ["clip", "dinov2"]
            mock_attention_service.list_q3_variant_per_head_availability.return_value = {
                "clip": {
                    "frozen": True,
                    "linear_probe": False,
                    "lora": True,
                    "full": False,
                },
                "dinov2": {
                    "frozen": True,
                    "linear_probe": True,
                    "lora": True,
                    "full": True,
                },
            }

            response = client.get("/api/attention/models")

        assert response.status_code == 200
        body = response.json()
        assert body["per_head_available_models"] == ["dinov2", "clip"]
        assert body["q3_per_head_variant_availability"]["dinov2"]["linear_probe"] is True
        assert body["q3_per_head_variant_availability"]["clip"]["full"] is False

    def test_models_endpoint_avoids_cache_dataset_enumeration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from app.backend.services.attention_service import AttentionCache, attention_service

        cache = AttentionCache(tmp_path / "attention.h5")
        cache.store("dinov2", "layer0", IMAGE_ID, torch.ones((2, 2)), variant="cls_head0")

        monkeypatch.setattr(attention_service, "_cache", cache)
        monkeypatch.setattr(attention_service, "_per_head_available_models_cache", None)
        monkeypatch.setattr(attention_service, "_q3_variant_availability_cache", None)
        monkeypatch.setattr(attention_service, "_per_head_available_models_signature", None)

        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("AttentionCache.list_cached() should not be called")

        monkeypatch.setattr(AttentionCache, "list_cached", fail)

        response = client.get("/api/attention/models")

        assert response.status_code == 200
        body = response.json()
        assert body["per_head_available_models"] == ["dinov2"]
        assert body["q3_per_head_variant_availability"]["dinov2"]["frozen"] is True

    def test_rejects_head_for_rollout_method(self) -> None:
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.exists.return_value = True

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 0, "method": "rollout", "head": 3},
            )

        assert response.status_code == 400
        assert "head parameter not supported" in response.json()["detail"]

    def test_rejects_head_for_resnet(self) -> None:
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.exists.return_value = True

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "resnet50", "layer": 0, "method": "gradcam", "head": 1},
            )

        assert response.status_code == 400
        assert "head parameter not supported" in response.json()["detail"]

    def test_passes_head_to_attention_service(self) -> None:
        payload = {
            "attention": [0.1, 0.2, 0.3, 0.4],
            "shape": [2, 2],
            "min_value": 0.1,
            "max_value": 0.4,
        }
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.resolve_variant.return_value = "cls_head5"
            mock_attention_service.exists.return_value = True
            mock_attention_service.get_raw_attention.return_value = payload

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 0, "method": "cls", "head": 5},
            )

        assert response.status_code == 200
        assert response.json()["shape"] == [2, 2]
        mock_attention_service.get_raw_attention.assert_called_once_with(
            image_id=IMAGE_ID,
            model="dinov2",
            layer=0,
            method="cls",
            head=5,
        )


class TestQ3MetricsApi:
    """Q3 endpoints should expose metric-generic per-head payloads."""

    def test_head_ranking_endpoint_returns_service_payload(self) -> None:
        payload = {
            "model": "dinov2",
            "variant": "frozen",
            "layer": "layer11",
            "method": "cls",
            "metric": "iou",
            "direction": "higher",
            "percentile": 90,
            "supported": True,
            "reason": None,
            "heads": [
                {
                    "head": 3,
                    "mean_score": 0.42,
                    "std_score": 0.04,
                    "mean_rank": 1.2,
                    "top1_count": 6,
                    "top3_count": 11,
                    "image_count": 12,
                }
            ],
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_head_ranking.return_value = payload

            response = client.get(
                "/api/metrics/model/dinov2/head_ranking",
                params={"layer": 11, "metric": "iou", "percentile": 90, "variant": "frozen"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["metric"] == "iou"
        assert body["heads"][0]["head"] == 3

    def test_head_feature_matrix_endpoint_returns_unsupported_payload(self) -> None:
        payload: dict[str, object] = {
            "model": "resnet50",
            "variant": "frozen",
            "layer": "layer0",
            "method": None,
            "metric": "coverage",
            "direction": "higher",
            "percentile": 90,
            "supported": False,
            "reason": "Q3 per-head analysis is not supported for model 'resnet50'.",
            "heads": [],
            "features": [],
            "total_feature_types": 0,
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_head_feature_matrix.return_value = payload

            response = client.get(
                "/api/metrics/model/resnet50/head_feature_matrix",
                params={"layer": 0, "metric": "coverage", "percentile": 90, "variant": "frozen"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["supported"] is False
        assert body["features"] == []

    def test_image_head_ranking_endpoint_returns_selection_payload(self) -> None:
        payload = {
            "image_id": IMAGE_ID,
            "model": "dinov2",
            "variant": "lora",
            "layer": "layer11",
            "method": "cls",
            "metric": "coverage",
            "direction": "higher",
            "percentile": 90,
            "selection": {
                "mode": "bbox",
                "bbox_index": 2,
                "bbox_label": "Window",
            },
            "supported": True,
            "reason": None,
            "heads": [
                {"head": 7, "score": 0.83},
                {"head": 3, "score": 0.74},
            ],
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_image_head_ranking.return_value = payload

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/head_ranking",
                params={
                    "model": "dinov2",
                    "layer": 11,
                    "metric": "coverage",
                    "percentile": 90,
                    "variant": "lora",
                    "bbox_index": 2,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["selection"]["mode"] == "bbox"
        assert body["selection"]["bbox_label"] == "Window"
        assert body["heads"][0]["head"] == 7

    def test_image_head_ranking_endpoint_returns_400_for_bad_bbox(self) -> None:
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_image_head_ranking.side_effect = ValueError("bbox_index 4 out of range")

            response = client.get(
                f"/api/metrics/{IMAGE_ID}/head_ranking",
                params={
                    "model": "dinov2",
                    "layer": 11,
                    "variant": "frozen",
                    "bbox_index": 4,
                },
            )

        assert response.status_code == 400
        assert "bbox_index 4 out of range" in response.json()["detail"]

    def test_head_exemplars_endpoint_returns_service_payload(self) -> None:
        payload = {
            "model": "dinov2",
            "variant": "lora",
            "layer": "layer11",
            "metric": "iou",
            "direction": "higher",
            "percentile": 90,
            "head": 3,
            "feature_label": 7,
            "feature_name": "Door",
            "supported": True,
            "reason": None,
            "candidates": [
                {
                    "image_id": "Q1.jpg",
                    "score": 0.72,
                    "thumbnail_url": "/api/images/Q1.jpg/thumbnail",
                    "style_names": ["Gothic"],
                    "matching_bbox_indices": [1],
                    "default_bbox_index": 1,
                }
            ],
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_head_exemplars.return_value = payload

            response = client.get(
                "/api/metrics/model/dinov2/head_exemplars",
                params={
                    "layer": 11,
                    "metric": "iou",
                    "percentile": 90,
                    "variant": "lora",
                    "head": 3,
                    "feature_label": 7,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["variant"] == "lora"
        assert body["candidates"][0]["default_bbox_index"] == 1
        mock_metrics_service.get_head_exemplars.assert_called_once_with(
            model="dinov2",
            layer="layer11",
            head=3,
            percentile=90,
            metric="iou",
            variant="lora",
            feature_label=7,
            limit=12,
        )
