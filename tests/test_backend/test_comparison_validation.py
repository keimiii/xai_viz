"""Tests for method validation in comparison and all_models endpoints.

Verifies that:
- compare_models returns 400 when a requested method is incompatible with a model.
- all_models silently skips incompatible models instead of falling back to defaults.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.backend.main import app

client = TestClient(app)


# Shared test image ID
IMAGE_ID = "Q1234_test"


@pytest.fixture(autouse=True)
def _mock_services():
    """Mock services to avoid needing real data files."""
    mock_annotation = MagicMock()
    mock_annotation.bboxes = []

    with (
        patch("app.backend.routers.comparison.image_service") as mock_img_cmp,
        patch("app.backend.routers.comparison.metrics_service") as mock_met_cmp,
        patch("app.backend.routers.metrics.image_service"),
        patch("app.backend.routers.metrics.metrics_service") as mock_met_all,
    ):
        # comparison router mocks
        mock_img_cmp.get_annotation.return_value = mock_annotation
        mock_img_cmp.heatmap_exists.return_value = False
        mock_met_cmp.db_exists = True
        mock_met_cmp.get_image_metrics.return_value = {
            "image_id": IMAGE_ID,
            "model": "dinov2",
            "layer": "layer0",
            "percentile": 90,
            "iou": 0.5,
            "coverage": 0.6,
            "mse": 0.1,
            "kl": 0.2,
            "emd": 0.3,
            "attention_area": 0.4,
            "annotation_area": 0.3,
            "method": "cls",
        }

        # metrics router mocks
        mock_met_all.db_exists = True
        mock_met_all.get_image_metrics.return_value = {
            "image_id": IMAGE_ID,
            "model": "dinov2",
            "layer": "layer0",
            "percentile": 90,
            "iou": 0.5,
            "coverage": 0.6,
            "mse": 0.1,
            "kl": 0.2,
            "emd": 0.3,
            "attention_area": 0.4,
            "annotation_area": 0.3,
            "method": "cls",
        }

        yield {
            "comparison_image_service": mock_img_cmp,
            "comparison_metrics_service": mock_met_cmp,
            "all_models_metrics_service": mock_met_all,
        }


def _bbox_metrics_payload(model: str) -> dict:
    return {
        "image_id": IMAGE_ID,
        "model": model,
        "layer": "layer0",
        "percentile": 90,
        "iou": 0.31,
        "coverage": 0.48,
        "mse": 0.022,
        "kl": 0.08,
        "emd": 0.06,
        "attention_area": 0.21,
        "annotation_area": 0.14,
        "method": "cls",
    }


class TestCompareModelsMethodValidation:
    """compare_models should reject incompatible model+method combos."""

    def test_incompatible_method_returns_400(self):
        """SigLIP does not support rollout — should get 400, not silent fallback."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["siglip2"], "method": "rollout"},
        )
        assert resp.status_code == 400
        assert "not available" in resp.json()["detail"]

    def test_compatible_method_returns_200(self):
        """DINOv2 supports rollout — should succeed."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["dinov2"], "method": "rollout"},
        )
        assert resp.status_code == 200

    def test_resnet_rejects_cls(self):
        """ResNet only supports gradcam — cls should be rejected."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["resnet50"], "method": "cls"},
        )
        assert resp.status_code == 400

    def test_mixed_models_incompatible_method(self):
        """If any model in the list is incompatible, the whole request fails."""
        resp = client.get(
            "/api/compare/models",
            params={
                "image_id": IMAGE_ID,
                "models": ["dinov2", "siglip2"],
                "method": "rollout",
            },
        )
        assert resp.status_code == 400


class TestAllModelsMethodFiltering:
    """all_models should skip incompatible models, not fall back to defaults."""

    def test_rollout_excludes_siglip_and_resnet(self):
        """When method=rollout, siglip, siglip2 and resnet50 should be absent."""
        resp = client.get(
            f"/api/metrics/{IMAGE_ID}/all_models",
            params={"method": "rollout"},
        )
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "siglip" not in models
        assert "siglip2" not in models
        assert "resnet50" not in models

    def test_no_method_includes_all_models(self):
        """Without a method filter, all models should be present."""
        resp = client.get(f"/api/metrics/{IMAGE_ID}/all_models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        # All 7 models should appear (each uses its own default method)
        for m in ("dinov2", "dinov3", "mae", "clip", "siglip", "siglip2", "resnet50"):
            assert m in models, f"Expected {m} in results when no method filter"

    def test_gradcam_only_includes_resnet(self):
        """method=gradcam should only include resnet50."""
        resp = client.get(
            f"/api/metrics/{IMAGE_ID}/all_models",
            params={"method": "gradcam"},
        )
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "resnet50" in models
        # ViT models don't support gradcam
        for m in ("dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"):
            assert m not in models, f"{m} should not appear for method=gradcam"


class TestAllModelsSummaryMethodFiltering:
    """all_models_summary should stay method-consistent when scoped."""

    def test_rollout_summary_filters_incompatible_models_and_forwards_method(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.52,
                "best_layer": "layer1",
                "method_used": "rollout",
            },
            {
                "rank": 2,
                "model": "clip",
                "metric": "iou",
                "score": 0.46,
                "best_layer": "layer1",
                "method_used": "rollout",
            },
        ]
        mock_met_cmp.get_layer_progression.side_effect = [
            {
                "model": "dinov2",
                "metric": "iou",
                "percentile": 90,
                "method": "rollout",
                "layers": ["layer0", "layer1"],
                "scores": [0.38, 0.52],
                "best_layer": "layer1",
                "best_score": 0.52,
            },
            {
                "model": "clip",
                "metric": "iou",
                "percentile": 90,
                "method": "rollout",
                "layers": ["layer0", "layer1"],
                "scores": [0.33, 0.46],
                "best_layer": "layer1",
                "best_score": 0.46,
            },
        ]

        resp = client.get(
            "/api/compare/all_models_summary",
            params={"percentile": 90, "metric": "iou", "method": "rollout"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ranking_mode"] is None
        assert payload["method"] == "rollout"
        assert payload["excluded_models"] == ["siglip", "siglip2", "resnet50"]
        assert [entry["model"] for entry in payload["leaderboard"]] == ["dinov2", "clip"]
        assert all(entry["method_used"] == "rollout" for entry in payload["leaderboard"])
        assert payload["models"]["dinov2"]["method_used"] == "rollout"
        mock_met_cmp.get_leaderboard.assert_called_once_with(90, metric="iou", method="rollout")
        assert mock_met_cmp.get_layer_progression.call_args_list == [
            call("dinov2", 90, method="rollout", metric="iou"),
            call("clip", 90, method="rollout", metric="iou"),
        ]

    def test_default_summary_uses_default_ranking_mode(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.58,
                "best_layer": "layer11",
                "method_used": "cls",
            }
        ]
        mock_met_cmp.get_layer_progression.return_value = {
            "model": "dinov2",
            "metric": "iou",
            "percentile": 90,
            "method": "cls",
            "layers": ["layer0", "layer11"],
            "scores": [0.31, 0.58],
            "best_layer": "layer11",
            "best_score": 0.58,
        }

        resp = client.get(
            "/api/compare/all_models_summary",
            params={"percentile": 90, "metric": "iou"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ranking_mode"] == "default_method"
        assert payload["method"] is None
        assert payload["excluded_models"] == []
        assert payload["leaderboard"][0]["method_used"] == "cls"
        assert payload["models"]["dinov2"]["method_used"] == "cls"
        mock_met_cmp.get_leaderboard.assert_called_once_with(90, metric="iou", ranking_mode="default_method")
        mock_met_cmp.get_layer_progression.assert_called_once_with("dinov2", 90, method="cls", metric="iou")

    def test_best_available_summary_uses_entry_methods(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.62,
                "best_layer": "layer10",
                "method_used": "rollout",
            },
            {
                "rank": 2,
                "model": "clip",
                "metric": "iou",
                "score": 0.55,
                "best_layer": "layer8",
                "method_used": "cls",
            },
        ]
        mock_met_cmp.get_layer_progression.side_effect = [
            {
                "model": "dinov2",
                "metric": "iou",
                "percentile": 90,
                "method": "rollout",
                "layers": ["layer0", "layer10"],
                "scores": [0.42, 0.62],
                "best_layer": "layer10",
                "best_score": 0.62,
            },
            {
                "model": "clip",
                "metric": "iou",
                "percentile": 90,
                "method": "cls",
                "layers": ["layer0", "layer8"],
                "scores": [0.28, 0.55],
                "best_layer": "layer8",
                "best_score": 0.55,
            },
        ]

        resp = client.get(
            "/api/compare/all_models_summary",
            params={"percentile": 90, "metric": "iou", "ranking_mode": "best_available"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ranking_mode"] == "best_available"
        assert payload["method"] is None
        assert payload["excluded_models"] == []
        assert payload["leaderboard"][0]["method_used"] == "rollout"
        assert payload["models"]["dinov2"]["method_used"] == "rollout"
        mock_met_cmp.get_leaderboard.assert_called_once_with(90, metric="iou", ranking_mode="best_available")
        assert mock_met_cmp.get_layer_progression.call_args_list == [
            call("dinov2", 90, method="rollout", metric="iou"),
            call("clip", 90, method="cls", metric="iou"),
        ]

    def test_invalid_summary_method_returns_400(self):
        resp = client.get(
            "/api/compare/all_models_summary",
            params={"percentile": 90, "metric": "iou", "method": "invalid_method"},
        )

        assert resp.status_code == 400
        assert "Invalid method" in resp.json()["detail"]

    def test_summary_rejects_method_and_ranking_mode_together(self):
        resp = client.get(
            "/api/compare/all_models_summary",
            params={"percentile": 90, "metric": "iou", "method": "cls", "ranking_mode": "best_available"},
        )

        assert resp.status_code == 400
        assert "cannot be combined" in resp.json()["detail"]


class TestMetricsLeaderboardRankingModes:
    """metrics/leaderboard should validate and forward ranking semantics."""

    def test_default_leaderboard_uses_default_ranking_mode(self, _mock_services):
        mock_met_all = _mock_services["all_models_metrics_service"]
        mock_met_all.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.58,
                "best_layer": "layer11",
                "method_used": "cls",
            }
        ]

        resp = client.get("/api/metrics/leaderboard", params={"percentile": 90, "metric": "iou"})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload[0]["method_used"] == "cls"
        mock_met_all.get_leaderboard.assert_called_once_with(90, metric="iou", ranking_mode="default_method")

    def test_best_available_leaderboard_forwards_ranking_mode(self, _mock_services):
        mock_met_all = _mock_services["all_models_metrics_service"]
        mock_met_all.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.62,
                "best_layer": "layer10",
                "method_used": "rollout",
            }
        ]

        resp = client.get(
            "/api/metrics/leaderboard",
            params={"percentile": 90, "metric": "iou", "ranking_mode": "best_available"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload[0]["method_used"] == "rollout"
        mock_met_all.get_leaderboard.assert_called_once_with(90, metric="iou", ranking_mode="best_available")

    def test_shared_method_leaderboard_forwards_method(self, _mock_services):
        mock_met_all = _mock_services["all_models_metrics_service"]
        mock_met_all.get_leaderboard.return_value = [
            {
                "rank": 1,
                "model": "dinov2",
                "metric": "iou",
                "score": 0.52,
                "best_layer": "layer1",
                "method_used": "rollout",
            }
        ]

        resp = client.get(
            "/api/metrics/leaderboard",
            params={"percentile": 90, "metric": "iou", "method": "rollout"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload[0]["method_used"] == "rollout"
        mock_met_all.get_leaderboard.assert_called_once_with(90, metric="iou", method="rollout")

    def test_leaderboard_rejects_method_and_ranking_mode_together(self):
        resp = client.get(
            "/api/metrics/leaderboard",
            params={"percentile": 90, "metric": "iou", "method": "cls", "ranking_mode": "best_available"},
        )

        assert resp.status_code == 400
        assert "cannot be combined" in resp.json()["detail"]


class TestFrozenVsFinetunedEndpoint:
    """frozen_vs_finetuned should return usable URLs when caches exist."""

    def test_variants_endpoint_compares_frozen_and_strategy_specific_variants(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned_lora"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/variants",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "left_variant": "frozen",
                "right_variant": "lora",
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["left"]["label"] == "Frozen (Pretrained)"
        assert payload["left"]["available"] is True
        assert payload["right"]["strategy"] == "lora"
        assert payload["right"]["available"] is True
        assert "model=dinov2_finetuned_lora" in payload["right"]["url"]

    def test_variants_endpoint_falls_back_to_legacy_full_overlay_key(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/variants",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "left_variant": "frozen",
                "right_variant": "full",
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["right"]["strategy"] == "full"
        assert payload["right"]["available"] is True
        assert "model=dinov2_finetuned" in payload["right"]["url"]

    def test_variants_endpoint_keeps_non_finetunable_models_as_unavailable(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            return model == "resnet50" and method == "gradcam" and variant in ("overlay", "overlay_bbox")

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/variants",
            params={
                "image_id": IMAGE_ID,
                "model": "resnet50",
                "layer": 0,
                "left_variant": "frozen",
                "right_variant": "full",
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["left"]["available"] is True
        assert payload["right"]["strategy"] == "full"
        assert payload["right"]["available"] is False
        assert payload["right"]["url"] is None

    def test_returns_urls_for_frozen_and_finetuned_when_available(self, _mock_services):
        """When both overlays exist, API should expose both URLs with explicit method."""
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/frozen_vs_finetuned",
            params={"image_id": IMAGE_ID, "model": "dinov2", "layer": 0},
        )
        assert resp.status_code == 200

        payload = resp.json()
        assert payload["frozen"]["available"] is True
        assert payload["finetuned"]["available"] is True
        assert payload["frozen"]["url"] is not None
        assert payload["finetuned"]["url"] is not None
        assert "method=cls" in payload["frozen"]["url"]
        assert "model=dinov2_finetuned" in payload["finetuned"]["url"]

    def test_compare_urls_resolve_to_overlay_success(self, _mock_services):
        """Returned compare URLs should serve overlay images for both variants."""
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _comparison_exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned"}

        mock_img_cmp.heatmap_exists.side_effect = _comparison_exists

        with patch("app.backend.routers.attention.image_service") as mock_attention_image_service:
            mock_attention_image_service.heatmap_exists.side_effect = _comparison_exists
            mock_attention_image_service.load_heatmap.return_value = PILImage.new("RGB", (8, 8), color=(10, 20, 30))

            compare_resp = client.get(
                "/api/compare/frozen_vs_finetuned",
                params={"image_id": IMAGE_ID, "model": "dinov2", "layer": 0},
            )

            assert compare_resp.status_code == 200
            payload = compare_resp.json()

            frozen_resp = client.get(payload["frozen"]["url"])
            finetuned_resp = client.get(payload["finetuned"]["url"])

        assert frozen_resp.status_code == 200
        assert finetuned_resp.status_code == 200
        assert frozen_resp.headers["content-type"] == "image/png"
        assert finetuned_resp.headers["content-type"] == "image/png"
        assert frozen_resp.content
        assert finetuned_resp.content


class TestVariantShiftEndpoint:
    """variants/shift should return numeric frozen-vs-variant attention deltas."""

    def test_returns_shift_payload_for_strategy_specific_variant(self) -> None:
        with patch("app.backend.routers.comparison.attention_service") as mock_attention_service:
            mock_attention_service.exists.side_effect = lambda model, *_args, **kwargs: (
                kwargs.get("method") == "cls"
                and model in {"dinov2", "dinov2_finetuned_lora"}
            )
            mock_attention_service.get_attention_shift.return_value = {
                "shape": [2, 2],
                "shift": [0.2, -0.2, 0.0, 0.6],
                "min_value": -0.2,
                "max_value": 0.6,
                "max_abs_value": 0.6,
            }

            resp = client.get(
                "/api/compare/variants/shift",
                params={
                    "image_id": IMAGE_ID,
                    "model": "dinov2",
                    "layer": 0,
                    "compared_variant": "lora",
                },
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is True
        assert payload["baseline_variant"] == "frozen"
        assert payload["compared_variant"] == "lora"
        assert payload["baseline_model_key"] == "dinov2"
        assert payload["compared_model_key"] == "dinov2_finetuned_lora"
        assert payload["shift"] == pytest.approx([0.2, -0.2, 0.0, 0.6])
        mock_attention_service.get_attention_shift.assert_called_once_with(
            image_id=IMAGE_ID,
            baseline_model="dinov2",
            compared_model="dinov2_finetuned_lora",
            layer=0,
            method="cls",
        )

    def test_falls_back_to_legacy_full_variant_cache_key(self) -> None:
        with patch("app.backend.routers.comparison.attention_service") as mock_attention_service:
            mock_attention_service.exists.side_effect = lambda model, *_args, **kwargs: (
                kwargs.get("method") == "cls"
                and model in {"dinov2", "dinov2_finetuned"}
            )
            mock_attention_service.get_attention_shift.return_value = {
                "shape": [2, 2],
                "shift": [0.1, 0.0, -0.1, 0.2],
                "min_value": -0.1,
                "max_value": 0.2,
                "max_abs_value": 0.2,
            }

            resp = client.get(
                "/api/compare/variants/shift",
                params={
                    "image_id": IMAGE_ID,
                    "model": "dinov2",
                    "layer": 0,
                    "compared_variant": "full",
                },
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is True
        assert payload["compared_variant"] == "full"
        assert payload["compared_model_key"] == "dinov2_finetuned"

    def test_reports_unavailable_shift_when_variant_cache_is_missing(self) -> None:
        with patch("app.backend.routers.comparison.attention_service") as mock_attention_service:
            mock_attention_service.exists.side_effect = lambda model, *_args, **kwargs: (
                kwargs.get("method") == "cls" and model == "dinov2"
            )

            resp = client.get(
                "/api/compare/variants/shift",
                params={
                    "image_id": IMAGE_ID,
                    "model": "dinov2",
                    "layer": 0,
                    "compared_variant": "linear_probe",
                },
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["available"] is False
        assert payload["compared_variant"] == "linear_probe"
        assert payload["shift"] == []
        assert "Compared variant attention is not cached" in payload["reason"]
        mock_attention_service.get_attention_shift.assert_not_called()


class TestCompareModelsBboxMetrics:
    """compare_models should support bbox-scoped metrics without DB dependence."""

    def test_union_compare_includes_selection_metadata(self):
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["dinov2"], "method": "cls"},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["selection"] == {
            "mode": "union",
            "bbox_index": None,
            "bbox_label": None,
        }
        assert payload["unavailable_models"] == {}

    def test_bbox_compare_returns_bbox_selection_without_metrics_db(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.db_exists = False
        mock_met_cmp.get_bbox_label.return_value = "Window"
        mock_met_cmp.get_bbox_metrics.side_effect = [
            _bbox_metrics_payload("dinov2"),
            _bbox_metrics_payload("clip"),
        ]

        resp = client.get(
            "/api/compare/models",
            params={
                "image_id": IMAGE_ID,
                "models": ["dinov2", "clip"],
                "layer": 0,
                "percentile": 90,
                "method": "cls",
                "bbox_index": 0,
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["selection"] == {
            "mode": "bbox",
            "bbox_index": 0,
            "bbox_label": "Window",
        }
        assert {result["model"] for result in payload["results"]} == {"dinov2", "clip"}
        assert payload["unavailable_models"] == {}

    def test_bbox_compare_invalid_index_returns_400(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.db_exists = False
        mock_met_cmp.get_bbox_label.side_effect = ValueError("bbox_index 3 out of range")

        resp = client.get(
            "/api/compare/models",
            params={
                "image_id": IMAGE_ID,
                "models": ["dinov2", "clip"],
                "layer": 0,
                "percentile": 90,
                "method": "cls",
                "bbox_index": 3,
            },
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "bbox_index 3 out of range"

    def test_bbox_compare_reports_unavailable_models_without_failing(self, _mock_services):
        mock_met_cmp = _mock_services["comparison_metrics_service"]
        mock_met_cmp.db_exists = False
        mock_met_cmp.get_bbox_label.return_value = "Window"
        mock_met_cmp.get_bbox_metrics.side_effect = [
            _bbox_metrics_payload("dinov2"),
            None,
        ]

        resp = client.get(
            "/api/compare/models",
            params={
                "image_id": IMAGE_ID,
                "models": ["dinov2", "clip"],
                "layer": 0,
                "percentile": 90,
                "method": "cls",
                "bbox_index": 0,
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert [result["model"] for result in payload["results"]] == ["dinov2"]
        assert payload["selection"]["mode"] == "bbox"
        assert "clip" in payload["unavailable_models"]
        assert "Feature-level metrics unavailable" in payload["unavailable_models"]["clip"]

    def test_strategy_specific_variant_is_used(self, _mock_services):
        """Strategy query should target strategy-specific fine-tuned key."""
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned_lora"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/frozen_vs_finetuned",
            params={"image_id": IMAGE_ID, "model": "dinov2", "layer": 0, "strategy": "lora"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["finetuned"]["available"] is True
        assert "model=dinov2_finetuned_lora" in payload["finetuned"]["url"]

    def test_strategy_specific_full_query_falls_back_to_legacy_overlay(self, _mock_services):
        """Explicit full-strategy requests should keep working with legacy overlay keys."""
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2", "dinov2_finetuned"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/frozen_vs_finetuned",
            params={"image_id": IMAGE_ID, "model": "dinov2", "layer": 0, "strategy": "full"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["finetuned"]["available"] is True
        assert "model=dinov2_finetuned" in payload["finetuned"]["url"]


class TestFinetunedVsFinetunedEndpoint:
    """finetuned_vs_finetuned should compare two explicit strategy variants."""

    def test_variants_endpoint_returns_requested_strategy_pair(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2_finetuned_linear_probe", "dinov2_finetuned_full"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/variants",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "left_variant": "linear_probe",
                "right_variant": "full",
            },
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["left"]["strategy"] == "linear_probe"
        assert payload["right"]["strategy"] == "full"
        assert "model=dinov2_finetuned_linear_probe" in payload["left"]["url"]
        assert "model=dinov2_finetuned_full" in payload["right"]["url"]

    def test_returns_urls_for_both_requested_strategies(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2_finetuned_linear_probe", "dinov2_finetuned_full"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/finetuned_vs_finetuned",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "strategy_a": "linear_probe",
                "strategy_b": "full",
            },
        )
        assert resp.status_code == 200

        payload = resp.json()
        assert payload["left"]["available"] is True
        assert payload["right"]["available"] is True
        assert payload["left"]["strategy"] == "linear_probe"
        assert payload["right"]["strategy"] == "full"
        assert "model=dinov2_finetuned_linear_probe" in payload["left"]["url"]
        assert "model=dinov2_finetuned_full" in payload["right"]["url"]

    def test_full_strategy_falls_back_to_legacy_overlay_key(self, _mock_services):
        mock_img_cmp = _mock_services["comparison_image_service"]

        def _exists(model: str, _layer: str, _image_id: str, method: str, variant: str) -> bool:
            if variant not in ("overlay", "overlay_bbox") or method != "cls":
                return False
            return model in {"dinov2_finetuned_linear_probe", "dinov2_finetuned"}

        mock_img_cmp.heatmap_exists.side_effect = _exists

        resp = client.get(
            "/api/compare/finetuned_vs_finetuned",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "strategy_a": "linear_probe",
                "strategy_b": "full",
            },
        )
        assert resp.status_code == 200

        payload = resp.json()
        assert payload["left"]["available"] is True
        assert payload["right"]["strategy"] == "full"
        assert payload["right"]["available"] is True
        assert "model=dinov2_finetuned" in payload["right"]["url"]

    def test_invalid_strategy_returns_400(self):
        resp = client.get(
            "/api/compare/finetuned_vs_finetuned",
            params={
                "image_id": IMAGE_ID,
                "model": "dinov2",
                "layer": 0,
                "strategy_a": "bogus",
                "strategy_b": "full",
            },
        )
        assert resp.status_code == 400
        assert "Invalid fine-tuning strategy" in resp.json()["detail"]
