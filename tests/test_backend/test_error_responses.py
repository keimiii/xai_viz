"""Tests for error-detail exposure gated behind DEBUG mode.

Verifies that:
- With DEBUG=True, error responses include internal exception messages.
- With DEBUG=False, error responses use generic safe messages.
- The DEBUG config defaults to False when SSL_DEBUG is unset.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

IMAGE_ID = "Q1234_test"


def _make_client():
    """Create a fresh TestClient, forcing module reimport to pick up DEBUG changes."""
    # Import app fresh — config.DEBUG is read at import time by routers
    from app.backend.main import app

    return TestClient(app)


def _make_request(path: str = "/boom") -> Request:
    """Create a minimal ASGI request for exception-handler unit tests."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "root_path": "",
            "http_version": "1.1",
        }
    )


def _reload_backend_config():
    """Reload backend config so DEBUG re-reads SSL_DEBUG from the environment."""
    import app.backend.config as config_module

    return importlib.reload(config_module)


@pytest.fixture()
def _mock_attention_services():
    """Mock services used by the attention router's raw endpoint."""
    with (
        patch("app.backend.routers.attention.attention_service") as mock_attn,
        patch("app.backend.routers.attention.image_service"),
        patch("app.backend.routers.attention.similarity_service"),
    ):
        # exists() must return True so the handler reaches get_raw_attention
        mock_attn.exists.return_value = True
        yield mock_attn


@pytest.fixture()
def _mock_similarity_services():
    """Mock services used by the similarity endpoint."""
    with (
        patch("app.backend.routers.attention.similarity_service") as mock_sim,
        patch("app.backend.routers.attention.attention_service"),
        patch("app.backend.routers.attention.image_service"),
    ):
        mock_sim.features_exist.return_value = True
        yield mock_sim


class TestDebugModeErrorExposure:
    """Error payloads should only include internal details when DEBUG is on."""

    def test_valueerror_debug_on_exposes_detail(self, _mock_attention_services):
        """With DEBUG=True, ValueError detail includes the exception message."""
        _mock_attention_services.get_raw_attention.side_effect = ValueError(
            "Cache miss for dinov2/layer5/cls/Q1234"
        )

        with patch("app.backend.routers.attention.DEBUG", True):
            client = _make_client()
            resp = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 5, "method": "cls"},
            )

        assert resp.status_code == 404
        assert "Cache miss for dinov2/layer5/cls/Q1234" in resp.json()["detail"]

    def test_valueerror_debug_off_hides_detail(self, _mock_attention_services):
        """With DEBUG=False, ValueError detail is generic."""
        _mock_attention_services.get_raw_attention.side_effect = ValueError(
            "Cache miss for dinov2/layer5/cls/Q1234"
        )

        with patch("app.backend.routers.attention.DEBUG", False):
            client = _make_client()
            resp = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 5, "method": "cls"},
            )

        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail == "Requested resource not found"
        assert "Cache miss" not in detail

    def test_runtime_error_debug_on_exposes_detail(self, _mock_attention_services):
        """With DEBUG=True, unhandled exceptions include the message."""
        _mock_attention_services.get_raw_attention.side_effect = RuntimeError(
            "unexpected internal error"
        )

        with patch("app.backend.routers.attention.DEBUG", True):
            client = _make_client()
            resp = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 5, "method": "cls"},
            )

        assert resp.status_code == 500
        assert "unexpected internal error" in resp.json()["detail"]

    def test_runtime_error_debug_off_hides_detail(self, _mock_attention_services):
        """With DEBUG=False, unhandled exceptions use a generic message."""
        _mock_attention_services.get_raw_attention.side_effect = RuntimeError(
            "unexpected internal error"
        )

        with patch("app.backend.routers.attention.DEBUG", False):
            client = _make_client()
            resp = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 5, "method": "cls"},
            )

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "unexpected internal error" not in detail

    def test_similarity_valueerror_debug_off_hides_detail(self, _mock_similarity_services):
        """Similarity endpoint ValueError should be generic when DEBUG is off."""
        _mock_similarity_services.compute_similarity.side_effect = ValueError(
            "Feature shape mismatch for dinov2/layer5"
        )

        with patch("app.backend.routers.attention.DEBUG", False):
            client = _make_client()
            resp = client.post(
                f"/api/attention/{IMAGE_ID}/similarity",
                params={"model": "dinov2", "layer": 5},
                json={"left": 0.1, "top": 0.1, "width": 0.5, "height": 0.5},
            )

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail == "Invalid request"
        assert "Feature shape" not in detail

    def test_similarity_valueerror_debug_on_exposes_detail(self, _mock_similarity_services):
        """Similarity endpoint ValueError should expose detail when DEBUG is on."""
        _mock_similarity_services.compute_similarity.side_effect = ValueError(
            "Feature shape mismatch for dinov2/layer5"
        )

        with patch("app.backend.routers.attention.DEBUG", True):
            client = _make_client()
            resp = client.post(
                f"/api/attention/{IMAGE_ID}/similarity",
                params={"model": "dinov2", "layer": 5},
                json={"left": 0.1, "top": 0.1, "width": 0.5, "height": 0.5},
            )

        assert resp.status_code == 400
        assert "Feature shape mismatch for dinov2/layer5" in resp.json()["detail"]


class TestGenericExceptionHandler:
    """Generic 500 handler should gate detail behind DEBUG."""

    @pytest.mark.anyio
    async def test_debug_off_hides_uncaught_exception_detail(self) -> None:
        from app.backend.main import generic_exception_handler

        with (
            patch("app.backend.main.DEBUG", False),
            patch("app.backend.main.logger.exception"),
        ):
            response = await generic_exception_handler(
                _make_request(),
                RuntimeError("sensitive internal detail"),
            )

        assert response.status_code == 500
        assert json.loads(response.body) == {"detail": "Internal server error"}

    @pytest.mark.anyio
    async def test_debug_on_exposes_uncaught_exception_detail(self) -> None:
        from app.backend.main import generic_exception_handler

        with (
            patch("app.backend.main.DEBUG", True),
            patch("app.backend.main.logger.exception"),
        ):
            response = await generic_exception_handler(
                _make_request(),
                RuntimeError("sensitive internal detail"),
            )

        assert response.status_code == 500
        assert json.loads(response.body) == {"detail": "sensitive internal detail"}


class TestDebugDefaultOff:
    """Verify the config module defaults DEBUG to False."""

    def test_debug_defaults_to_false(self, monkeypatch):
        """When SSL_DEBUG is unset, DEBUG should be False."""
        monkeypatch.delenv("SSL_DEBUG", raising=False)
        config = _reload_backend_config()
        assert config.DEBUG is False

    def test_debug_true_when_set(self, monkeypatch):
        """When SSL_DEBUG=1, DEBUG should be True."""
        monkeypatch.setenv("SSL_DEBUG", "1")
        config = _reload_backend_config()
        assert config.DEBUG is True

    def test_debug_false_when_explicitly_zero(self, monkeypatch):
        """When SSL_DEBUG=0, DEBUG should be False."""
        monkeypatch.setenv("SSL_DEBUG", "0")
        config = _reload_backend_config()
        assert config.DEBUG is False
