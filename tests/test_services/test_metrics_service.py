"""Tests for MetricsService ordering and metric-selection semantics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from app.backend import config as config_module
from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.evaluation import fine_tuning_artifacts as artifacts_module


def _write_q2_summary_payload(
    path: Path,
    *,
    experiment_id: str,
    analysis_git_commit_sha: str,
    mean_delta: float,
) -> None:
    """Write a minimal Q2 payload that exercises service-level filtering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "analysis_git_commit_sha": analysis_git_commit_sha,
                "analyzed_layer": 11,
                "timestamp": "2026-03-06T00:00:00",
                "rows": [
                    {
                        "model_name": "dinov2",
                        "strategy_id": "lora",
                        "metric": "mse",
                        "percentile": None,
                        "label": "MSE",
                        "direction": "lower",
                        "percentile_dependent": False,
                        "mean_delta": mean_delta,
                    }
                ],
                "strategy_comparisons": [],
            }
        ),
        encoding="utf-8",
    )


def _create_progression_db(conn: sqlite3.Connection, num_layers: int = 12) -> None:
    """Populate an in-memory SQLite DB with synthetic aggregate metrics."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER
        )"""
    )

    for idx in reversed(range(num_layers)):
        conn.execute(
            """INSERT INTO aggregate_metrics
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dinov2",
                f"layer{idx}",
                "cls",
                90,
                0.1 + idx * 0.01,
                0.05,
                0.1 + idx * 0.01,
                0.5,
                0.30 - idx * 0.01,
                0.02,
                0.30 - idx * 0.01,
                0.42 - idx * 0.015,
                0.03,
                0.42 - idx * 0.015,
                0.36 - idx * 0.012,
                0.02,
                0.36 - idx * 0.012,
                139,
            ),
        )
    conn.commit()


def _create_leaderboard_db(conn: sqlite3.Connection) -> None:
    """Populate rows for leaderboard ranking tests."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER
        )"""
    )

    rows = [
        ("dinov2", "layer0", "cls", 90, 0.40, 0.01, 0.40, 0.5, 0.20, 0.01, 0.20, 0.14, 0.01, 0.14, 0.11, 0.01, 0.11, 139),
        ("dinov2", "layer1", "cls", 90, 0.55, 0.01, 0.55, 0.5, 0.12, 0.01, 0.12, 0.08, 0.01, 0.08, 0.06, 0.01, 0.06, 139),
        ("clip", "layer0", "cls", 90, 0.45, 0.01, 0.45, 0.5, 0.18, 0.01, 0.18, 0.15, 0.01, 0.15, 0.13, 0.01, 0.13, 139),
        ("clip", "layer1", "cls", 90, 0.50, 0.01, 0.50, 0.5, 0.16, 0.01, 0.16, 0.10, 0.01, 0.10, 0.09, 0.01, 0.09, 139),
    ]
    conn.executemany("INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_method_filtered_leaderboard_db(conn: sqlite3.Connection) -> None:
    """Populate rows for method-filtered leaderboard tests."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER
        )"""
    )

    rows = [
        ("dinov2", "layer0", "rollout", 90, 0.38, 0.01, 0.38, 0.5, 0.22, 0.01, 0.22, 0.20, 0.01, 0.20, 0.16, 0.01, 0.16, 139),
        ("dinov2", "layer1", "rollout", 90, 0.52, 0.01, 0.52, 0.5, 0.14, 0.01, 0.14, 0.12, 0.01, 0.12, 0.09, 0.01, 0.09, 139),
        ("clip", "layer0", "rollout", 90, 0.33, 0.01, 0.33, 0.5, 0.28, 0.01, 0.28, 0.21, 0.01, 0.21, 0.18, 0.01, 0.18, 139),
        ("clip", "layer1", "rollout", 90, 0.46, 0.01, 0.46, 0.5, 0.20, 0.01, 0.20, 0.15, 0.01, 0.15, 0.11, 0.01, 0.11, 139),
        ("siglip2", "layer0", "mean", 90, 0.41, 0.01, 0.41, 0.5, 0.18, 0.01, 0.18, 0.19, 0.01, 0.19, 0.17, 0.01, 0.17, 139),
        ("resnet50", "layer0", "gradcam", 90, 0.44, 0.01, 0.44, 0.5, 0.16, 0.01, 0.16, 0.13, 0.01, 0.13, 0.10, 0.01, 0.10, 139),
    ]
    conn.executemany("INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_best_available_leaderboard_db(conn: sqlite3.Connection) -> None:
    """Populate rows where a non-default method wins for one model."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER
        )"""
    )

    rows = [
        ("dinov2", "layer0", "cls", 90, 0.42, 0.01, 0.42, 0.5, 0.22, 0.01, 0.22, 0.18, 0.01, 0.18, 0.14, 0.01, 0.14, 139),
        ("dinov2", "layer1", "cls", 90, 0.50, 0.01, 0.50, 0.5, 0.19, 0.01, 0.19, 0.15, 0.01, 0.15, 0.11, 0.01, 0.11, 139),
        ("dinov2", "layer0", "rollout", 90, 0.71, 0.01, 0.71, 0.5, 0.16, 0.01, 0.16, 0.11, 0.01, 0.11, 0.09, 0.01, 0.09, 139),
        ("dinov2", "layer1", "rollout", 90, 0.67, 0.01, 0.67, 0.5, 0.18, 0.01, 0.18, 0.12, 0.01, 0.12, 0.10, 0.01, 0.10, 139),
        ("clip", "layer0", "cls", 90, 0.46, 0.01, 0.46, 0.5, 0.21, 0.01, 0.21, 0.16, 0.01, 0.16, 0.13, 0.01, 0.13, 139),
        ("clip", "layer1", "cls", 90, 0.60, 0.01, 0.60, 0.5, 0.17, 0.01, 0.17, 0.13, 0.01, 0.13, 0.10, 0.01, 0.10, 139),
        ("clip", "layer0", "rollout", 90, 0.55, 0.01, 0.55, 0.5, 0.20, 0.01, 0.20, 0.15, 0.01, 0.15, 0.12, 0.01, 0.12, 139),
    ]
    conn.executemany("INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_best_available_tie_db(conn: sqlite3.Connection) -> None:
    """Populate rows where best-available mode should prefer the default method on ties."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER
        )"""
    )

    rows = [
        ("dinov2", "layer1", "cls", 90, 0.60, 0.01, 0.60, 0.5, 0.18, 0.01, 0.18, 0.14, 0.01, 0.14, 0.10, 0.01, 0.10, 139),
        ("dinov2", "layer0", "rollout", 90, 0.60, 0.01, 0.60, 0.5, 0.18, 0.01, 0.18, 0.14, 0.01, 0.14, 0.10, 0.01, 0.10, 139),
    ]
    conn.executemany("INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_image_progression_db(
    conn: sqlite3.Connection,
    model: str = "dinov2",
    num_layers: int = 12,
) -> None:
    """Populate per-image metrics rows for image-detail progression tests."""
    conn.execute(
        """CREATE TABLE image_metrics (
            image_id TEXT,
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            iou REAL,
            coverage REAL,
            mse REAL,
            kl REAL,
            emd REAL,
            attention_area REAL,
            annotation_area REAL
        )"""
    )

    rows = []
    for idx in reversed(range(num_layers)):
        rows.append(
            (
                "Q123_test.jpg",
                model,
                f"layer{idx}",
                "cls",
                90,
                0.15 + idx * 0.01,
                0.42 + idx * 0.005,
                0.12 - idx * 0.004,
                0.09 - idx * 0.003,
                0.08 - idx * 0.002,
                0.20,
                0.10,
            )
        )
        rows.append(
            (
                "Q123_test.jpg",
                model,
                f"layer{idx}",
                "cls",
                50,
                0.05 + idx * 0.008,
                0.42 + idx * 0.005,
                0.12 - idx * 0.004,
                0.09 - idx * 0.003,
                0.08 - idx * 0.002,
                0.20,
                0.10,
            )
        )

    conn.executemany("INSERT INTO image_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_breakdown_db(conn: sqlite3.Connection) -> None:
    """Populate style and feature breakdown tables for metric-generic tests."""
    conn.execute(
        """CREATE TABLE style_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            metric TEXT,
            direction TEXT,
            style_name TEXT,
            percentile INTEGER,
            mean_score REAL,
            num_images INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE feature_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            metric TEXT,
            direction TEXT,
            feature_label INTEGER,
            feature_name TEXT,
            percentile INTEGER,
            mean_score REAL,
            std_score REAL,
            bbox_count INTEGER
        )"""
    )

    conn.executemany(
        "INSERT INTO style_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", "coverage", "higher", "Romanesque", 90, 0.52, 12),
            ("dinov2", "layer11", "cls", "coverage", "higher", "Gothic", 90, 0.31, 9),
            ("dinov2", "layer11", "cls", "iou", "higher", "Romanesque", 80, 0.18, 12),
            ("dinov2", "layer11", "cls", "iou", "higher", "Gothic", 80, 0.24, 9),
        ],
    )
    conn.executemany(
        "INSERT INTO feature_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", "emd", "lower", 42, "window", 90, 0.18, 0.04, 10),
            ("dinov2", "layer11", "cls", "emd", "lower", 7, "door", 90, 0.06, 0.02, 6),
            ("dinov2", "layer11", "cls", "iou", "higher", 42, "window", 80, 0.33, 0.07, 10),
            ("dinov2", "layer11", "cls", "iou", "higher", 7, "door", 80, 0.21, 0.05, 6),
        ],
    )
    conn.commit()


def _create_head_q3_db(conn: sqlite3.Connection) -> None:
    """Populate per-head Q3 tables for service-level tests."""
    conn.execute(
        """CREATE TABLE head_image_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            head INTEGER,
            metric TEXT,
            direction TEXT,
            image_id TEXT,
            percentile INTEGER,
            score REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE head_summary_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            head INTEGER,
            metric TEXT,
            direction TEXT,
            percentile INTEGER,
            mean_score REAL,
            std_score REAL,
            mean_rank REAL,
            top1_count INTEGER,
            top3_count INTEGER,
            image_count INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE head_feature_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            head INTEGER,
            metric TEXT,
            direction TEXT,
            feature_label INTEGER,
            feature_name TEXT,
            percentile INTEGER,
            mean_score REAL,
            std_score REAL,
            bbox_count INTEGER
        )"""
    )
    conn.execute(
        """CREATE TABLE head_feature_image_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            head INTEGER,
            metric TEXT,
            direction TEXT,
            feature_label INTEGER,
            feature_name TEXT,
            image_id TEXT,
            percentile INTEGER,
            score REAL,
            bbox_count INTEGER,
            default_bbox_index INTEGER
        )"""
    )

    conn.executemany(
        "INSERT INTO head_image_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", 5, "iou", "higher", "img_beta.jpg", 80, 0.66),
            ("dinov2", "layer11", "cls", 5, "iou", "higher", "img_alpha.jpg", 80, 0.73),
            ("dinov2", "layer11", "cls", 5, "iou", "higher", "img_gamma.jpg", 80, 0.51),
            ("dinov2", "layer11", "cls", 1, "iou", "higher", "img_focus.jpg", 80, 0.77),
            ("dinov2", "layer11", "cls", 0, "iou", "higher", "img_focus.jpg", 80, 0.77),
            ("dinov2", "layer11", "cls", 4, "iou", "higher", "img_focus.jpg", 80, 0.53),
            ("dinov2_finetuned_lora", "layer11", "cls", 5, "iou", "higher", "img_lora.jpg", 80, 0.88),
            ("dinov2", "layer11", "cls", 1, "emd", "lower", "img_delta.jpg", 90, 0.18),
            ("dinov2", "layer11", "cls", 1, "emd", "lower", "img_epsilon.jpg", 90, 0.09),
            ("dinov2", "layer11", "cls", 6, "emd", "lower", "img_focus.jpg", 90, 0.07),
            ("dinov2", "layer11", "cls", 2, "emd", "lower", "img_focus.jpg", 90, 0.07),
            ("dinov2", "layer11", "cls", 8, "emd", "lower", "img_focus.jpg", 90, 0.14),
        ],
    )
    conn.executemany(
        "INSERT INTO head_summary_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", 5, "iou", "higher", 80, 0.52, 0.04, 1.3, 7, 10, 12),
            ("dinov2", "layer11", "cls", 2, "iou", "higher", 80, 0.41, 0.06, 2.2, 3, 9, 12),
            ("dinov2", "layer11", "cls", 1, "emd", "lower", 90, 0.08, 0.01, 1.1, 8, 11, 12),
            ("dinov2", "layer11", "cls", 7, "emd", "lower", 90, 0.14, 0.03, 2.6, 2, 5, 12),
        ],
    )
    conn.executemany(
        "INSERT INTO head_feature_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", 0, "coverage", "higher", 7, "door", 90, 0.21, 0.03, 5),
            ("dinov2", "layer11", "cls", 1, "coverage", "higher", 7, "door", 90, 0.33, 0.02, 5),
            ("dinov2", "layer11", "cls", 0, "coverage", "higher", 42, "window", 90, 0.48, 0.05, 9),
            ("dinov2", "layer11", "cls", 1, "coverage", "higher", 42, "window", 90, 0.51, 0.04, 9),
        ],
    )
    conn.executemany(
        "INSERT INTO head_feature_image_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("dinov2", "layer11", "cls", 5, "iou", "higher", 7, "Door", "img_alpha.jpg", 80, 0.84, 2, 2),
            ("dinov2", "layer11", "cls", 5, "iou", "higher", 7, "Door", "img_gamma.jpg", 80, 0.59, 1, 0),
            ("dinov2", "layer11", "cls", 5, "emd", "lower", 7, "Door", "img_epsilon.jpg", 90, 0.07, 1, 0),
            ("dinov2", "layer11", "cls", 5, "emd", "lower", 7, "Door", "img_alpha.jpg", 90, 0.09, 2, 0),
            ("dinov2_finetuned_lora", "layer11", "cls", 5, "iou", "higher", 7, "Door", "img_lora.jpg", 80, 0.88, 1, 0),
        ],
    )
    conn.commit()


class TestLayerProgression:
    """Verify get_layer_progression() ordering and best-layer logic."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def mem_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_progression_db(conn, num_layers=12)
        yield conn
        conn.close()

    def test_layers_returned_in_numeric_order(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        assert result["layers"] == [f"layer{i}" for i in range(12)]

    def test_iou_scores_match_layer_order(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        for i, score in enumerate(result["scores"]):
            assert score == pytest.approx(0.1 + i * 0.01, abs=1e-9)

    def test_best_layer_is_highest_numeric_for_iou(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.1 + 11 * 0.01, abs=1e-9)

    def test_mse_uses_lowest_score_for_best_layer(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2",
                percentile=90,
                method="cls",
                metric="mse",
            )

        assert result["metric"] == "mse"
        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.30 - 11 * 0.01, abs=1e-9)

    def test_kl_uses_lowest_score_for_best_layer(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2",
                percentile=90,
                method="cls",
                metric="kl",
            )

        assert result["metric"] == "kl"
        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.42 - 11 * 0.015, abs=1e-9)

    def test_emd_uses_lowest_score_for_best_layer(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2",
                percentile=90,
                method="cls",
                metric="emd",
            )

        assert result["metric"] == "emd"
        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.36 - 11 * 0.012, abs=1e-9)


class TestLeaderboardOrdering:
    """Verify leaderboard ordering for selectable metrics."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    def test_mse_leaderboard_sorts_ascending(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="mse")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.12, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "cls" for entry in leaderboard)
        assert all(entry["metric"] == "mse" for entry in leaderboard)
        conn.close()

    def test_kl_leaderboard_sorts_ascending(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="kl")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.08, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "cls" for entry in leaderboard)
        assert all(entry["metric"] == "kl" for entry in leaderboard)
        conn.close()

    def test_emd_leaderboard_sorts_ascending(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="emd")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.06, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "cls" for entry in leaderboard)
        assert all(entry["metric"] == "emd" for entry in leaderboard)
        conn.close()

    def test_best_available_mode_can_select_non_default_method(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_best_available_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="iou", ranking_mode="best_available")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.71, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer0"
        assert leaderboard[0]["method_used"] == "rollout"
        assert leaderboard[1]["method_used"] == "cls"
        conn.close()

    def test_best_available_mode_prefers_default_method_on_ties(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_best_available_tie_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="iou", ranking_mode="best_available")

        assert len(leaderboard) == 1
        assert leaderboard[0]["method_used"] == "cls"
        assert leaderboard[0]["best_layer"] == "layer1"
        conn.close()

    def test_rollout_leaderboard_only_includes_rollout_compatible_models(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_method_filtered_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="iou", method="rollout")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "rollout" for entry in leaderboard)
        assert all(entry["metric"] == "iou" for entry in leaderboard)
        conn.close()

    def test_gradcam_leaderboard_only_includes_resnet50(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_method_filtered_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="iou", method="gradcam")

        assert [entry["model"] for entry in leaderboard] == ["resnet50"]
        assert leaderboard[0]["best_layer"] == "layer0"
        assert leaderboard[0]["method_used"] == "gradcam"
        conn.close()


class TestLegacyContinuousMetricFallback:
    """Verify continuous-metric fallbacks work for DBs that predate new columns."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    def test_get_image_metrics_recomputes_missing_continuous_metrics(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE image_metrics (
                image_id TEXT,
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                iou REAL,
                coverage REAL,
                mse REAL,
                attention_area REAL,
                annotation_area REAL
            )"""
        )
        conn.execute(
            """INSERT INTO image_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("Q123_test.jpg", "dinov2", "layer0", "cls", 90, 0.2, 0.4, 0.05, 0.2, 0.1),
        )
        conn.commit()

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_image_kl_from_cache", return_value=0.07),
            patch.object(type(service), "_compute_image_emd_from_cache", return_value=0.11),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_metrics(
                image_id="Q123_test.jpg",
                model="dinov2",
                layer="layer0",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert result["kl"] == pytest.approx(0.07, abs=1e-9)
        assert result["emd"] == pytest.approx(0.11, abs=1e-9)
        conn.close()

    def test_get_aggregate_metrics_recomputes_missing_continuous_metrics(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE aggregate_metrics (
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                mean_iou REAL,
                std_iou REAL,
                median_iou REAL,
                mean_coverage REAL,
                mean_mse REAL,
                std_mse REAL,
                median_mse REAL,
                num_images INTEGER
            )"""
        )
        conn.execute(
            """INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("dinov2", "layer0", "cls", 90, 0.3, 0.02, 0.3, 0.5, 0.07, 0.01, 0.07, 139),
        )
        conn.commit()

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_kl_aggregate_from_images", return_value=(0.09, 0.02, 0.08)),
            patch.object(type(service), "_compute_emd_aggregate_from_images", return_value=(0.12, 0.03, 0.11)),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_aggregate_metrics(
                model="dinov2",
                layer="layer0",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert result["mean_kl"] == pytest.approx(0.09, abs=1e-9)
        assert result["std_kl"] == pytest.approx(0.02, abs=1e-9)
        assert result["median_kl"] == pytest.approx(0.08, abs=1e-9)
        assert result["mean_emd"] == pytest.approx(0.12, abs=1e-9)
        assert result["std_emd"] == pytest.approx(0.03, abs=1e-9)
        assert result["median_emd"] == pytest.approx(0.11, abs=1e-9)
        conn.close()

    def test_get_leaderboard_recomputes_kl_when_columns_missing(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE aggregate_metrics (
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                mean_iou REAL,
                std_iou REAL,
                median_iou REAL,
                mean_coverage REAL,
                mean_mse REAL,
                std_mse REAL,
                median_mse REAL,
                num_images INTEGER
            )"""
        )
        conn.executemany(
            """INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("dinov2", "layer0", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("dinov2", "layer1", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer0", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer1", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
            ],
        )
        conn.commit()

        def kl_side_effect(*, model: str, layer: str, percentile: int, method: str):
            values = {
                ("dinov2", "layer0"): (0.30, 0.01, 0.30),
                ("dinov2", "layer1"): (0.18, 0.01, 0.18),
                ("clip", "layer0"): (0.26, 0.01, 0.26),
                ("clip", "layer1"): (0.22, 0.01, 0.22),
            }
            return values[(model, layer)]

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_kl_aggregate_from_images", side_effect=kl_side_effect),
            patch.object(type(service), "_compute_emd_aggregate_from_images", return_value=(0.31, 0.01, 0.31)),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="kl")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.18, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "cls" for entry in leaderboard)
        conn.close()

    def test_get_leaderboard_recomputes_kl_when_columns_missing_for_shared_method(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE aggregate_metrics (
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                mean_iou REAL,
                std_iou REAL,
                median_iou REAL,
                mean_coverage REAL,
                mean_mse REAL,
                std_mse REAL,
                median_mse REAL,
                num_images INTEGER
            )"""
        )
        conn.executemany(
            """INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("dinov2", "layer0", "rollout", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("dinov2", "layer1", "rollout", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer0", "rollout", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer1", "rollout", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("siglip2", "layer0", "mean", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
            ],
        )
        conn.commit()

        def kl_side_effect(*, model: str, layer: str, percentile: int, method: str):
            values = {
                ("dinov2", "layer0", "rollout"): (0.27, 0.01, 0.27),
                ("dinov2", "layer1", "rollout"): (0.18, 0.01, 0.18),
                ("clip", "layer0", "rollout"): (0.23, 0.01, 0.23),
                ("clip", "layer1", "rollout"): (0.21, 0.01, 0.21),
            }
            return values[(model, layer, method)]

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_kl_aggregate_from_images", side_effect=kl_side_effect),
            patch.object(type(service), "_compute_emd_aggregate_from_images", return_value=(0.31, 0.01, 0.31)),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="kl", method="rollout")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.18, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "rollout" for entry in leaderboard)
        conn.close()

    def test_get_leaderboard_recomputes_emd_when_columns_missing(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE aggregate_metrics (
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                mean_iou REAL,
                std_iou REAL,
                median_iou REAL,
                mean_coverage REAL,
                mean_mse REAL,
                std_mse REAL,
                median_mse REAL,
                num_images INTEGER
            )"""
        )
        conn.executemany(
            """INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("dinov2", "layer0", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("dinov2", "layer1", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer0", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("clip", "layer1", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
            ],
        )
        conn.commit()

        def emd_side_effect(*, model: str, layer: str, percentile: int, method: str):
            values = {
                ("dinov2", "layer0"): (0.22, 0.01, 0.22),
                ("dinov2", "layer1"): (0.14, 0.01, 0.14),
                ("clip", "layer0"): (0.30, 0.01, 0.30),
                ("clip", "layer1"): (0.18, 0.01, 0.18),
            }
            return values[(model, layer)]

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_kl_aggregate_from_images", return_value=(0.4, 0.01, 0.4)),
            patch.object(type(service), "_compute_emd_aggregate_from_images", side_effect=emd_side_effect),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="emd")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.14, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["method_used"] == "cls" for entry in leaderboard)
        conn.close()

    def test_get_layer_progression_recomputes_kl_when_columns_missing(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE aggregate_metrics (
                model TEXT,
                layer TEXT,
                method TEXT,
                percentile INTEGER,
                mean_iou REAL,
                std_iou REAL,
                median_iou REAL,
                mean_coverage REAL,
                mean_mse REAL,
                std_mse REAL,
                median_mse REAL,
                num_images INTEGER
            )"""
        )
        conn.executemany(
            """INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("dinov2", "layer0", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
                ("dinov2", "layer1", "cls", 90, 0.2, 0.01, 0.2, 0.4, 0.10, 0.01, 0.10, 139),
            ],
        )
        conn.commit()

        def kl_side_effect(*, model: str, layer: str, percentile: int, method: str):
            values = {
                ("dinov2", "layer0"): (0.24, 0.01, 0.24),
                ("dinov2", "layer1"): (0.16, 0.01, 0.16),
            }
            return values[(model, layer)]

        with (
            patch.object(type(service), "get_connection") as mock_ctx,
            patch.object(type(service), "_compute_kl_aggregate_from_images", side_effect=kl_side_effect),
            patch.object(type(service), "_compute_emd_aggregate_from_images", return_value=(0.25, 0.01, 0.25)),
        ):
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2",
                percentile=90,
                method="cls",
                metric="kl",
            )

        assert result["layers"] == ["layer0", "layer1"]
        assert result["scores"] == pytest.approx([0.24, 0.16], abs=1e-9)
        assert result["best_layer"] == "layer1"
        assert result["best_score"] == pytest.approx(0.16, abs=1e-9)
        conn.close()


class TestImageDetailLayerProgression:
    """Verify the extensible image-detail progression helpers."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def image_progression_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_image_progression_db(conn, model="dinov2", num_layers=12)
        yield conn
        conn.close()

    def test_union_progression_layers_are_numeric_and_include_descriptors(self, service, image_progression_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: image_progression_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_layer_progression(
                image_id="Q123_test.jpg",
                model="dinov2",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert result["selection"]["mode"] == "union"
        assert [metric["key"] for metric in result["metrics"]] == ["iou", "coverage", "mse", "kl", "emd"]
        assert [metric["direction"] for metric in result["metrics"]] == ["higher", "higher", "lower", "lower", "lower"]
        assert [metric["percentile_dependent"] for metric in result["metrics"]] == [True, False, False, False, False]
        assert [point["layer_key"] for point in result["layers"]] == [f"layer{i}" for i in range(12)]

    def test_union_progression_respects_model_specific_layer_count(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_image_progression_db(conn, model="resnet50", num_layers=4)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_layer_progression(
                image_id="Q123_test.jpg",
                model="resnet50",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert [point["layer"] for point in result["layers"]] == [0, 1, 2, 3]
        conn.close()

    def test_percentile_changes_only_change_thresholded_metric_values(self, service, image_progression_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: image_progression_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            p90 = service.get_image_layer_progression("Q123_test.jpg", "dinov2", percentile=90, method="cls")
            p50 = service.get_image_layer_progression("Q123_test.jpg", "dinov2", percentile=50, method="cls")

        assert p90 is not None and p50 is not None
        assert p90["layers"][0]["values"]["iou"] != p50["layers"][0]["values"]["iou"]
        assert p90["layers"][0]["values"]["coverage"] == pytest.approx(p50["layers"][0]["values"]["coverage"], abs=1e-9)
        assert p90["layers"][0]["values"]["mse"] == pytest.approx(p50["layers"][0]["values"]["mse"], abs=1e-9)
        assert p90["layers"][0]["values"]["kl"] == pytest.approx(p50["layers"][0]["values"]["kl"], abs=1e-9)
        assert p90["layers"][0]["values"]["emd"] == pytest.approx(p50["layers"][0]["values"]["emd"], abs=1e-9)

    def test_bbox_progression_returns_metric_values_for_each_layer(self, service):
        annotation = ImageAnnotation(
            image_id="Q123_test.jpg",
            styles=(),
            bboxes=(
                BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=7, group_label=7),
            ),
        )
        attention = torch.tensor(
            [
                [0.05, 0.15, 0.20, 0.10],
                [0.10, 0.70, 0.80, 0.15],
                [0.05, 0.75, 0.90, 0.10],
                [0.02, 0.10, 0.12, 0.03],
            ],
            dtype=torch.float32,
        )

        with (
            patch.object(type(service), "_get_annotation", return_value=annotation),
            patch.object(type(service), "_get_bbox_label", return_value="Window"),
            patch.object(type(service), "_load_attention_tensor", return_value=attention),
        ):
            result = service.get_bbox_layer_progression(
                image_id="Q123_test.jpg",
                model="resnet50",
                bbox_index=0,
                percentile=90,
                method="gradcam",
            )

        assert result is not None
        assert result["selection"] == {"mode": "bbox", "bbox_index": 0, "bbox_label": "Window"}
        assert [point["layer"] for point in result["layers"]] == [0, 1, 2, 3]
        assert all(point["values"]["mse"] is not None for point in result["layers"])
        assert all(point["values"]["kl"] is not None for point in result["layers"])
        assert all(point["values"]["emd"] is not None for point in result["layers"])

    def test_bbox_progression_rejects_out_of_range_index(self, service):
        annotation = ImageAnnotation(
            image_id="Q123_test.jpg",
            styles=(),
            bboxes=(
                BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=7, group_label=7),
            ),
        )

        with (
            patch.object(type(service), "_get_annotation", return_value=annotation),
            pytest.raises(ValueError, match="bbox_index 3 out of range"),
        ):
            service.get_bbox_layer_progression(
                image_id="Q123_test.jpg",
                model="dinov2",
                bbox_index=3,
                percentile=90,
                method="cls",
            )


class TestSummaryMetadata:
    """Verify static summary metadata is explicit even for legacy cache files."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    def test_get_summary_backfills_default_method_metadata(self, service, tmp_path):
        summary_path = tmp_path / "metrics_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "models": {
                        "dinov2": {
                            "best_layer": "layer11",
                            "best_iou": 0.81,
                            "layer_progression": {"layer11": 0.81},
                        }
                    },
                    "leaderboard": [{"model": "dinov2", "best_iou": 0.81}],
                    "leaderboards": {
                        "iou": [{"model": "dinov2", "best_layer": "layer11", "best_score": 0.81}],
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch("app.backend.services.metrics_service.METRICS_SUMMARY_PATH", summary_path):
            result = service.get_summary()

        assert result is not None
        assert result["ranking_mode"] == "default_method"
        assert result["models"]["dinov2"]["method_used"] == "cls"
        assert result["leaderboard"][0]["method_used"] == "cls"
        assert result["leaderboards"]["iou"][0]["method_used"] == "cls"


class TestMetricGenericBreakdowns:
    """Verify style and feature breakdowns follow metric semantics."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def mem_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_breakdown_db(conn)
        yield conn
        conn.close()

    def test_style_breakdown_uses_metric_specific_direction_and_reference_percentile(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_style_breakdown(
                model="dinov2",
                layer="layer11",
                percentile=50,
                metric="coverage",
                method="cls",
            )

        assert result["metric"] == "coverage"
        assert result["direction"] == "higher"
        assert result["percentile"] == 50
        assert list(result["styles"]) == ["Romanesque", "Gothic"]
        assert result["styles"]["Romanesque"] == pytest.approx(0.52)
        assert result["style_counts"]["Gothic"] == 9

    def test_feature_breakdown_sorts_lower_is_better_metrics_ascending(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_feature_breakdown(
                model="dinov2",
                layer="layer11",
                percentile=50,
                metric="emd",
                method="cls",
            )

        assert result["metric"] == "emd"
        assert result["direction"] == "lower"
        assert result["percentile"] == 50
        assert [feature["feature_name"] for feature in result["features"]] == ["door", "window"]
        assert result["features"][0]["mean_score"] == pytest.approx(0.06)


class TestQ3HeadMetrics:
    """Verify Q3 service queries respect metric semantics and unsupported models."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def mem_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_head_q3_db(conn)
        yield conn
        conn.close()

    def test_get_head_ranking_uses_metric_specific_percentile_and_ordering(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            iou_result = service.get_head_ranking(
                model="dinov2",
                layer="layer11",
                percentile=80,
                metric="iou",
                variant="frozen",
            )
            emd_result = service.get_head_ranking(
                model="dinov2",
                layer="layer11",
                percentile=50,
                metric="emd",
                variant="frozen",
            )

        assert iou_result["supported"] is True
        assert iou_result["method"] == "cls"
        assert [entry["head"] for entry in iou_result["heads"]] == [5, 2]
        assert iou_result["heads"][0]["mean_rank"] == pytest.approx(1.3)

        assert emd_result["supported"] is True
        assert emd_result["percentile"] == 50
        assert [entry["head"] for entry in emd_result["heads"]] == [1, 7]
        assert emd_result["heads"][0]["mean_score"] == pytest.approx(0.08)

    def test_get_head_feature_matrix_builds_dense_head_columns(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_head_feature_matrix(
                model="dinov2",
                layer="layer11",
                percentile=50,
                metric="coverage",
                variant="frozen",
            )

        assert result["supported"] is True
        assert result["percentile"] == 50
        assert result["heads"][:2] == [0, 1]
        assert result["total_feature_types"] == 2
        assert result["features"][0]["feature_name"] == "door"
        assert result["features"][0]["scores"][0] == pytest.approx(0.21)
        assert result["features"][0]["scores"][1] == pytest.approx(0.33)
        assert result["features"][0]["scores"][2] is None

    def test_get_image_head_ranking_uses_image_rows_and_tie_breaks_by_head(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_head_ranking(
                image_id="img_focus.jpg",
                model="dinov2",
                layer="layer11",
                percentile=80,
                metric="iou",
                variant="frozen",
            )

        assert result["supported"] is True
        assert result["selection"]["mode"] == "union"
        assert [entry["head"] for entry in result["heads"]] == [0, 1, 4]
        assert result["heads"][0]["score"] == pytest.approx(0.77)

    def test_get_image_head_ranking_sorts_lower_is_better_metrics_ascending(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_head_ranking(
                image_id="img_focus.jpg",
                model="dinov2",
                layer="layer11",
                percentile=50,
                metric="emd",
                variant="frozen",
            )

        assert result["supported"] is True
        assert result["percentile"] == 50
        assert [entry["head"] for entry in result["heads"]] == [2, 6, 8]
        assert result["heads"][0]["score"] == pytest.approx(0.07)

    def test_get_image_head_ranking_computes_bbox_local_scores_for_each_head(self, service, mem_db):
        annotation = ImageAnnotation(
            image_id="img_focus.jpg",
            styles=("Q46261",),
            bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),),
        )
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch.object(type(service), "_get_annotation", return_value=annotation), patch.object(
                type(service), "_get_bbox_label", return_value="Door"
            ), patch.object(
                type(service),
                "_compute_bbox_metrics",
                side_effect=[
                    {"iou": 0.31, "coverage": 0.52, "mse": 0.09, "kl": 0.18, "emd": 0.21},
                    {"iou": 0.68, "coverage": 0.73, "mse": 0.05, "kl": 0.12, "emd": 0.14},
                    None,
                    {"iou": 0.68, "coverage": 0.61, "mse": 0.04, "kl": 0.11, "emd": 0.10},
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ):
                result = service.get_image_head_ranking(
                    image_id="img_focus.jpg",
                    model="dinov2",
                    layer="layer11",
                    percentile=90,
                    metric="iou",
                    variant="frozen",
                    bbox_index=0,
                )

        assert result["supported"] is True
        assert result["selection"] == {
            "mode": "bbox",
            "bbox_index": 0,
            "bbox_label": "Door",
        }
        assert [entry["head"] for entry in result["heads"]] == [1, 3, 0]
        assert result["heads"][0]["score"] == pytest.approx(0.68)

    def test_get_image_head_ranking_returns_unsupported_payload_when_bbox_cache_is_missing(self, service, mem_db):
        annotation = ImageAnnotation(
            image_id="img_focus.jpg",
            styles=("Q46261",),
            bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),),
        )
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch.object(type(service), "_get_annotation", return_value=annotation), patch.object(
                type(service), "_get_bbox_label", return_value="Door"
            ), patch.object(type(service), "_compute_bbox_metrics", return_value=None):
                result = service.get_image_head_ranking(
                    image_id="img_focus.jpg",
                    model="dinov2",
                    layer="layer11",
                    percentile=90,
                    metric="coverage",
                    variant="frozen",
                    bbox_index=0,
                )

        assert result["supported"] is False
        assert result["selection"]["mode"] == "bbox"
        assert "generate_attention_cache.py --per-head" in str(result["reason"])

    def test_get_head_ranking_returns_unsupported_payload_for_resnet(self, service):
        result = service.get_head_ranking(
            model="resnet50",
            layer="layer0",
            percentile=90,
            metric="iou",
            variant="frozen",
        )

        assert result["supported"] is False
        assert "not supported" in str(result["reason"]).lower()

    def test_get_head_exemplars_respects_metric_direction_and_limit(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch("app.backend.services.metrics_service.image_service") as mock_image_service:
                mock_image_service.get_feature_name.return_value = None
                mock_image_service.get_annotation.side_effect = lambda image_id: ImageAnnotation(
                    image_id=image_id,
                    styles=("Q46261",),
                    bboxes=(),
                )
                mock_image_service.get_style_names.return_value = ["Gothic"]

                result = service.get_head_exemplars(
                    model="dinov2",
                    layer="layer11",
                    head=5,
                    percentile=80,
                    metric="iou",
                    variant="frozen",
                    limit=2,
                )

        assert result["supported"] is True
        assert [candidate["image_id"] for candidate in result["candidates"]] == [
            "img_alpha.jpg",
            "img_beta.jpg",
        ]
        assert result["candidates"][0]["score"] == pytest.approx(0.73)

    def test_get_head_exemplars_filters_to_matching_feature_bboxes(self, service, mem_db):
        annotations = {
            "img_alpha.jpg": ImageAnnotation(
                image_id="img_alpha.jpg",
                styles=("Q46261",),
                bboxes=(
                    BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),
                    BoundingBox(0.2, 0.2, 0.2, 0.2, 42, 42),
                    BoundingBox(0.3, 0.3, 0.2, 0.2, 7, 7),
                ),
            ),
            "img_beta.jpg": ImageAnnotation(
                image_id="img_beta.jpg",
                styles=("Q46261",),
                bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 42, 42),),
            ),
            "img_gamma.jpg": ImageAnnotation(
                image_id="img_gamma.jpg",
                styles=("Q12554",),
                bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),),
            ),
        }
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch("app.backend.services.metrics_service.image_service") as mock_image_service:
                mock_image_service.get_feature_name.return_value = "Door"
                mock_image_service.get_annotation.side_effect = lambda image_id: annotations[image_id]
                mock_image_service.get_style_names.side_effect = lambda styles: ["Gothic" if "Q46261" in styles else "Romanesque"]

                result = service.get_head_exemplars(
                    model="dinov2",
                    layer="layer11",
                    head=5,
                    percentile=80,
                    metric="iou",
                    variant="frozen",
                    feature_label=7,
                )

        assert result["feature_name"] == "Door"
        assert [candidate["image_id"] for candidate in result["candidates"]] == [
            "img_alpha.jpg",
            "img_gamma.jpg",
        ]
        assert result["candidates"][0]["matching_bbox_indices"] == [0, 2]
        assert result["candidates"][0]["default_bbox_index"] == 2

    def test_get_head_exemplars_uses_feature_image_rows_for_lower_is_better_metrics(self, service, mem_db):
        annotations = {
            "img_alpha.jpg": ImageAnnotation(
                image_id="img_alpha.jpg",
                styles=("Q46261",),
                bboxes=(
                    BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),
                    BoundingBox(0.2, 0.2, 0.2, 0.2, 42, 42),
                    BoundingBox(0.3, 0.3, 0.2, 0.2, 7, 7),
                ),
            ),
            "img_epsilon.jpg": ImageAnnotation(
                image_id="img_epsilon.jpg",
                styles=("Q12554",),
                bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),),
            ),
        }
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch("app.backend.services.metrics_service.image_service") as mock_image_service:
                mock_image_service.get_feature_name.return_value = "Door"
                mock_image_service.get_annotation.side_effect = lambda image_id: annotations[image_id]
                mock_image_service.get_style_names.side_effect = lambda styles: ["Gothic" if "Q46261" in styles else "Romanesque"]

                result = service.get_head_exemplars(
                    model="dinov2",
                    layer="layer11",
                    head=5,
                    percentile=50,
                    metric="emd",
                    variant="frozen",
                    feature_label=7,
                )

        assert result["supported"] is True
        assert [candidate["image_id"] for candidate in result["candidates"]] == [
            "img_epsilon.jpg",
            "img_alpha.jpg",
        ]
        assert result["candidates"][0]["default_bbox_index"] == 0

    def test_get_head_exemplars_returns_unsupported_payload_when_feature_image_cache_is_missing(self, service, mem_db):
        mem_db.execute("DROP TABLE head_feature_image_metrics")
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch("app.backend.services.metrics_service.image_service") as mock_image_service:
                mock_image_service.get_feature_name.return_value = "Door"

                result = service.get_head_exemplars(
                    model="dinov2",
                    layer="layer11",
                    head=5,
                    percentile=80,
                    metric="iou",
                    variant="frozen",
                    feature_label=7,
                )

        assert result["supported"] is False
        assert "head-feature exemplars" in str(result["reason"]).lower()

    def test_get_head_exemplars_uses_variant_specific_storage_key(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None
            with patch("app.backend.services.metrics_service.image_service") as mock_image_service:
                mock_image_service.get_feature_name.return_value = "Door"
                mock_image_service.get_annotation.side_effect = lambda image_id: ImageAnnotation(
                    image_id=image_id,
                    styles=("Q46261",),
                    bboxes=(BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),),
                )
                mock_image_service.get_style_names.return_value = ["Gothic"]

                result = service.get_head_exemplars(
                    model="dinov2",
                    layer="layer11",
                    head=5,
                    percentile=80,
                    metric="iou",
                    variant="lora",
                    feature_label=7,
                )

        assert [candidate["image_id"] for candidate in result["candidates"]] == ["img_lora.jpg"]


class TestQ2SummaryFiltering:
    """Verify Q2 summary filtering stays consistent across payload sections."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    def _configure_active_q2_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> Path:
        """Point backend Q2 path resolution at a temporary results tree."""
        results_root = tmp_path / "outputs" / "results"
        monkeypatch.setattr(artifacts_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(artifacts_module, "RESULTS_ROOT", results_root)
        monkeypatch.setattr(artifacts_module, "EXPERIMENTS_ROOT", results_root / "experiments")
        monkeypatch.setattr(artifacts_module, "ACTIVE_EXPERIMENT_PATH", results_root / "active_experiment.json")
        legacy_path = results_root / "q2_metrics_analysis.json"
        monkeypatch.setattr(config_module, "LEGACY_Q2_RESULTS_PATH", legacy_path)
        return legacy_path

    def test_get_q2_summary_filters_strategy_rows_and_pairwise_comparisons(self, service, tmp_path):
        q2_path = tmp_path / "q2_metrics_analysis.json"
        q2_path.write_text(
            json.dumps(
                {
                    "analysis_git_commit_sha": "analysissha",
                    "analyzed_layer": 11,
                    "timestamp": "2026-03-06T00:00:00",
                    "rows": [
                        {"model_name": "dinov2", "strategy_id": "linear_probe", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.01},
                        {"model_name": "dinov2", "strategy_id": "lora", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.05},
                        {"model_name": "dinov2", "strategy_id": "lora", "metric": "iou", "percentile": 80, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.04},
                        {"model_name": "dinov2", "strategy_id": "full", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.03},
                        {"model_name": "clip", "strategy_id": "full", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.07},
                    ],
                    "strategy_comparisons": [
                        {"model_name": "dinov2", "metric": "iou", "percentile": 90, "strategy_a": "linear_probe", "strategy_b": "lora"},
                        {"model_name": "dinov2", "metric": "iou", "percentile": 90, "strategy_a": "linear_probe", "strategy_b": "full"},
                        {"model_name": "dinov2", "metric": "iou", "percentile": 90, "strategy_a": "full", "strategy_b": "lora"},
                        {"model_name": "dinov2", "metric": "iou", "percentile": 80, "strategy_a": "linear_probe", "strategy_b": "lora"},
                        {"model_name": "clip", "metric": "iou", "percentile": 90, "strategy_a": "linear_probe", "strategy_b": "full"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch("app.backend.services.metrics_service.get_current_q2_results_path", return_value=q2_path):
            result = service.get_q2_summary(metric="iou", strategy="lora")

        assert result is not None
        assert result["analysis_git_commit_sha"] == "analysissha"
        assert {row["strategy_id"] for row in result["rows"]} == {"lora"}
        assert {row["model_name"] for row in result["rows"]} == {"dinov2"}
        assert all(
            row["strategy_a"] == "lora" or row["strategy_b"] == "lora"
            for row in result["strategy_comparisons"]
        )
        assert all(row["model_name"] == "dinov2" for row in result["strategy_comparisons"])

    def test_get_q2_summary_composes_model_strategy_and_percentile_filters(self, service, tmp_path):
        q2_path = tmp_path / "q2_metrics_analysis.json"
        q2_path.write_text(
            json.dumps(
                {
                    "analysis_git_commit_sha": "analysissha",
                    "analyzed_layer": 11,
                    "timestamp": "2026-03-06T00:00:00",
                    "rows": [
                        {"model_name": "dinov2", "strategy_id": "linear_probe", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.01},
                        {"model_name": "dinov2", "strategy_id": "lora", "metric": "iou", "percentile": 90, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.05},
                        {"model_name": "dinov2", "strategy_id": "lora", "metric": "iou", "percentile": 80, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.04},
                        {"model_name": "clip", "strategy_id": "lora", "metric": "iou", "percentile": 80, "label": "IoU", "direction": "higher", "percentile_dependent": True, "mean_delta": 0.08},
                    ],
                    "strategy_comparisons": [
                        {"model_name": "dinov2", "metric": "iou", "percentile": 90, "strategy_a": "linear_probe", "strategy_b": "lora"},
                        {"model_name": "dinov2", "metric": "iou", "percentile": 80, "strategy_a": "linear_probe", "strategy_b": "lora"},
                        {"model_name": "clip", "metric": "iou", "percentile": 80, "strategy_a": "linear_probe", "strategy_b": "lora"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch("app.backend.services.metrics_service.get_current_q2_results_path", return_value=q2_path):
            result = service.get_q2_summary(
                metric="iou",
                percentile=80,
                model="dinov2",
                strategy="lora",
            )

        assert result is not None
        assert result["selected_percentile"] == 80
        assert result["rows"] == [
            {
                "model_name": "dinov2",
                "strategy_id": "lora",
                "metric": "iou",
                "percentile": 80,
                "label": "IoU",
                "direction": "higher",
                "percentile_dependent": True,
                "mean_delta": 0.04,
            }
        ]
        assert result["strategy_comparisons"] == [
            {
                "model_name": "dinov2",
                "metric": "iou",
                "percentile": 80,
                "strategy_a": "linear_probe",
                "strategy_b": "lora",
            }
        ]

    def test_get_q2_summary_ignores_percentile_for_threshold_free_metrics(self, service, tmp_path):
        q2_path = tmp_path / "q2_metrics_analysis.json"
        q2_path.write_text(
            json.dumps(
                {
                    "analysis_git_commit_sha": "analysissha",
                    "analyzed_layer": 11,
                    "timestamp": "2026-03-06T00:00:00",
                    "rows": [
                        {"model_name": "dinov2", "strategy_id": "lora", "metric": "mse", "percentile": None, "label": "MSE", "direction": "lower", "percentile_dependent": False, "mean_delta": -0.01},
                    ],
                    "strategy_comparisons": [],
                }
            ),
            encoding="utf-8",
        )

        with patch("app.backend.services.metrics_service.get_current_q2_results_path", return_value=q2_path):
            result = service.get_q2_summary(metric="mse", percentile=90)

        assert result is not None
        assert result["selected_percentile"] is None
        assert result["rows"][0]["metric"] == "mse"

    def test_get_q2_summary_normalizes_legacy_analysis_provenance(self, service, tmp_path):
        q2_path = tmp_path / "q2_metrics_analysis.json"
        q2_path.write_text(
            json.dumps(
                {
                    "git_commit_sha": "legacysha",
                    "analyzed_layer": 11,
                    "timestamp": "2026-03-06T00:00:00",
                    "rows": [
                        {
                            "model_name": "dinov2",
                            "strategy_id": "lora",
                            "metric": "mse",
                            "percentile": None,
                            "label": "MSE",
                            "direction": "lower",
                            "percentile_dependent": False,
                            "mean_delta": -0.01,
                        }
                    ],
                    "strategy_comparisons": [],
                }
            ),
            encoding="utf-8",
        )

        with patch("app.backend.services.metrics_service.get_current_q2_results_path", return_value=q2_path):
            result = service.get_q2_summary(metric="mse")

        assert result is not None
        assert result["analysis_git_commit_sha"] == "legacysha"
        assert "git_commit_sha" not in result

    def test_get_q2_summary_resolves_active_experiment_path_per_call(
        self,
        service,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._configure_active_q2_environment(monkeypatch, tmp_path)
        q2_path_a = tmp_path / "outputs" / "results" / "q2_metrics_analysis_a.json"
        q2_path_b = tmp_path / "outputs" / "results" / "q2_metrics_analysis_b.json"
        _write_q2_summary_payload(
            q2_path_a,
            experiment_id="exp_a",
            analysis_git_commit_sha="sha-a",
            mean_delta=-0.01,
        )
        _write_q2_summary_payload(
            q2_path_b,
            experiment_id="exp_b",
            analysis_git_commit_sha="sha-b",
            mean_delta=-0.02,
        )

        artifacts_module.write_active_experiment("exp_a", q2_metrics_path=q2_path_a)
        result_a = service.get_q2_summary(metric="mse")

        artifacts_module.write_active_experiment("exp_b", q2_metrics_path=q2_path_b)
        result_b = service.get_q2_summary(metric="mse")

        assert result_a is not None
        assert result_a["experiment_id"] == "exp_a"
        assert result_a["analysis_git_commit_sha"] == "sha-a"
        assert result_a["rows"][0]["mean_delta"] == pytest.approx(-0.01)

        assert result_b is not None
        assert result_b["experiment_id"] == "exp_b"
        assert result_b["analysis_git_commit_sha"] == "sha-b"
        assert result_b["rows"][0]["mean_delta"] == pytest.approx(-0.02)

    def test_get_q2_summary_falls_back_to_legacy_path_when_active_pointer_has_no_q2_artifact(
        self,
        service,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        legacy_path = self._configure_active_q2_environment(monkeypatch, tmp_path)
        _write_q2_summary_payload(
            legacy_path,
            experiment_id="legacy_exp",
            analysis_git_commit_sha="legacysha",
            mean_delta=-0.03,
        )

        artifacts_module.write_active_experiment("exp_without_q2")
        result = service.get_q2_summary(metric="mse")

        assert result is not None
        assert result["experiment_id"] == "legacy_exp"
        assert result["analysis_git_commit_sha"] == "legacysha"
        assert result["rows"][0]["mean_delta"] == pytest.approx(-0.03)
