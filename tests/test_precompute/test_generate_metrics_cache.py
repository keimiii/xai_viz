"""Tests for metrics-cache schema creation, export, and fine-tuned support."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from app.precompute import generate_metrics_cache as gm
from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


class _NoGetItemAnnotatedDataset:
    """AnnotatedSubset stand-in that fails fast on image-loading access."""

    def __init__(self, image_ids: list[str]) -> None:
        self._image_ids = image_ids
        self._annotations = {
            image_id: SimpleNamespace(styles=(), bboxes=())
            for image_id in image_ids
        }

    @property
    def image_ids(self) -> list[str]:
        return list(self._image_ids)

    @property
    def annotations(self) -> dict[str, SimpleNamespace]:
        return self._annotations

    def __getitem__(self, idx: int) -> dict[str, object]:
        raise AssertionError(
            f"metrics precompute should not index the dataset (attempted idx={idx})"
        )


def test_create_database_migrates_existing_schema_with_continuous_metric_columns(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE image_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            image_id TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            iou REAL NOT NULL,
            coverage REAL NOT NULL,
            attention_area REAL NOT NULL,
            annotation_area REAL NOT NULL,
            UNIQUE(model, layer, method, image_id, percentile)
        )"""
    )
    cursor.execute(
        """CREATE TABLE aggregate_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            median_iou REAL NOT NULL,
            mean_coverage REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, percentile)
        )"""
    )

    cursor.execute(
        """INSERT INTO image_metrics
           (model, layer, method, image_id, percentile, iou, coverage, attention_area, annotation_area)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2", "layer0", "cls", "image.jpg", 90, 0.5, 0.6, 0.1, 0.2),
    )
    cursor.execute(
        """INSERT INTO aggregate_metrics
           (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2", "layer0", "cls", 90, 0.5, 0.1, 0.5, 0.6, 139),
    )
    conn.commit()
    conn.close()

    migrated = gm.create_database(db_path)
    migrated_cursor = migrated.cursor()

    migrated_cursor.execute("PRAGMA table_info(image_metrics)")
    image_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert "mse" in image_columns
    assert "kl" in image_columns
    assert "emd" in image_columns

    migrated_cursor.execute("PRAGMA table_info(aggregate_metrics)")
    aggregate_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert {"mean_mse", "std_mse", "median_mse"}.issubset(aggregate_columns)
    assert {"mean_kl", "std_kl", "median_kl"}.issubset(aggregate_columns)
    assert {"mean_emd", "std_emd", "median_emd"}.issubset(aggregate_columns)

    migrated_cursor.execute("SELECT COUNT(*) FROM image_metrics")
    assert migrated_cursor.fetchone()[0] == 1
    migrated_cursor.execute("SELECT COUNT(*) FROM aggregate_metrics")
    assert migrated_cursor.fetchone()[0] == 1

    migrated.close()


def test_create_database_preserves_metric_generic_breakdown_rows(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO style_metrics
           (model, layer, method, metric, direction, style_name, percentile, mean_score, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("clip", "layer11", "cls", "coverage", "higher", "Gothic", 90, 0.42, 12),
    )
    cursor.execute(
        """INSERT INTO feature_metrics
           (model, layer, method, metric, direction, feature_label, feature_name, percentile, mean_score, std_score, bbox_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("clip", "layer11", "cls", "emd", "lower", 7, "Rose Window", 90, 0.15, 0.01, 9),
    )
    conn.commit()
    conn.close()

    migrated = gm.create_database(db_path)
    migrated_cursor = migrated.cursor()
    migrated_cursor.execute("SELECT COUNT(*) FROM style_metrics")
    assert migrated_cursor.fetchone()[0] == 1
    migrated_cursor.execute("SELECT COUNT(*) FROM feature_metrics")
    assert migrated_cursor.fetchone()[0] == 1
    migrated.close()


def test_create_database_creates_q3_head_tables(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(head_image_metrics)")
    head_image_columns = {row[1] for row in cursor.fetchall()}
    assert {"model", "layer", "method", "head", "metric", "image_id", "percentile", "score"}.issubset(head_image_columns)

    cursor.execute("PRAGMA table_info(head_summary_metrics)")
    head_summary_columns = {row[1] for row in cursor.fetchall()}
    assert {"mean_score", "std_score", "mean_rank", "top1_count", "top3_count", "image_count"}.issubset(head_summary_columns)

    cursor.execute("PRAGMA table_info(head_feature_metrics)")
    head_feature_columns = {row[1] for row in cursor.fetchall()}
    assert {"feature_label", "feature_name", "mean_score", "std_score", "bbox_count"}.issubset(head_feature_columns)

    cursor.execute("PRAGMA table_info(head_feature_image_metrics)")
    head_feature_image_columns = {row[1] for row in cursor.fetchall()}
    assert {"feature_label", "feature_name", "image_id", "score", "bbox_count", "default_bbox_index"}.issubset(
        head_feature_image_columns
    )

    conn.close()


def test_rank_head_scores_respects_metric_direction():
    higher = gm.rank_head_scores({0: 0.25, 1: 0.8, 2: 0.5}, direction="higher")
    lower = gm.rank_head_scores({0: 0.25, 1: 0.8, 2: 0.5}, direction="lower")

    assert higher == {1: 1, 2: 2, 0: 3}
    assert lower == {0: 1, 2: 2, 1: 3}


def test_aggregate_image_feature_scores_respects_direction_and_tie_breaks():
    higher = gm.aggregate_image_feature_scores(
        {7: [(3, 0.25), (1, 0.8), (2, 0.8)]},
        direction="higher",
        feature_names=["placeholder"] * 8,
    )
    lower = gm.aggregate_image_feature_scores(
        {7: [(3, 0.25), (1, 0.1), (2, 0.1)]},
        direction="lower",
        feature_names=["placeholder"] * 8,
    )

    assert higher[7]["default_bbox_index"] == 1
    assert lower[7]["default_bbox_index"] == 1


def test_create_database_recreates_legacy_breakdown_tables(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE style_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            style_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            num_images INTEGER NOT NULL
        )"""
    )
    cursor.execute(
        """CREATE TABLE feature_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            feature_label INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            bbox_count INTEGER NOT NULL
        )"""
    )
    cursor.execute(
        """INSERT INTO style_metrics
           (model, layer, method, style_name, percentile, mean_iou, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("clip", "layer11", "cls", "Gothic", 90, 0.25, 12),
    )
    cursor.execute(
        """INSERT INTO feature_metrics
           (model, layer, method, feature_label, feature_name, percentile, mean_iou, std_iou, bbox_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("clip", "layer11", "cls", 7, "Rose Window", 90, 0.25, 0.05, 9),
    )
    conn.commit()
    conn.close()

    migrated = gm.create_database(db_path)
    migrated_cursor = migrated.cursor()
    migrated_cursor.execute("PRAGMA table_info(style_metrics)")
    style_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert {"metric", "direction", "mean_score"}.issubset(style_columns)
    migrated_cursor.execute("PRAGMA table_info(feature_metrics)")
    feature_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert {"metric", "direction", "mean_score", "std_score"}.issubset(feature_columns)
    migrated_cursor.execute("SELECT COUNT(*) FROM style_metrics")
    assert migrated_cursor.fetchone()[0] == 0
    migrated_cursor.execute("SELECT COUNT(*) FROM feature_metrics")
    assert migrated_cursor.fetchone()[0] == 0
    migrated.close()


def test_compute_metrics_for_finetuned_model_uses_canonical_storage_key(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)

    image_id = "Q1234_test.jpg"
    annotation = SimpleNamespace(styles=(), bboxes=())
    dataset = SimpleNamespace(
        image_ids=[image_id],
        annotations={image_id: annotation},
    )
    attention_cache = MagicMock()
    attention_cache.load.return_value = torch.ones(14, 14)

    monkeypatch.setattr(gm, "compute_image_mse", lambda **_kwargs: 0.01)
    monkeypatch.setattr(gm, "compute_image_kl", lambda **_kwargs: 0.02)
    monkeypatch.setattr(gm, "compute_image_emd", lambda **_kwargs: 0.03)
    monkeypatch.setattr(
        gm,
        "compute_image_iou",
        lambda **_kwargs: SimpleNamespace(
            iou=0.5,
            coverage=0.6,
            attention_area=0.4,
            annotation_area=0.3,
        ),
    )
    monkeypatch.setattr(gm, "load_annotations_with_features", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(gm, "compute_per_bbox_iou", lambda *_args, **_kwargs: [])
    stats = gm.compute_metrics_for_model(
        base_model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=attention_cache,
        conn=conn,
        percentiles=[90],
        layers=[0],
        methods=["cls"],
        skip_existing=False,
        storage_model_key="dinov2_finetuned_full",
        cache_model_keys=["dinov2_finetuned_full", "dinov2_finetuned"],
    )

    cursor = conn.cursor()
    cursor.execute("SELECT model, layer, method, image_id FROM image_metrics")
    image_row = cursor.fetchone()
    cursor.execute("SELECT model, layer, method FROM aggregate_metrics")
    aggregate_row = cursor.fetchone()

    assert stats["processed"] == 1
    assert stats["errors"] == 0
    assert image_row == ("dinov2_finetuned_full", "layer0", "cls", image_id)
    assert aggregate_row == ("dinov2_finetuned_full", "layer0", "cls")
    attention_cache.load.assert_any_call("dinov2_finetuned_full", "layer0", image_id, variant="cls")

    conn.close()


def test_compute_metrics_for_full_strategy_falls_back_to_legacy_cache_key(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)

    image_id = "Q1234_test.jpg"
    annotation = SimpleNamespace(styles=(), bboxes=())
    dataset = SimpleNamespace(
        image_ids=[image_id],
        annotations={image_id: annotation},
    )
    attention_cache = MagicMock()

    def _load(model: str, _layer: str, _image_id: str, variant: str):
        if model == "dinov2_finetuned_full":
            raise KeyError("missing canonical cache")
        assert model == "dinov2_finetuned"
        assert variant == "cls"
        return torch.ones(14, 14)

    attention_cache.load.side_effect = _load

    monkeypatch.setattr(gm, "compute_image_mse", lambda **_kwargs: 0.01)
    monkeypatch.setattr(gm, "compute_image_kl", lambda **_kwargs: 0.02)
    monkeypatch.setattr(gm, "compute_image_emd", lambda **_kwargs: 0.03)
    monkeypatch.setattr(
        gm,
        "compute_image_iou",
        lambda **_kwargs: SimpleNamespace(
            iou=0.5,
            coverage=0.6,
            attention_area=0.4,
            annotation_area=0.3,
        ),
    )
    monkeypatch.setattr(gm, "load_annotations_with_features", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(gm, "compute_per_bbox_iou", lambda *_args, **_kwargs: [])
    stats = gm.compute_metrics_for_model(
        base_model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=attention_cache,
        conn=conn,
        percentiles=[90],
        layers=[0],
        methods=["cls"],
        skip_existing=False,
        storage_model_key="dinov2_finetuned_full",
        cache_model_keys=["dinov2_finetuned_full", "dinov2_finetuned"],
    )

    assert stats["processed"] == 1
    assert stats["errors"] == 0
    attention_cache.load.assert_any_call("dinov2_finetuned_full", "layer0", image_id, variant="cls")
    attention_cache.load.assert_any_call("dinov2_finetuned", "layer0", image_id, variant="cls")

    conn.close()


def test_compute_per_head_metrics_for_model_writes_feature_image_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)

    image_id = "Q1234_test.jpg"
    annotation = ImageAnnotation(
        image_id=image_id,
        styles=(),
        bboxes=(
            BoundingBox(0.1, 0.1, 0.2, 0.2, 7, 7),
            BoundingBox(0.2, 0.2, 0.2, 0.2, 42, 42),
            BoundingBox(0.3, 0.3, 0.2, 0.2, 7, 7),
        ),
    )
    dataset = SimpleNamespace(
        image_ids=[image_id],
        annotations={image_id: annotation},
    )
    attention_cache = MagicMock()

    def _load(_model: str, _layer: str, _image_id: str, variant: str):
        if variant == "cls_head0":
            return torch.ones(14, 14)
        raise KeyError("missing head cache")

    attention_cache.load.side_effect = _load

    feature_types = [SimpleNamespace(name=f"feature_{idx}") for idx in range(43)]
    feature_types[7].name = "Door"
    feature_types[42].name = "Window"

    monkeypatch.setattr(
        gm,
        "compute_image_iou",
        lambda **_kwargs: SimpleNamespace(
            iou=0.5,
            coverage=0.6,
            attention_area=0.4,
            annotation_area=0.3,
        ),
    )
    monkeypatch.setattr(gm, "compute_per_bbox_iou", lambda *_args, **_kwargs: [(7, 0.25), (42, 0.9), (7, 0.8)])
    monkeypatch.setattr(gm, "load_annotations_with_features", lambda *_args, **_kwargs: (None, feature_types))

    stats = gm.compute_per_head_metrics_for_model(
        base_model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=attention_cache,
        conn=conn,
        percentiles=[90],
        layers=[0],
        methods=["cls"],
        skip_existing=False,
    )

    cursor = conn.cursor()
    cursor.execute(
        """SELECT feature_name, image_id, score, bbox_count, default_bbox_index
           FROM head_feature_image_metrics
           WHERE model = ? AND layer = ? AND method = ? AND head = ? AND metric = ? AND feature_label = ? AND percentile = ?""",
        ("dinov2", "layer0", "cls", 0, "iou", 7, 90),
    )
    row = cursor.fetchone()

    assert stats["errors"] == 0
    assert row is not None
    assert row[0] == "Door"
    assert row[1] == image_id
    assert row[2] == pytest.approx(0.525)
    assert row[3] == 2
    assert row[4] == 2

    conn.close()


def test_resolve_models_to_process_filters_finetuned_models():
    models, invalid, non_finetunable = gm.resolve_models_to_process(["all"], finetuned=True)
    assert models == sorted(gm.FINETUNE_MODELS)
    assert invalid == []
    assert non_finetunable == []

    models, invalid, non_finetunable = gm.resolve_models_to_process(
        ["dinov2", "resnet50", "unknown"],
        finetuned=True,
    )
    assert models == ["dinov2"]
    assert invalid == ["unknown"]
    assert non_finetunable == ["resnet50"]


def test_resolve_finetuned_strategies_returns_all_and_invalids():
    strategies, invalid = gm.resolve_finetuned_strategies(["all"])
    assert strategies == [strategy.value for strategy in gm.FINETUNE_STRATEGIES]
    assert invalid == []

    strategies, invalid = gm.resolve_finetuned_strategies(["linear_probe", "bogus", "full"])
    assert strategies == ["linear_probe", "full"]
    assert invalid == ["bogus"]


def test_export_summary_json_excludes_finetuned_rows(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    cursor = conn.cursor()

    aggregate_row = (
        "layer0",
        "cls",
        90,
        0.5,
        0.1,
        0.5,
        0.6,
        0.01,
        0.001,
        0.01,
        0.02,
        0.002,
        0.02,
        0.03,
        0.003,
        0.03,
        1,
    )

    cursor.execute(
        """INSERT INTO aggregate_metrics
           (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage,
            mean_mse, std_mse, median_mse, mean_kl, std_kl, median_kl,
            mean_emd, std_emd, median_emd, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2", *aggregate_row),
    )
    cursor.execute(
        """INSERT INTO aggregate_metrics
           (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage,
            mean_mse, std_mse, median_mse, mean_kl, std_kl, median_kl,
            mean_emd, std_emd, median_emd, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2_finetuned_full", *aggregate_row),
    )
    conn.commit()

    output_path = tmp_path / "metrics_summary.json"
    gm.export_summary_json(conn, output_path)

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(summary["models"]) == {"dinov2"}
    assert [entry["model"] for entry in summary["leaderboard"]] == ["dinov2"]
    assert all(entry["model"] != "dinov2_finetuned" for entry in summary["leaderboards"]["iou"])

    conn.close()


def test_export_summary_json_orders_two_digit_layers_numerically_and_breaks_ties(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    cursor = conn.cursor()

    rows = []
    for idx in reversed(range(12)):
        mean_iou = 0.40 + idx * 0.01
        if idx in {2, 10}:
            mean_iou = 0.99

        rows.append(
            (
                "dinov2",
                f"layer{idx}",
                "cls",
                90,
                mean_iou,
                0.01,
                mean_iou,
                0.50,
                0.30 - idx * 0.01,
                0.01,
                0.30 - idx * 0.01,
                0.20 - idx * 0.005,
                0.01,
                0.20 - idx * 0.005,
                0.10 - idx * 0.003,
                0.01,
                0.10 - idx * 0.003,
                139,
            )
        )

    cursor.executemany(
        """INSERT INTO aggregate_metrics
           (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage,
            mean_mse, std_mse, median_mse, mean_kl, std_kl, median_kl,
            mean_emd, std_emd, median_emd, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    output_path = tmp_path / "metrics_summary.json"
    gm.export_summary_json(conn, output_path)

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    model_summary = summary["models"]["dinov2"]

    assert list(model_summary["layer_progression"]) == [f"layer{i}" for i in range(12)]
    assert model_summary["best_layer"] == "layer2"
    assert summary["leaderboards"]["iou"][0]["best_layer"] == "layer2"

    conn.close()


def test_compute_metrics_for_model_main_loop_never_indexes_dataset(monkeypatch, tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    dataset = _NoGetItemAnnotatedDataset(["Q100_wd0.jpg", "Q200_wd0.jpg"])
    attention_cache = MagicMock()
    attention_cache.load.return_value = torch.ones(14, 14)

    monkeypatch.setattr(gm, "compute_image_mse", lambda **_kwargs: 0.01)
    monkeypatch.setattr(gm, "compute_image_kl", lambda **_kwargs: 0.02)
    monkeypatch.setattr(gm, "compute_image_emd", lambda **_kwargs: 0.03)
    monkeypatch.setattr(
        gm,
        "compute_image_iou",
        lambda **_kwargs: SimpleNamespace(
            iou=0.5,
            coverage=0.6,
            attention_area=0.1,
            annotation_area=0.2,
        ),
    )
    monkeypatch.setattr(gm, "load_annotations_with_features", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(gm, "compute_per_bbox_iou", lambda *_args, **_kwargs: [])
    stats = gm.compute_metrics_for_model(
        base_model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=attention_cache,
        conn=conn,
        percentiles=[90],
        layers=[0],
        methods=["cls"],
        skip_existing=False,
    )

    assert stats["processed"] == len(dataset.image_ids)
    assert stats["errors"] == 0

    conn.close()


def test_compute_metrics_for_model_feature_breakdown_never_indexes_dataset(monkeypatch, tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = gm.create_database(db_path)
    dataset = _NoGetItemAnnotatedDataset(["Q300_wd0.jpg"])
    attention_cache = MagicMock()
    attention_cache.load.return_value = torch.ones(14, 14)
    per_bbox_iou = MagicMock(return_value=[(0, 0.5)])

    monkeypatch.setattr(gm, "compute_image_mse", lambda **_kwargs: 0.01)
    monkeypatch.setattr(gm, "compute_image_kl", lambda **_kwargs: 0.02)
    monkeypatch.setattr(gm, "compute_image_emd", lambda **_kwargs: 0.03)
    monkeypatch.setattr(
        gm,
        "compute_image_iou",
        lambda **_kwargs: SimpleNamespace(
            iou=0.5,
            coverage=0.6,
            attention_area=0.1,
            annotation_area=0.2,
        ),
    )
    monkeypatch.setattr(
        gm,
        "load_annotations_with_features",
        lambda *_args, **_kwargs: (None, [SimpleNamespace(name="window")]),
    )
    monkeypatch.setattr(gm, "compute_per_bbox_iou", per_bbox_iou)
    gm.compute_metrics_for_model(
        base_model_name="dinov2",
        dataset=dataset,  # type: ignore[arg-type]
        attention_cache=attention_cache,
        conn=conn,
        percentiles=[90],
        layers=[0],
        methods=["cls"],
        skip_existing=False,
    )

    cursor = conn.cursor()
    cursor.execute("SELECT feature_name, metric, mean_score, bbox_count FROM feature_metrics")
    feature_row = cursor.fetchone()

    assert per_bbox_iou.call_count == len(dataset.image_ids)
    assert feature_row == ("window", "iou", 0.5, 1)

    conn.close()
