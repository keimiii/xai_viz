"""Generate pre-computed metrics cache for the visualization app.

Computes IoU, coverage, Gaussian-ground-truth MSE, Gaussian-ground-truth
KL divergence, and Gaussian-ground-truth EMD/Wasserstein-1 metrics for all
model/layer/image combinations at multiple percentile thresholds, storing
results in SQLite for fast queries.

Usage:
    python -m app.precompute.generate_metrics_cache
    python -m app.precompute.generate_metrics_cache --models dinov2 clip

Usage (fine-tuned models):
    python -m app.precompute.generate_metrics_cache --finetuned --models all
    python -m app.precompute.generate_metrics_cache --finetuned --models dinov2
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import display_model_name
from ssl_attention.cache import AttentionCache
from ssl_attention.config import (
    ANNOTATIONS_PATH,
    CACHE_PATH,
    DATASET_PATH,
    DEFAULT_METHOD,
    FINETUNE_MODELS,
    FINETUNE_STRATEGIES,
    MODEL_METHODS,
    MODELS,
    STYLE_MAPPING,
    STYLE_NAMES,
)
from ssl_attention.data import AnnotatedSubset
from ssl_attention.data.annotations import load_annotations_with_features
from ssl_attention.evaluation.fine_tuning import get_finetuned_cache_key
from ssl_attention.metrics import (
    compute_image_emd,
    compute_image_iou,
    compute_image_kl,
    compute_image_mse,
)
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_kl_divergence,
    compute_mse,
    gaussian_bbox_heatmap,
)
from ssl_attention.metrics.iou import compute_coverage, compute_per_bbox_iou

DEFAULT_PERCENTILES = [90, 85, 80, 75, 70, 60, 50]
IMAGE_METRIC_MIGRATIONS = {"mse": "REAL", "kl": "REAL", "emd": "REAL"}
AGGREGATE_METRIC_MIGRATIONS = {
    "mean_mse": "REAL",
    "std_mse": "REAL",
    "median_mse": "REAL",
    "mean_kl": "REAL",
    "std_kl": "REAL",
    "median_kl": "REAL",
    "mean_emd": "REAL",
    "std_emd": "REAL",
    "median_emd": "REAL",
}
_FINETUNED_SUFFIX = "_finetuned"
BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE = 90
PER_HEAD_METHODS = {"cls", "mean"}
BREAKDOWN_METRICS = {
    "iou": {"direction": "higher", "percentile_dependent": True, "column": "iou"},
    "coverage": {"direction": "higher", "percentile_dependent": False, "column": "coverage"},
    "mse": {"direction": "lower", "percentile_dependent": False, "column": "mse"},
    "kl": {"direction": "lower", "percentile_dependent": False, "column": "kl"},
    "emd": {"direction": "lower", "percentile_dependent": False, "column": "emd"},
}
STYLE_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "metric": "TEXT",
    "direction": "TEXT",
    "style_name": "TEXT",
    "percentile": "INTEGER",
    "mean_score": "REAL",
    "num_images": "INTEGER",
}
FEATURE_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "metric": "TEXT",
    "direction": "TEXT",
    "feature_label": "INTEGER",
    "feature_name": "TEXT",
    "percentile": "INTEGER",
    "mean_score": "REAL",
    "std_score": "REAL",
    "bbox_count": "INTEGER",
}
HEAD_IMAGE_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "head": "INTEGER",
    "metric": "TEXT",
    "direction": "TEXT",
    "image_id": "TEXT",
    "percentile": "INTEGER",
    "score": "REAL",
}
HEAD_SUMMARY_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "head": "INTEGER",
    "metric": "TEXT",
    "direction": "TEXT",
    "percentile": "INTEGER",
    "mean_score": "REAL",
    "std_score": "REAL",
    "mean_rank": "REAL",
    "top1_count": "INTEGER",
    "top3_count": "INTEGER",
    "image_count": "INTEGER",
}
HEAD_FEATURE_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "head": "INTEGER",
    "metric": "TEXT",
    "direction": "TEXT",
    "feature_label": "INTEGER",
    "feature_name": "TEXT",
    "percentile": "INTEGER",
    "mean_score": "REAL",
    "std_score": "REAL",
    "bbox_count": "INTEGER",
}
HEAD_FEATURE_IMAGE_METRICS_COLUMNS = {
    "id": "INTEGER",
    "model": "TEXT",
    "layer": "TEXT",
    "method": "TEXT",
    "head": "INTEGER",
    "metric": "TEXT",
    "direction": "TEXT",
    "feature_label": "INTEGER",
    "feature_name": "TEXT",
    "image_id": "TEXT",
    "percentile": "INTEGER",
    "score": "REAL",
    "bbox_count": "INTEGER",
    "default_bbox_index": "INTEGER",
}


def metric_query_percentiles(metric_name: str, percentiles: list[int]) -> list[int]:
    """Return the relevant percentile keys for a metric."""
    metric_config = BREAKDOWN_METRICS[metric_name]
    if metric_config["percentile_dependent"]:
        return list(percentiles)
    return [BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE]


def ensure_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, str],
) -> None:
    """Add missing columns to an existing SQLite table in place."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in expected_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> dict[str, str]:
    """Return the current column map for a SQLite table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]): str(row[2]).upper() for row in cursor.fetchall()}


def recreate_metric_breakdown_tables(conn: sqlite3.Connection) -> None:
    """Create the metric-generic style and feature breakdown tables."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS style_metrics")
    cursor.execute("DROP TABLE IF EXISTS feature_metrics")

    cursor.execute("""
        CREATE TABLE style_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            style_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, metric, style_name, percentile)
        )
    """)

    cursor.execute("""
        CREATE TABLE feature_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            feature_label INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            std_score REAL NOT NULL,
            bbox_count INTEGER NOT NULL,
            UNIQUE(model, layer, method, metric, feature_label, percentile)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_style_model_layer_metric ON style_metrics(model, layer, method, metric, percentile)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_model_layer_metric ON feature_metrics(model, layer, method, metric, percentile)"
    )


def ensure_metric_breakdown_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate metric-generic breakdown tables in place.

    Recreate only when an older IoU-only schema is detected. If the tables
    already match the current metric-generic shape, preserve the existing rows
    so frozen and fine-tuned passes can append safely into the same database.
    """
    style_columns = get_table_columns(conn, "style_metrics")
    feature_columns = get_table_columns(conn, "feature_metrics")

    style_matches = style_columns == STYLE_METRICS_COLUMNS
    feature_matches = feature_columns == FEATURE_METRICS_COLUMNS

    if style_matches and feature_matches:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_style_model_layer_metric ON style_metrics(model, layer, method, metric, percentile)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_feature_model_layer_metric ON feature_metrics(model, layer, method, metric, percentile)"
        )
        return

    recreate_metric_breakdown_tables(conn)


def create_head_metric_tables(conn: sqlite3.Connection) -> None:
    """Create Q3 per-head metric tables if they do not exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS head_image_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            head INTEGER NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            image_id TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            score REAL NOT NULL,
            UNIQUE(model, layer, method, head, metric, image_id, percentile)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS head_summary_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            head INTEGER NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            std_score REAL NOT NULL,
            mean_rank REAL NOT NULL,
            top1_count INTEGER NOT NULL,
            top3_count INTEGER NOT NULL,
            image_count INTEGER NOT NULL,
            UNIQUE(model, layer, method, head, metric, percentile)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS head_feature_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            head INTEGER NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            feature_label INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            std_score REAL NOT NULL,
            bbox_count INTEGER NOT NULL,
            UNIQUE(model, layer, method, head, metric, feature_label, percentile)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS head_feature_image_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            head INTEGER NOT NULL,
            metric TEXT NOT NULL,
            direction TEXT NOT NULL,
            feature_label INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            image_id TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            score REAL NOT NULL,
            bbox_count INTEGER NOT NULL,
            default_bbox_index INTEGER,
            UNIQUE(model, layer, method, head, metric, feature_label, image_id, percentile)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_head_image_metric_lookup "
        "ON head_image_metrics(model, layer, method, metric, percentile, image_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_head_summary_metric_lookup "
        "ON head_summary_metrics(model, layer, method, metric, percentile)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_head_feature_metric_lookup "
        "ON head_feature_metrics(model, layer, method, metric, percentile)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_head_feature_image_metric_lookup "
        "ON head_feature_image_metrics(model, layer, method, head, metric, feature_label, percentile)"
    )


def ensure_head_metric_schema(conn: sqlite3.Connection) -> None:
    """Ensure the Q3 per-head metric tables expose the current schema."""
    create_head_metric_tables(conn)
    ensure_table_columns(conn, "head_image_metrics", HEAD_IMAGE_METRICS_COLUMNS)
    ensure_table_columns(conn, "head_summary_metrics", HEAD_SUMMARY_METRICS_COLUMNS)
    ensure_table_columns(conn, "head_feature_metrics", HEAD_FEATURE_METRICS_COLUMNS)
    ensure_table_columns(conn, "head_feature_image_metrics", HEAD_FEATURE_IMAGE_METRICS_COLUMNS)


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create metrics database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Per-image metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_metrics (
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
            mse REAL,
            kl REAL,
            emd REAL,
            UNIQUE(model, layer, method, image_id, percentile)
        )
    """)

    # Aggregate metrics table (per model/layer/method)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aggregate_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            median_iou REAL NOT NULL,
            mean_coverage REAL NOT NULL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, percentile)
        )
    """)

    ensure_metric_breakdown_schema(conn)
    ensure_head_metric_schema(conn)

    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_model_layer ON image_metrics(model, layer, method)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_image_id ON image_metrics(image_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agg_model ON aggregate_metrics(model, method)")

    ensure_table_columns(conn, "image_metrics", IMAGE_METRIC_MIGRATIONS)
    ensure_table_columns(conn, "aggregate_metrics", AGGREGATE_METRIC_MIGRATIONS)

    conn.commit()
    return conn


def aggregate_feature_scores(
    scores_by_label: dict[int, list[float]],
    feature_names: list[str] | None = None,
) -> dict[int, dict[str, float | str]]:
    """Aggregate per-bbox metric values by architectural feature label."""
    aggregated: dict[int, dict[str, float | str]] = {}
    for label, scores in scores_by_label.items():
        if not scores:
            continue
        tensor = torch.tensor(scores, dtype=torch.float32)
        aggregated[label] = {
            "mean_score": tensor.mean().item(),
            "std_score": tensor.std(unbiased=tensor.numel() > 1).item(),
            "count": float(tensor.numel()),
            "name": feature_names[label] if feature_names and label < len(feature_names) else f"feature_{label}",
        }
    return aggregated


def aggregate_image_feature_scores(
    scores_by_label: dict[int, list[tuple[int, float]]],
    *,
    direction: str,
    feature_names: list[str] | None = None,
) -> dict[int, dict[str, float | str | int]]:
    """Aggregate one image's per-bbox metric values by feature label."""
    aggregated: dict[int, dict[str, float | str | int]] = {}
    reverse = direction == "higher"

    for label, indexed_scores in scores_by_label.items():
        if not indexed_scores:
            continue

        tensor = torch.tensor([score for _bbox_index, score in indexed_scores], dtype=torch.float32)
        default_bbox_index = sorted(
            indexed_scores,
            key=lambda item: (-item[1], item[0]) if reverse else (item[1], item[0]),
        )[0][0]
        aggregated[label] = {
            "mean_score": tensor.mean().item(),
            "count": int(tensor.numel()),
            "name": feature_names[label] if feature_names and label < len(feature_names) else f"feature_{label}",
            "default_bbox_index": default_bbox_index,
        }

    return aggregated


def rank_head_scores(
    scores_by_head: dict[int, float],
    *,
    direction: str,
) -> dict[int, int]:
    """Rank head scores for one image/metric combination."""
    reverse = direction == "higher"
    ordered = sorted(
        scores_by_head.items(),
        key=lambda item: (-item[1], item[0]) if reverse else (item[1], item[0]),
    )
    return {
        head: rank
        for rank, (head, _score) in enumerate(ordered, start=1)
    }


def compute_metrics_for_model(
    base_model_name: str,
    dataset: AnnotatedSubset,
    attention_cache: AttentionCache,
    conn: sqlite3.Connection,
    percentiles: list[int],
    layers: list[int] | None = None,
    methods: list[str] | None = None,
    skip_existing: bool = True,
    storage_model_key: str | None = None,
    cache_model_keys: list[str] | None = None,
) -> dict[str, int]:
    """Compute metrics for a single model across all its attention methods.

    Args:
        base_model_name: Base model name used for config and method lookup.
        dataset: Annotated dataset.
        attention_cache: AttentionCache with pre-computed attention.
        conn: SQLite connection.
        percentiles: List of percentile thresholds.
        layers: Specific layers to process. None = all.
        methods: Specific methods to process. None = all for this model.
        skip_existing: Skip if already in database.
        storage_model_key: Model key used for SQLite writes.
        cache_model_keys: Ordered model keys to try when reading cached attention.

    Returns:
        Dict with statistics.
    """
    import torch

    stats = {"processed": 0, "skipped": 0, "errors": 0}
    model_key = storage_model_key or base_model_name
    cache_keys = cache_model_keys or [model_key]

    def _std_value(values: torch.Tensor) -> float:
        """Return a stable std-dev value even for singleton aggregates."""
        return values.std(unbiased=values.numel() > 1).item()

    model_config = MODELS[base_model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    # Get all methods for this model (or filter by CLI arg)
    all_methods = [m.value for m in MODEL_METHODS[base_model_name]]
    methods_to_process = [m for m in methods if m in all_methods] if methods else all_methods

    print(f"\nProcessing {model_key} ({len(layers_to_process)} layers, methods: {methods_to_process})")

    cursor = conn.cursor()

    # Build style mapping for images (metadata-only, no image I/O)
    image_styles: dict[str, str | None] = {}
    for image_id in dataset.image_ids:
        annotation = dataset.annotations[image_id]
        style_name = None
        for style_qid in annotation.styles:
            if style_qid in STYLE_MAPPING:
                style_name = STYLE_NAMES[STYLE_MAPPING[style_qid]]
                break
        image_styles[image_id] = style_name

    for variant in methods_to_process:
        print(f"  Method: {variant}")

        # Process each image (metadata-only, no image I/O)
        for image_id in tqdm(dataset.image_ids, desc=f"{model_key}/{variant}"):
            annotation = dataset.annotations[image_id]

            for layer in layers_to_process:
                layer_key = f"layer{layer}"

                # Load attention from cache
                attention = None
                for cache_key in cache_keys:
                    try:
                        attention = attention_cache.load(cache_key, layer_key, image_id, variant=variant)
                        break
                    except KeyError:
                        continue

                if attention is None:
                    stats["skipped"] += len(percentiles)
                    continue

                try:
                    image_mse = compute_image_mse(attention=attention, annotation=annotation)
                    image_kl = compute_image_kl(attention=attention, annotation=annotation)
                    image_emd = compute_image_emd(attention=attention, annotation=annotation)
                except Exception as e:
                    print(f"\nError computing continuous metrics for {image_id} layer{layer}/{variant}: {e}")
                    stats["errors"] += len(percentiles)
                    continue

                for percentile in percentiles:
                    # Check if already exists
                    if skip_existing:
                        cursor.execute(
                            """SELECT mse, kl, emd FROM image_metrics
                               WHERE model=? AND layer=? AND method=? AND image_id=? AND percentile=?""",
                            (model_key, layer_key, variant, image_id, percentile),
                        )
                        existing_row = cursor.fetchone()
                        if (
                            existing_row
                            and existing_row[0] is not None
                            and existing_row[1] is not None
                            and existing_row[2] is not None
                        ):
                            stats["skipped"] += 1
                            continue

                    try:
                        result = compute_image_iou(
                            attention=attention,
                            annotation=annotation,
                            image_id=image_id,
                            percentile=percentile,
                        )

                        # Insert into database
                        cursor.execute(
                            """INSERT OR REPLACE INTO image_metrics
                               (model, layer, method, image_id, percentile, iou, coverage, attention_area, annotation_area, mse, kl, emd)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                model_key,
                                layer_key,
                                variant,
                                image_id,
                                percentile,
                                result.iou,
                                result.coverage,
                                result.attention_area,
                                result.annotation_area,
                                image_mse,
                                image_kl,
                                image_emd,
                            ),
                        )

                        stats["processed"] += 1

                    except Exception as e:
                        print(f"\nError computing metrics for {image_id} layer{layer}/{variant}: {e}")
                        stats["errors"] += 1

        conn.commit()

        # Compute aggregate metrics for this method
        print(f"  Computing aggregates for {model_key}/{variant}...")
        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            for percentile in percentiles:
                cursor.execute(
                    """SELECT iou, coverage, mse, kl, emd FROM image_metrics
                       WHERE model=? AND layer=? AND method=? AND percentile=?""",
                    (model_key, layer_key, variant, percentile),
                )
                rows = cursor.fetchall()

                if not rows:
                    continue

                complete_rows = [row for row in rows if row[2] is not None and row[3] is not None and row[4] is not None]
                if not complete_rows:
                    continue

                ious = torch.tensor([r[0] for r in complete_rows])
                coverages = torch.tensor([r[1] for r in complete_rows])
                mses = torch.tensor([r[2] for r in complete_rows], dtype=torch.float32)
                kls = torch.tensor([r[3] for r in complete_rows], dtype=torch.float32)
                emds = torch.tensor([r[4] for r in complete_rows], dtype=torch.float32)

                cursor.execute(
                    """INSERT OR REPLACE INTO aggregate_metrics
                       (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage,
                        mean_mse, std_mse, median_mse, mean_kl, std_kl, median_kl,
                        mean_emd, std_emd, median_emd, num_images)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_key,
                        layer_key,
                        variant,
                        percentile,
                        ious.mean().item(),
                        _std_value(ious),
                        ious.median().item(),
                        coverages.mean().item(),
                        mses.mean().item(),
                        _std_value(mses),
                        mses.median().item(),
                        kls.mean().item(),
                        _std_value(kls),
                        kls.median().item(),
                        emds.mean().item(),
                        _std_value(emds),
                        emds.median().item(),
                        len(complete_rows),
                    ),
                )

        conn.commit()

        # Compute style breakdown for this method
        print(f"  Computing style breakdown for {model_key}/{variant}...")
        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            for metric_name, metric_config in BREAKDOWN_METRICS.items():
                query_percentiles = (
                    percentiles
                    if metric_config["percentile_dependent"]
                    else [BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE]
                )
                metric_column = metric_config["column"]

                for percentile in query_percentiles:
                    for style_name in STYLE_NAMES:
                        style_images = [img_id for img_id, s in image_styles.items() if s == style_name]
                        if not style_images:
                            continue

                        placeholders = ",".join("?" * len(style_images))
                        cursor.execute(
                            f"""SELECT AVG({metric_column}), COUNT(*) FROM image_metrics
                               WHERE model=? AND layer=? AND method=? AND percentile=? AND image_id IN ({placeholders})""",
                            (model_key, layer_key, variant, percentile, *style_images),
                        )
                        row = cursor.fetchone()

                        if row and row[0] is not None:
                            cursor.execute(
                                """INSERT OR REPLACE INTO style_metrics
                                   (model, layer, method, metric, direction, style_name, percentile, mean_score, num_images)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    model_key,
                                    layer_key,
                                    variant,
                                    metric_name,
                                    metric_config["direction"],
                                    style_name,
                                    percentile,
                                    row[0],
                                    row[1],
                                ),
                            )

        conn.commit()

        # Compute feature breakdown for this method
        print(f"  Computing feature breakdown for {model_key}/{variant}...")
        _, feature_types = load_annotations_with_features(ANNOTATIONS_PATH)
        feature_names = [ft.name for ft in feature_types]

        for layer in layers_to_process:
            layer_key = f"layer{layer}"
            feature_scores: dict[tuple[str, int], dict[int, list[float]]] = {
                ("iou", percentile): {} for percentile in percentiles
            }
            for metric_name in ("coverage", "mse", "kl", "emd"):
                feature_scores[(metric_name, BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)] = {}

            for image_id in dataset.image_ids:
                annotation = dataset.annotations[image_id]
                try:
                    attention = attention_cache.load(model_key, layer_key, image_id, variant=variant)
                except KeyError:
                    continue

                for percentile in percentiles:
                    bbox_ious = compute_per_bbox_iou(attention, annotation, percentile)
                    for label, score in bbox_ious:
                        feature_scores[("iou", percentile)].setdefault(label, []).append(score)

                height, width = attention.shape[-2:]
                for bbox in annotation.bboxes:
                    label = bbox.label
                    bbox_mask = bbox.to_mask(height, width).to(attention.device)
                    bbox_heatmap = gaussian_bbox_heatmap(
                        bbox,
                        height,
                        width,
                        device=attention.device,
                    )
                    feature_scores[("coverage", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                        label,
                        [],
                    ).append(compute_coverage(attention, bbox_mask))
                    feature_scores[("mse", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                        label,
                        [],
                    ).append(compute_mse(attention, bbox_heatmap))
                    feature_scores[("kl", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                        label,
                        [],
                    ).append(compute_kl_divergence(attention, bbox_heatmap))
                    feature_scores[("emd", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                        label,
                        [],
                    ).append(compute_emd(attention, bbox_heatmap))

            for (metric_name, percentile), scores_by_label in feature_scores.items():
                metric_config = BREAKDOWN_METRICS[metric_name]
                feature_stats = aggregate_feature_scores(scores_by_label, feature_names)

                for label, stats_dict in feature_stats.items():
                    feature_name = stats_dict.get("name", f"feature_{label}")
                    cursor.execute(
                        """INSERT OR REPLACE INTO feature_metrics
                           (model, layer, method, metric, direction, feature_label, feature_name, percentile, mean_score, std_score, bbox_count)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            model_key,
                            layer_key,
                            variant,
                            metric_name,
                            metric_config["direction"],
                            label,
                            feature_name,
                            percentile,
                            stats_dict["mean_score"],
                            stats_dict["std_score"],
                            int(stats_dict["count"]),
                        ),
                    )

        conn.commit()

    return stats


def compute_per_head_metrics_for_model(
    base_model_name: str,
    dataset: AnnotatedSubset,
    attention_cache: AttentionCache,
    conn: sqlite3.Connection,
    percentiles: list[int],
    layers: list[int] | None = None,
    methods: list[str] | None = None,
    skip_existing: bool = True,
    storage_model_key: str | None = None,
    cache_model_keys: list[str] | None = None,
) -> dict[str, int]:
    """Compute Q3 per-head metrics for a single model."""
    stats = {"processed": 0, "skipped": 0, "errors": 0}
    model_key = storage_model_key or base_model_name
    cache_keys = cache_model_keys or [model_key]
    model_config = MODELS[base_model_name]

    if model_config.num_heads <= 1:
        return stats

    def _std_value(values: torch.Tensor) -> float:
        return values.std(unbiased=values.numel() > 1).item()

    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))
    all_methods = [m.value for m in MODEL_METHODS[base_model_name] if m.value in PER_HEAD_METHODS]
    methods_to_process = [m for m in methods if m in all_methods] if methods else all_methods

    if not methods_to_process:
        return stats

    print(f"\nProcessing per-head metrics for {model_key} ({len(layers_to_process)} layers, methods: {methods_to_process})")

    cursor = conn.cursor()
    _, feature_types = load_annotations_with_features(ANNOTATIONS_PATH)
    feature_names = [ft.name for ft in feature_types]
    query_percentiles_by_metric = {
        metric_name: metric_query_percentiles(metric_name, percentiles)
        for metric_name in BREAKDOWN_METRICS
    }
    expected_summary_rows = model_config.num_heads * sum(
        len(query_percentiles_by_metric[metric_name]) for metric_name in BREAKDOWN_METRICS
    )

    for variant in methods_to_process:
        print(f"  Method: {variant}")

        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            if skip_existing:
                cursor.execute(
                    """SELECT COUNT(*) FROM head_summary_metrics
                       WHERE model = ? AND layer = ? AND method = ?""",
                    (model_key, layer_key, variant),
                )
                existing_count = cursor.fetchone()[0]
                cursor.execute(
                    """SELECT COUNT(*) FROM head_feature_image_metrics
                       WHERE model = ? AND layer = ? AND method = ?""",
                    (model_key, layer_key, variant),
                )
                feature_image_count = cursor.fetchone()[0]
                if existing_count >= expected_summary_rows and feature_image_count > 0:
                    stats["skipped"] += existing_count
                    continue

            print(f"    Layer: {layer_key}")
            score_accumulator: dict[tuple[str, int, int], list[float]] = {}
            rank_accumulator: dict[tuple[str, int, int], list[float]] = {}
            top1_counts: dict[tuple[str, int, int], int] = {}
            top3_counts: dict[tuple[str, int, int], int] = {}
            feature_scores: dict[tuple[str, int, int], dict[int, list[float]]] = {}

            for metric_name, query_percentiles in query_percentiles_by_metric.items():
                for percentile in query_percentiles:
                    for head_idx in range(model_config.num_heads):
                        key = (metric_name, percentile, head_idx)
                        score_accumulator[key] = []
                        rank_accumulator[key] = []
                        top1_counts[key] = 0
                        top3_counts[key] = 0
                        feature_scores[key] = {}

            for image_id in tqdm(dataset.image_ids, desc=f"{model_key}/{variant}/{layer_key}"):
                annotation = dataset.annotations[image_id]
                image_metric_scores: dict[tuple[str, int], dict[int, float]] = {}

                for head_idx in range(model_config.num_heads):
                    per_head_variant = f"{variant}_head{head_idx}"
                    attention = None
                    for cache_key in cache_keys:
                        try:
                            attention = attention_cache.load(
                                cache_key,
                                layer_key,
                                image_id,
                                variant=per_head_variant,
                            )
                            break
                        except KeyError:
                            continue

                    if attention is None:
                        continue

                    try:
                        height, width = attention.shape[-2:]
                        union_mask = annotation.get_union_mask(height, width).to(attention.device)
                        union_heatmap = annotation_to_gaussian_heatmap(
                            annotation,
                            height,
                            width,
                            device=attention.device,
                        )
                        bbox_targets = [
                            (
                                bbox_index,
                                bbox.label,
                                bbox.to_mask(height, width).to(attention.device),
                                gaussian_bbox_heatmap(
                                    bbox,
                                    height,
                                    width,
                                    device=attention.device,
                                ),
                            )
                            for bbox_index, bbox in enumerate(annotation.bboxes)
                        ]
                        image_feature_scores: dict[tuple[str, int], dict[int, list[tuple[int, float]]]] = {}

                        coverage = compute_coverage(attention, union_mask)
                        mse = compute_mse(attention, union_heatmap)
                        kl = compute_kl_divergence(attention, union_heatmap)
                        emd = compute_emd(attention, union_heatmap)

                        metric_values = {
                            ("coverage", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE): coverage,
                            ("mse", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE): mse,
                            ("kl", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE): kl,
                            ("emd", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE): emd,
                        }

                        for percentile in percentiles:
                            iou_result = compute_image_iou(
                                attention=attention,
                                annotation=annotation,
                                image_id=image_id,
                                percentile=percentile,
                            )
                            metric_values[("iou", percentile)] = iou_result.iou

                            bbox_ious = compute_per_bbox_iou(attention, annotation, percentile)
                            image_feature_scores[("iou", percentile)] = {}
                            for bbox_index, (label, score) in enumerate(bbox_ious):
                                feature_scores[("iou", percentile, head_idx)].setdefault(label, []).append(score)
                                image_feature_scores[("iou", percentile)].setdefault(label, []).append((bbox_index, score))

                        image_feature_scores[("coverage", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)] = {}
                        image_feature_scores[("mse", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)] = {}
                        image_feature_scores[("kl", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)] = {}
                        image_feature_scores[("emd", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)] = {}
                        for bbox_index, label, bbox_mask, bbox_heatmap in bbox_targets:
                            coverage_score = compute_coverage(attention, bbox_mask)
                            mse_score = compute_mse(attention, bbox_heatmap)
                            kl_score = compute_kl_divergence(attention, bbox_heatmap)
                            emd_score = compute_emd(attention, bbox_heatmap)
                            feature_scores[("coverage", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE, head_idx)].setdefault(
                                label,
                                [],
                            ).append(coverage_score)
                            image_feature_scores[("coverage", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                                label,
                                [],
                            ).append((bbox_index, coverage_score))
                            feature_scores[("mse", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE, head_idx)].setdefault(
                                label,
                                [],
                            ).append(mse_score)
                            image_feature_scores[("mse", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                                label,
                                [],
                            ).append((bbox_index, mse_score))
                            feature_scores[("kl", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE, head_idx)].setdefault(
                                label,
                                [],
                            ).append(kl_score)
                            image_feature_scores[("kl", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                                label,
                                [],
                            ).append((bbox_index, kl_score))
                            feature_scores[("emd", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE, head_idx)].setdefault(
                                label,
                                [],
                            ).append(emd_score)
                            image_feature_scores[("emd", BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE)].setdefault(
                                label,
                                [],
                            ).append((bbox_index, emd_score))

                        for (metric_name, percentile), score in metric_values.items():
                            metric_config = BREAKDOWN_METRICS[metric_name]
                            cursor.execute(
                                """INSERT OR REPLACE INTO head_image_metrics
                                   (model, layer, method, head, metric, direction, image_id, percentile, score)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    model_key,
                                    layer_key,
                                    variant,
                                    head_idx,
                                    metric_name,
                                    metric_config["direction"],
                                    image_id,
                                    percentile,
                                    score,
                                ),
                            )
                            score_accumulator[(metric_name, percentile, head_idx)].append(score)
                            image_metric_scores.setdefault((metric_name, percentile), {})[head_idx] = score
                            stats["processed"] += 1

                        for (metric_name, percentile), image_scores_by_label in image_feature_scores.items():
                            metric_config = BREAKDOWN_METRICS[metric_name]
                            feature_stats = aggregate_image_feature_scores(
                                image_scores_by_label,
                                direction=str(metric_config["direction"]),
                                feature_names=feature_names,
                            )
                            for label, stats_dict in feature_stats.items():
                                feature_name = str(stats_dict.get("name", f"feature_{label}"))
                                default_bbox_index = stats_dict.get("default_bbox_index")
                                cursor.execute(
                                    """INSERT OR REPLACE INTO head_feature_image_metrics
                                       (model, layer, method, head, metric, direction, feature_label, feature_name, image_id, percentile, score, bbox_count, default_bbox_index)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        model_key,
                                        layer_key,
                                        variant,
                                        head_idx,
                                        metric_name,
                                        metric_config["direction"],
                                        label,
                                        feature_name,
                                        image_id,
                                        percentile,
                                        stats_dict["mean_score"],
                                        int(stats_dict["count"]),
                                        int(default_bbox_index) if isinstance(default_bbox_index, int) else None,
                                    ),
                                )

                    except Exception as exc:
                        print(
                            f"\nError computing per-head metrics for "
                            f"{image_id} {layer_key}/{per_head_variant}: {exc}"
                        )
                        stats["errors"] += 1

                for (metric_name, percentile), scores_by_head in image_metric_scores.items():
                    metric_config = BREAKDOWN_METRICS[metric_name]
                    direction = str(metric_config["direction"])
                    ranks = rank_head_scores(scores_by_head, direction=direction)
                    ordered_heads = sorted(ranks, key=lambda head: ranks[head])
                    for head_idx, rank in ranks.items():
                        rank_accumulator[(metric_name, percentile, head_idx)].append(float(rank))
                    if ordered_heads:
                        top1_counts[(metric_name, percentile, ordered_heads[0])] += 1
                    for head_idx in ordered_heads[:3]:
                        top3_counts[(metric_name, percentile, head_idx)] += 1

            for (metric_name, percentile, head_idx), scores in score_accumulator.items():
                if not scores:
                    continue

                rank_values = rank_accumulator[(metric_name, percentile, head_idx)]
                score_tensor = torch.tensor(scores, dtype=torch.float32)
                rank_tensor = torch.tensor(rank_values, dtype=torch.float32)
                metric_config = BREAKDOWN_METRICS[metric_name]
                cursor.execute(
                    """INSERT OR REPLACE INTO head_summary_metrics
                       (model, layer, method, head, metric, direction, percentile, mean_score, std_score, mean_rank, top1_count, top3_count, image_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_key,
                        layer_key,
                        variant,
                        head_idx,
                        metric_name,
                        metric_config["direction"],
                        percentile,
                        score_tensor.mean().item(),
                        _std_value(score_tensor),
                        rank_tensor.mean().item() if rank_values else 0.0,
                        top1_counts[(metric_name, percentile, head_idx)],
                        top3_counts[(metric_name, percentile, head_idx)],
                        len(scores),
                    ),
                )

            for (metric_name, percentile, head_idx), feature_scores_by_label in feature_scores.items():
                metric_config = BREAKDOWN_METRICS[metric_name]
                feature_stats = aggregate_feature_scores(feature_scores_by_label, feature_names)
                for label, stats_dict in feature_stats.items():
                    feature_name = str(stats_dict.get("name", f"feature_{label}"))
                    cursor.execute(
                        """INSERT OR REPLACE INTO head_feature_metrics
                           (model, layer, method, head, metric, direction, feature_label, feature_name, percentile, mean_score, std_score, bbox_count)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            model_key,
                            layer_key,
                            variant,
                            head_idx,
                            metric_name,
                            metric_config["direction"],
                            label,
                            feature_name,
                            percentile,
                            stats_dict["mean_score"],
                            stats_dict["std_score"],
                            int(stats_dict["count"]),
                        ),
                    )

            conn.commit()

    return stats


def resolve_models_to_process(
    requested_models: list[str],
    *,
    finetuned: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve requested model names for frozen or fine-tuned generation."""
    if "all" in requested_models:
        return (
            sorted(FINETUNE_MODELS) if finetuned else list(MODELS.keys()),
            [],
            [],
        )

    models_to_process = [model for model in requested_models if model in MODELS]
    invalid_models = [model for model in requested_models if model not in MODELS]
    non_finetunable_models: list[str] = []

    if finetuned:
        non_finetunable_models = [model for model in models_to_process if model not in FINETUNE_MODELS]
        models_to_process = [model for model in models_to_process if model in FINETUNE_MODELS]

    return models_to_process, invalid_models, non_finetunable_models


def resolve_finetuned_strategies(requested_strategies: list[str]) -> tuple[list[str], list[str]]:
    """Resolve requested fine-tuning strategies."""
    valid_strategies = [strategy.value for strategy in FINETUNE_STRATEGIES]

    if "all" in requested_strategies:
        return valid_strategies, []

    resolved = [strategy for strategy in requested_strategies if strategy in valid_strategies]
    invalid = [strategy for strategy in requested_strategies if strategy not in valid_strategies]
    return resolved, invalid


def export_summary_json(conn: sqlite3.Connection, output_path: Path) -> None:
    """Export summary statistics to JSON for fast frontend loading.

    Uses each model's default method for the summary (same as leaderboard).
    """
    from typing import Any

    cursor = conn.cursor()

    models_data: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    leaderboards: dict[str, list[dict[str, Any]]] = {}
    metric_configs = {
        "iou": {"column": "mean_iou", "order": "DESC", "legacy_key": "best_iou"},
        "coverage": {"column": "mean_coverage", "order": "DESC", "legacy_key": "best_coverage"},
        "mse": {"column": "mean_mse", "order": "ASC", "legacy_key": "best_mse"},
        "kl": {"column": "mean_kl", "order": "ASC", "legacy_key": "best_kl"},
        "emd": {"column": "mean_emd", "order": "ASC", "legacy_key": "best_emd"},
    }

    # Get distinct models
    cursor.execute("SELECT DISTINCT model FROM aggregate_metrics")
    models = [
        row[0]
        for row in cursor.fetchall()
        if row[0] in DEFAULT_METHOD and "_finetuned" not in row[0]
    ]

    # Build metric-specific leaderboards using each model's default method
    for metric_name, config in metric_configs.items():
        metric_entries: list[dict[str, Any]] = []
        score_column = config["column"]
        order_direction = config["order"]

        for model in models:
            default_method = DEFAULT_METHOD[model].value
            cursor.execute(
                f"""SELECT layer, {score_column}
                    FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY {score_column} {order_direction}, CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                    LIMIT 1""",
                (model, default_method),
            )
            row = cursor.fetchone()
            if row:
                metric_entries.append(
                    {
                        "model": display_model_name(model),
                        "metric": metric_name,
                        "best_layer": row[0],
                        "best_score": row[1],
                        "method_used": default_method,
                    }
                )

        metric_entries.sort(
            key=lambda entry: (entry["best_score"], entry["model"])
            if order_direction == "ASC"
            else (-entry["best_score"], entry["model"])
        )
        leaderboards[metric_name] = metric_entries

    leaderboard = [
        {
            "model": entry["model"],
            "best_iou": entry["best_score"],
            "method_used": entry["method_used"],
        }
        for entry in leaderboards.get("iou", [])
    ]

    # Get per-model summaries (at default method)
    for model in models:
        default_method = DEFAULT_METHOD[model].value
        metrics_summary: dict[str, Any] = {}

        for metric_name, config in metric_configs.items():
            score_column = config["column"]
            order_direction = config["order"]
            cursor.execute(
                f"""SELECT layer, {score_column} FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY {score_column} {order_direction}, CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                    LIMIT 1""",
                (model, default_method),
            )
            best_row = cursor.fetchone()
            best_layer = best_row[0] if best_row else "layer11"
            best_score = best_row[1] if best_row else 0

            cursor.execute(
                f"""SELECT layer, {score_column} FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                (model, default_method),
            )
            layer_progression = {row[0]: row[1] for row in cursor.fetchall()}
            metrics_summary[metric_name] = {
                "best_layer": best_layer,
                "best_score": best_score,
                "method_used": default_method,
                "layer_progression": layer_progression,
            }

        iou_summary = metrics_summary["iou"]
        models_data[display_model_name(model)] = {
            "best_layer": iou_summary["best_layer"],
            "best_iou": iou_summary["best_score"],
            "method_used": default_method,
            "layer_progression": iou_summary["layer_progression"],
            "metrics": metrics_summary,
        }

    summary = {
        "ranking_mode": "default_method",
        "models": models_data,
        "leaderboard": leaderboard,
        "leaderboards": leaderboards,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported summary to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate metrics cache")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process (or 'all')",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to process (default: all 12)",
    )
    parser.add_argument(
        "--attention-cache",
        type=Path,
        default=CACHE_PATH / "attention_viz.h5",
        help="Path to attention cache HDF5",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=CACHE_PATH / "metrics.db",
        help="Path to metrics SQLite database",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=DEFAULT_PERCENTILES,
        help="Percentile thresholds to compute",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Specific methods to process (e.g., cls rollout). Default: all for each model.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing database entries",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Compute metrics for fine-tuned cache keys ({model}_finetuned_{strategy})",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all"],
        help=(
            "Fine-tuning strategies to process in --finetuned mode. "
            "Defaults to all canonical strategies."
        ),
    )
    parser.add_argument(
        "--per-head",
        action="store_true",
        help="Also compute Q3 per-head metrics from per-head attention cache variants.",
    )
    args = parser.parse_args()

    # Validate
    if not args.attention_cache.exists():
        print(f"Error: Attention cache not found: {args.attention_cache}")
        print("Run generate_attention_cache.py first.")
        return 1

    models_to_process, invalid_models, non_finetunable_models = resolve_models_to_process(
        args.models,
        finetuned=args.finetuned,
    )

    if invalid_models:
        print(f"Warning: Unknown models ignored: {invalid_models}")
        print(f"Available: {list(MODELS.keys())}")

    if non_finetunable_models:
        print(f"Warning: Non-fine-tunable models ignored: {non_finetunable_models}")

    if not models_to_process:
        print("No valid models specified")
        return 1

    # Setup
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    attention_cache = AttentionCache(args.attention_cache)
    print(f"Attention cache: {args.attention_cache}")

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_database(args.db_path)
    print(f"Database: {args.db_path}")

    print(f"Percentiles: {args.percentiles}")
    print(f"Mode: {'FINE-TUNED' if args.finetuned else 'FROZEN'}")
    resolved_strategies: list[str] = []
    if args.finetuned:
        resolved_strategies, invalid_strategies = resolve_finetuned_strategies(args.strategies)
        if invalid_strategies:
            print(f"Warning: Unknown strategies ignored: {invalid_strategies}")
        if not resolved_strategies:
            print("No valid fine-tuning strategies specified")
            return 1
        print(f"Strategies: {resolved_strategies}")

    # Process each model
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for model_name in models_to_process:
        if args.finetuned:
            for strategy_id in resolved_strategies:
                storage_model_key = get_finetuned_cache_key(model_name, strategy_id)
                cache_model_keys = [storage_model_key]
                if strategy_id == "full":
                    cache_model_keys.append(f"{model_name}{_FINETUNED_SUFFIX}")

                stats = compute_metrics_for_model(
                    base_model_name=model_name,
                    dataset=dataset,
                    attention_cache=attention_cache,
                    conn=conn,
                    percentiles=args.percentiles,
                    layers=args.layers,
                    methods=args.methods,
                    skip_existing=not args.no_skip,
                    storage_model_key=storage_model_key,
                    cache_model_keys=cache_model_keys,
                )

                if args.per_head:
                    head_stats = compute_per_head_metrics_for_model(
                        base_model_name=model_name,
                        dataset=dataset,
                        attention_cache=attention_cache,
                        conn=conn,
                        percentiles=args.percentiles,
                        layers=args.layers,
                        methods=args.methods,
                        skip_existing=not args.no_skip,
                        storage_model_key=storage_model_key,
                        cache_model_keys=cache_model_keys,
                    )
                    for key in total_stats:
                        total_stats[key] += head_stats[key]
                    print(f"{storage_model_key} per-head complete: {head_stats}")

                for key in total_stats:
                    total_stats[key] += stats[key]

                print(f"{storage_model_key} complete: {stats}")
        else:
            stats = compute_metrics_for_model(
                base_model_name=model_name,
                dataset=dataset,
                attention_cache=attention_cache,
                conn=conn,
                percentiles=args.percentiles,
                layers=args.layers,
                methods=args.methods,
                skip_existing=not args.no_skip,
                storage_model_key=model_name,
            )

            if args.per_head:
                head_stats = compute_per_head_metrics_for_model(
                    base_model_name=model_name,
                    dataset=dataset,
                    attention_cache=attention_cache,
                    conn=conn,
                    percentiles=args.percentiles,
                    layers=args.layers,
                    methods=args.methods,
                    skip_existing=not args.no_skip,
                    storage_model_key=model_name,
                )
                for key in total_stats:
                    total_stats[key] += head_stats[key]
                print(f"{model_name} per-head complete: {head_stats}")

            for key in total_stats:
                total_stats[key] += stats[key]

            print(f"{model_name} complete: {stats}")

    # Export summary JSON
    export_summary_json(conn, args.db_path.parent / "metrics_summary.json")

    conn.close()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
