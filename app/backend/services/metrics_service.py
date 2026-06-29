"""Service for querying pre-computed metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from app.backend.config import (
    AVAILABLE_MODELS,
    METRICS_DB_PATH,
    METRICS_SUMMARY_PATH,
    MODEL_METHODS,
    MODEL_NUM_HEADS,
    PER_HEAD_METHODS,
    display_model_name,
    get_current_q2_results_path,
    get_model_num_layers,
    resolve_model_name,
)
from app.backend.services.image_service import image_service
from app.backend.validators import model_supports_method, resolve_default_method
from ssl_attention.evaluation.fine_tuning_artifacts import normalize_q2_analysis_payload

if TYPE_CHECKING:
    from collections.abc import Generator


AnalysisMetricName = Literal["iou", "coverage", "mse", "kl", "emd"]
MetricName = Literal["iou", "coverage", "mse", "kl", "emd"]
RankingMode = Literal["default_method", "best_available"]
Q3Variant = Literal["frozen", "linear_probe", "lora", "full"]

BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE = 90

IMAGE_DETAIL_METRICS = (
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
)
IMAGE_DETAIL_METRICS_BY_KEY = {
    cast(AnalysisMetricName, metric["key"]): metric for metric in IMAGE_DETAIL_METRICS
}


class MetricsService:
    """Service for querying pre-computed metrics from SQLite."""

    _instance: MetricsService | None = None

    def __new__(cls) -> MetricsService:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def db_path(self) -> Path:
        """Path to metrics database."""
        return METRICS_DB_PATH

    @property
    def db_exists(self) -> bool:
        """Check if metrics database exists."""
        return self.db_path.exists()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection context manager."""
        if not self.db_exists:
            raise FileNotFoundError(f"Metrics database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_image_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict | None:
        """Get metrics for a specific image/model/layer combination.

        Returns:
            Dict with iou, coverage, mse, kl, emd, attention_area, annotation_area or None.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            kl_select = "kl" if self._table_has_column(conn, "image_metrics", "kl") else "NULL AS kl"
            emd_select = "emd" if self._table_has_column(conn, "image_metrics", "emd") else "NULL AS emd"
            cursor.execute(
                f"""SELECT iou, coverage, mse, {kl_select}, {emd_select}, attention_area, annotation_area
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (image_id, db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                mse = row["mse"]
                if mse is None:
                    mse = self._compute_image_mse_from_cache(
                        image_id=image_id,
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                kl = row["kl"]
                if kl is None:
                    kl = self._compute_image_kl_from_cache(
                        image_id=image_id,
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                emd = row["emd"]
                if emd is None:
                    emd = self._compute_image_emd_from_cache(
                        image_id=image_id,
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                if mse is None or kl is None or emd is None:
                    return None

                return {
                    "image_id": image_id,
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "iou": row["iou"],
                    "coverage": row["coverage"],
                    "mse": mse,
                    "kl": kl,
                    "emd": emd,
                    "attention_area": row["attention_area"],
                    "annotation_area": row["annotation_area"],
                }
            return None

    def get_image_layer_progression(
        self,
        image_id: str,
        model: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get union-of-bboxes metric progression across all layers for one image."""
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        layer_points = self._initialize_layer_points(model)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            kl_select = "kl" if self._table_has_column(conn, "image_metrics", "kl") else "NULL AS kl"
            emd_select = "emd" if self._table_has_column(conn, "image_metrics", "emd") else "NULL AS emd"
            cursor.execute(
                f"""SELECT layer, iou, coverage, mse, {kl_select}, {emd_select}
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND method = ? AND percentile = ?
                   ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                (image_id, db_model, resolved_method, percentile),
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        has_values = False
        for row in rows:
            layer_key = row["layer"]
            if layer_key not in layer_points:
                continue

            mse = row["mse"]
            if mse is None:
                mse = self._compute_image_mse_from_cache(
                    image_id=image_id,
                    model=model,
                    layer=layer_key,
                    method=resolved_method,
                )
            kl = row["kl"]
            if kl is None:
                kl = self._compute_image_kl_from_cache(
                    image_id=image_id,
                    model=model,
                    layer=layer_key,
                    method=resolved_method,
                )
            emd = row["emd"]
            if emd is None:
                emd = self._compute_image_emd_from_cache(
                    image_id=image_id,
                    model=model,
                    layer=layer_key,
                    method=resolved_method,
                )

            values = layer_points[layer_key]["values"]
            values["iou"] = row["iou"]
            values["coverage"] = row["coverage"]
            values["mse"] = mse
            values["kl"] = kl
            values["emd"] = emd
            has_values = has_values or any(value is not None for value in values.values())

        if not has_values:
            return None

        return self._build_image_layer_progression_response(
            image_id=image_id,
            model=model,
            method=resolved_method,
            percentile=percentile,
            selection={
                "mode": "union",
                "bbox_index": None,
                "bbox_label": None,
            },
            layer_points=layer_points,
        )

    def get_bbox_layer_progression(
        self,
        image_id: str,
        model: str,
        bbox_index: int,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get bbox-specific metric progression across all layers for one image."""
        annotation = self._get_annotation(image_id)
        if annotation is None:
            return None

        resolved_method = method if method else resolve_default_method(model)
        bbox_label = self._get_bbox_label(annotation, bbox_index)
        layer_points = self._initialize_layer_points(model)
        has_values = False

        for layer_index in range(get_model_num_layers(model)):
            layer_key = f"layer{layer_index}"
            metrics = self._compute_bbox_metrics(
                image_id=image_id,
                model=model,
                layer=layer_key,
                bbox_index=bbox_index,
                percentile=percentile,
                method=resolved_method,
                annotation=annotation,
            )
            if metrics is None:
                continue

            values = layer_points[layer_key]["values"]
            values["iou"] = metrics["iou"]
            values["coverage"] = metrics["coverage"]
            values["mse"] = metrics["mse"]
            values["kl"] = metrics["kl"]
            values["emd"] = metrics["emd"]
            has_values = True

        if not has_values:
            return None

        return self._build_image_layer_progression_response(
            image_id=image_id,
            model=model,
            method=resolved_method,
            percentile=percentile,
            selection={
                "mode": "bbox",
                "bbox_index": bbox_index,
                "bbox_label": bbox_label,
            },
            layer_points=layer_points,
        )

    def get_bbox_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        bbox_index: int,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get bbox-specific metrics for a single layer."""
        annotation = self._get_annotation(image_id)
        if annotation is None:
            return None

        return self._compute_bbox_metrics(
            image_id=image_id,
            model=model,
            layer=layer,
            bbox_index=bbox_index,
            percentile=percentile,
            method=method,
            annotation=annotation,
        )

    def get_bbox_label(self, image_id: str, bbox_index: int) -> str:
        """Get the display label for a bbox on an image."""
        annotation = self._get_annotation(image_id)
        if annotation is None:
            raise ValueError(f"Annotation not found for {image_id}")

        return self._get_bbox_label(annotation, bbox_index)

    def get_aggregate_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict | None:
        """Get aggregate metrics for a model/layer.

        Returns:
            Dict with mean_iou, std_iou, median_iou, mean_coverage, mean_mse,
            std_mse, median_mse, mean_kl, std_kl, median_kl, mean_emd,
            std_emd, median_emd, num_images.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            has_kl_aggregate = all(
                self._table_has_column(conn, "aggregate_metrics", column_name)
                for column_name in ("mean_kl", "std_kl", "median_kl")
            )
            kl_select = (
                "mean_kl, std_kl, median_kl"
                if has_kl_aggregate
                else "NULL AS mean_kl, NULL AS std_kl, NULL AS median_kl"
            )
            has_emd_aggregate = all(
                self._table_has_column(conn, "aggregate_metrics", column_name)
                for column_name in ("mean_emd", "std_emd", "median_emd")
            )
            emd_select = (
                "mean_emd, std_emd, median_emd"
                if has_emd_aggregate
                else "NULL AS mean_emd, NULL AS std_emd, NULL AS median_emd"
            )
            cursor.execute(
                f"""SELECT mean_iou, std_iou, median_iou, mean_coverage,
                          mean_mse, std_mse, median_mse,
                          {kl_select},
                          {emd_select},
                          num_images
                   FROM aggregate_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                mean_mse = row["mean_mse"]
                std_mse = row["std_mse"]
                median_mse = row["median_mse"]
                if mean_mse is None or std_mse is None or median_mse is None:
                    mean_mse, std_mse, median_mse = self._compute_mse_aggregate_from_images(
                        model=model,
                        layer=layer,
                        percentile=percentile,
                        method=resolved_method,
                    )
                mean_kl = row["mean_kl"]
                std_kl = row["std_kl"]
                median_kl = row["median_kl"]
                if mean_kl is None or std_kl is None or median_kl is None:
                    mean_kl, std_kl, median_kl = self._compute_kl_aggregate_from_images(
                        model=model,
                        layer=layer,
                        percentile=percentile,
                        method=resolved_method,
                    )
                mean_emd = row["mean_emd"]
                std_emd = row["std_emd"]
                median_emd = row["median_emd"]
                if mean_emd is None or std_emd is None or median_emd is None:
                    mean_emd, std_emd, median_emd = self._compute_emd_aggregate_from_images(
                        model=model,
                        layer=layer,
                        percentile=percentile,
                        method=resolved_method,
                    )

                return {
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "mean_iou": row["mean_iou"],
                    "std_iou": row["std_iou"],
                    "median_iou": row["median_iou"],
                    "mean_coverage": row["mean_coverage"],
                    "mean_mse": mean_mse,
                    "std_mse": std_mse,
                    "median_mse": median_mse,
                    "mean_kl": mean_kl,
                    "std_kl": std_kl,
                    "median_kl": median_kl,
                    "mean_emd": mean_emd,
                    "std_emd": std_emd,
                    "median_emd": median_emd,
                    "num_images": row["num_images"],
                }
            return None

    def _metric_sql_config(self, metric: MetricName) -> tuple[str, str, str]:
        """Resolve aggregate score column, SQL sort direction, and selector."""
        if metric == "coverage":
            return "mean_coverage", "DESC", "MAX"
        if metric == "mse":
            return "mean_mse", "ASC", "MIN"
        if metric == "kl":
            return "mean_kl", "ASC", "MIN"
        if metric == "emd":
            return "mean_emd", "ASC", "MIN"
        return "mean_iou", "DESC", "MAX"

    def _aggregate_metric_value(self, aggregate_metrics: dict[str, Any], metric: MetricName) -> float | None:
        """Extract the aggregate score for the selected metric."""
        if metric == "iou":
            return aggregate_metrics.get("mean_iou")
        if metric == "coverage":
            return aggregate_metrics.get("mean_coverage")
        return aggregate_metrics.get(f"mean_{metric}")

    def _aggregate_metric_column_available(self, conn: sqlite3.Connection, metric: MetricName) -> bool:
        """Return whether the aggregate table already stores the selected metric."""
        if metric in {"iou", "coverage"}:
            return True

        score_column, _, _ = self._metric_sql_config(metric)
        return self._table_has_column(conn, "aggregate_metrics", score_column)

    def _metric_prefers_lower(self, metric: MetricName) -> bool:
        """Return whether the selected metric is optimized by a lower score."""
        return metric in {"mse", "kl", "emd"}

    def _legacy_aggregate_rows(
        self,
        model: str,
        percentile: int,
        method: str,
        metric: MetricName,
    ) -> list[tuple[str, float]]:
        """Recompute aggregate metric rows when legacy DBs lack stored columns."""
        db_model = resolve_model_name(model)
        rows: list[tuple[str, float]] = []
        for layer_idx in range(get_model_num_layers(db_model)):
            layer_key = f"layer{layer_idx}"
            aggregate_metrics = self.get_aggregate_metrics(
                model=model,
                layer=layer_key,
                percentile=percentile,
                method=method,
            )
            if not aggregate_metrics:
                continue

            score = self._aggregate_metric_value(aggregate_metrics, metric)
            if score is None:
                continue
            rows.append((layer_key, score))

        return rows

    def _best_layer_score_from_rows(
        self,
        layer_scores: list[tuple[str, float]],
        metric: MetricName,
    ) -> tuple[str, float]:
        """Pick the best layer/score pair for a model+method candidate."""
        if self._metric_prefers_lower(metric):
            return min(layer_scores, key=lambda row: (row[1], int(row[0][5:])))
        return max(layer_scores, key=lambda row: (row[1], -int(row[0][5:])))

    def _best_method_entry_for_model(
        self,
        conn: sqlite3.Connection,
        model: str,
        percentile: int,
        metric: MetricName,
        ranking_mode: RankingMode,
        method: str | None,
    ) -> dict[str, Any] | None:
        """Resolve the leaderboard row for one model under the requested ranking semantics."""
        if method is not None:
            candidate_methods = [method] if model_supports_method(model, method) else []
        elif ranking_mode == "best_available":
            candidate_methods = [candidate.value for candidate in MODEL_METHODS.get(model, [])]
        else:
            candidate_methods = [resolve_default_method(model)]

        if not candidate_methods:
            return None

        db_model = resolve_model_name(model)
        score_column, order_direction, _ = self._metric_sql_config(metric)
        has_aggregate_column = self._aggregate_metric_column_available(conn, metric)
        default_method = resolve_default_method(model)
        candidate_entries: list[dict[str, Any]] = []

        for selected_method in candidate_methods:
            if has_aggregate_column:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT layer, {score_column} AS score
                       FROM aggregate_metrics
                       WHERE model = ? AND method = ? AND percentile = ?
                         AND {score_column} IS NOT NULL
                       ORDER BY {score_column} {order_direction},
                                CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                       LIMIT 1""",
                    (db_model, selected_method, percentile),
                )
                row = cursor.fetchone()
                if row is None:
                    continue
                best_layer = row["layer"]
                best_score = row["score"]
            else:
                layer_scores = self._legacy_aggregate_rows(
                    model=model,
                    percentile=percentile,
                    method=selected_method,
                    metric=metric,
                )
                if not layer_scores:
                    continue
                best_layer, best_score = self._best_layer_score_from_rows(layer_scores, metric)

            candidate_entries.append(
                {
                    "model": display_model_name(db_model),
                    "metric": metric,
                    "score": best_score,
                    "best_layer": best_layer,
                    "method_used": selected_method,
                }
            )

        if not candidate_entries:
            return None

        def candidate_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
            primary_score: float = entry["score"]
            if not self._metric_prefers_lower(metric):
                primary_score = -primary_score
            return (
                primary_score,
                0 if entry["method_used"] == default_method else 1,
                int(entry["best_layer"][5:]),
                entry["method_used"],
            )

        return min(candidate_entries, key=candidate_sort_key)

    def _augment_summary_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Backfill explicit ranking semantics for legacy summary JSON files."""
        data.setdefault("ranking_mode", "default_method")

        models = data.get("models")
        if isinstance(models, dict):
            for model_name, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue
                default_method = resolve_default_method(model_name)
                model_data.setdefault("method_used", default_method)
                metrics = model_data.get("metrics")
                if isinstance(metrics, dict):
                    for metric_data in metrics.values():
                        if isinstance(metric_data, dict):
                            metric_data.setdefault("method_used", default_method)

        leaderboard = data.get("leaderboard")
        if isinstance(leaderboard, list):
            for entry in leaderboard:
                if isinstance(entry, dict) and "model" in entry:
                    entry.setdefault("method_used", resolve_default_method(entry["model"]))

        leaderboards = data.get("leaderboards")
        if isinstance(leaderboards, dict):
            for metric_entries in leaderboards.values():
                if not isinstance(metric_entries, list):
                    continue
                for entry in metric_entries:
                    if isinstance(entry, dict) and "model" in entry:
                        entry.setdefault("method_used", resolve_default_method(entry["model"]))

        return data

    def get_leaderboard(
        self,
        percentile: int = 90,
        metric: MetricName = "iou",
        method: str | None = None,
        ranking_mode: RankingMode = "default_method",
    ) -> list[dict]:
        """Get model leaderboard ranked by best score for the selected metric.

        Uses the selected ranking mode unless an explicit shared method filter is provided.

        Returns:
            List of dicts with rank, model, metric, score, best_layer, method_used.
        """
        with self.get_connection() as conn:
            rows = [
                entry
                for model_name in AVAILABLE_MODELS
                if (
                    entry := self._best_method_entry_for_model(
                        conn=conn,
                        model=model_name,
                        percentile=percentile,
                        metric=metric,
                        ranking_mode=ranking_mode,
                        method=method,
                    )
                )
                is not None
            ]

        ordered_rows = sorted(
            rows,
            key=lambda row: (row["score"], row["model"])
            if self._metric_prefers_lower(metric)
            else (-row["score"], row["model"]),
        )
        return [
            {
                "rank": index + 1,
                **row,
            }
            for index, row in enumerate(ordered_rows)
        ]

    def get_layer_progression(
        self,
        model: str,
        percentile: int = 90,
        method: str | None = None,
        metric: MetricName = "iou",
    ) -> dict:
        """Get metric progression across all layers.

        Returns:
            Dict with model, metric, percentile, layers, scores, best_layer, best_score.
        """
        score_column, _, _ = self._metric_sql_config(metric)
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            if not self._aggregate_metric_column_available(conn, metric):
                legacy_rows = self._legacy_aggregate_rows(
                    model=model,
                    percentile=percentile,
                    method=resolved_method,
                    metric=metric,
                )
                layers = [layer for layer, _ in legacy_rows]
                scores = [score for _, score in legacy_rows]
            else:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT layer, {score_column} AS score FROM aggregate_metrics
                       WHERE model = ? AND method = ? AND percentile = ?
                         AND {score_column} IS NOT NULL
                       ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                    (db_model, resolved_method, percentile),
                )
                db_rows = cursor.fetchall()

                layers = [row["layer"] for row in db_rows]
                scores = [row["score"] for row in db_rows]

            best_idx = 0
            if scores:
                comparator = min if metric in {"mse", "kl", "emd"} else max
                best_idx = scores.index(comparator(scores))
            best_layer = layers[best_idx] if layers else "layer11"
            best_score = scores[best_idx] if scores else 0.0

            return {
                "model": model,  # Return original name for display
                "metric": metric,
                "percentile": percentile,
                "method": resolved_method,
                "layers": layers,
                "scores": scores,
                "best_layer": best_layer,
                "best_score": best_score,
            }

    def get_style_breakdown(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        metric: AnalysisMetricName = "iou",
        method: str | None = None,
    ) -> dict:
        """Get metric breakdown by architectural style.

        Returns:
            Dict with model, layer, metric, percentile, styles, and style_counts.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = (
            percentile
            if metric_config["percentile_dependent"]
            else BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE
        )
        score_order = "DESC" if metric_config["direction"] == "higher" else "ASC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            score_column = "mean_score" if self._table_has_column(conn, "style_metrics", "mean_score") else "mean_iou"
            cursor.execute(
                f"""SELECT style_name, {score_column} AS mean_score, num_images FROM style_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND metric = ? AND percentile = ?
                   ORDER BY {score_column} {score_order}""",
                (db_model, layer, resolved_method, metric, query_percentile),
            )

            styles = {}
            counts = {}
            for row in cursor.fetchall():
                styles[row["style_name"]] = row["mean_score"]
                counts[row["style_name"]] = row["num_images"]

            return {
                "model": model,  # Return original name for display
                "layer": layer,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "method": resolved_method,
                "styles": styles,
                "style_counts": counts,
            }

    def get_all_image_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> list[dict]:
        """Get metrics for all images for a model/layer.

        Returns:
            List of dicts with image_id, iou, coverage, mse, kl, emd.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            kl_select = "kl" if self._table_has_column(conn, "image_metrics", "kl") else "NULL AS kl"
            emd_select = "emd" if self._table_has_column(conn, "image_metrics", "emd") else "NULL AS emd"
            cursor.execute(
                f"""SELECT image_id, iou, coverage, mse, {kl_select}, {emd_select}, attention_area, annotation_area
                   FROM image_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?
                   ORDER BY iou DESC""",
                (db_model, layer, resolved_method, percentile),
            )
            results = []
            for row in cursor.fetchall():
                mse = row["mse"]
                if mse is None:
                    mse = self._compute_image_mse_from_cache(
                        image_id=row["image_id"],
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                kl = row["kl"]
                if kl is None:
                    kl = self._compute_image_kl_from_cache(
                        image_id=row["image_id"],
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                emd = row["emd"]
                if emd is None:
                    emd = self._compute_image_emd_from_cache(
                        image_id=row["image_id"],
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                if mse is None or kl is None or emd is None:
                    continue

                results.append(
                    {
                        "image_id": row["image_id"],
                        "iou": row["iou"],
                        "coverage": row["coverage"],
                        "mse": mse,
                        "kl": kl,
                        "emd": emd,
                        "attention_area": row["attention_area"],
                        "annotation_area": row["annotation_area"],
                    }
                )

            return results

    def get_summary(self) -> dict[str, Any] | None:
        """Load pre-computed summary from JSON file."""
        if not METRICS_SUMMARY_PATH.exists():
            return None

        with open(METRICS_SUMMARY_PATH, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return self._augment_summary_metadata(data)

    def get_q2_summary(
        self,
        metric: AnalysisMetricName = "iou",
        percentile: int | None = None,
        model: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any] | None:
        """Load metric-generic Q2 results with optional filters."""
        q2_results_path = get_current_q2_results_path()
        if not q2_results_path.exists():
            return None

        with open(q2_results_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        data = normalize_q2_analysis_payload(data, drop_legacy=True)

        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        resolved_model = resolve_model_name(model) if model else None
        selected_percentile = percentile if metric_config["percentile_dependent"] else None

        filtered_rows = [
            row
            for row in data.get("rows", [])
            if row.get("metric") == metric
            and (resolved_model is None or row.get("model_name") == resolved_model)
            and (strategy is None or row.get("strategy_id") == strategy)
            and (
                not metric_config["percentile_dependent"]
                or selected_percentile is None
                or row.get("percentile") == selected_percentile
            )
        ]

        filtered_strategy_comparisons = [
            row
            for row in data.get("strategy_comparisons", [])
            if row.get("metric") == metric
            and (resolved_model is None or row.get("model_name") == resolved_model)
            and (strategy is None or row.get("strategy_a") == strategy or row.get("strategy_b") == strategy)
            and (
                not metric_config["percentile_dependent"]
                or selected_percentile is None
                or row.get("percentile") == selected_percentile
            )
        ]

        display_rows = [
            {
                **row,
                "model_name": display_model_name(str(row.get("model_name", ""))),
            }
            for row in filtered_rows
        ]
        display_strategy_comparisons = [
            {
                **row,
                "model_name": display_model_name(str(row.get("model_name", ""))),
            }
            for row in filtered_strategy_comparisons
        ]

        return {
            "metric": metric,
            "label": metric_config["label"],
            "direction": metric_config["direction"],
            "percentile_dependent": metric_config["percentile_dependent"],
            "selected_percentile": selected_percentile,
            "experiment_id": data.get("experiment_id"),
            "split_id": data.get("split_id"),
            "analysis_git_commit_sha": data.get("analysis_git_commit_sha"),
            "analyzed_layer": data.get("analyzed_layer", get_model_num_layers("dinov2") - 1),
            "evaluation_image_count": data.get("evaluation_image_count"),
            "checkpoint_selection_rule": data.get("checkpoint_selection_rule"),
            "result_set_scope": data.get("result_set_scope"),
            "timestamp": data.get("timestamp"),
            "rows": display_rows,
            "strategy_comparisons": display_strategy_comparisons,
        }

    def get_q2_image_deltas(
        self,
        model: str,
        strategy: Literal["linear_probe", "lora", "full"],
        percentile: int = 90,
        top_k: int = 12,
    ) -> dict[str, Any] | None:
        """Load per-image IoU deltas for one Q2 model/strategy/percentile slice."""
        q2_results_path = get_current_q2_results_path().with_name("q2_delta_iou_analysis.json")
        if not q2_results_path.exists():
            return None

        with open(q2_results_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        resolved_model = resolve_model_name(model)
        bucket: dict[str, Any] | None = None
        # Newer artifact shape: flat rows with per_image_deltas.
        rows = data.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (
                    row.get("metric") == "iou"
                    and row.get("model_name") == resolved_model
                    and row.get("strategy_id") == strategy
                    and row.get("percentile") == percentile
                ):
                    bucket = cast(dict[str, Any], row)
                    break

        # Legacy artifact shape: models -> strategy -> percentile buckets.
        if bucket is None:
            model_data = cast(dict[str, Any], data.get("models", {}).get(resolved_model))
            if not model_data:
                return None

            strategy_data = cast(dict[str, Any], model_data.get(strategy))
            if not strategy_data:
                return None

            bucket = cast(dict[str, Any], strategy_data.get(str(percentile)))
            if not bucket:
                return None

        per_image_deltas = cast(dict[str, float], bucket.get("per_image_deltas", {}))
        if not per_image_deltas:
            return None

        sorted_items = sorted(per_image_deltas.items(), key=lambda item: item[1], reverse=True)
        top_positive = sorted_items[:top_k]
        top_negative = sorted(per_image_deltas.items(), key=lambda item: item[1])[:top_k]

        def _to_rows(items: list[tuple[str, float]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for image_id, delta in items:
                annotation = image_service.get_annotation(image_id)
                style_names = image_service.get_style_names(list(annotation.styles)) if annotation else []
                rows.append(
                    {
                        "image_id": image_id,
                        "delta_iou": delta,
                        "style_names": style_names,
                    }
                )
            return rows

        return {
            "model_name": display_model_name(resolved_model),
            "strategy_id": strategy,
            "percentile": percentile,
            "method": bucket.get("method"),
            "mean_delta_iou": bucket.get("mean_delta_iou", bucket.get("mean_delta")),
            "num_images": bucket.get("num_images"),
            "top_positive": _to_rows(top_positive),
            "top_negative": _to_rows(top_negative),
        }

    def get_feature_breakdown(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        metric: AnalysisMetricName = "iou",
        sort_by: str = "mean_score",
        min_count: int = 0,
        method: str | None = None,
    ) -> dict:
        """Get metric breakdown by architectural feature type.

        Args:
            model: Model name.
            layer: Layer key (e.g., "layer11").
            percentile: Percentile threshold.
            metric: Selected metric for the feature breakdown.
            sort_by: Field to sort by ("mean_score", "bbox_count", "feature_name").
            min_count: Minimum bbox count to include a feature.
            method: Attention method. None = model default.

        Returns:
            Dict with model, layer, percentile, features list, total_feature_types.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = (
            percentile
            if metric_config["percentile_dependent"]
            else BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE
        )
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Determine sort order
            order_clause = "mean_score DESC" if metric_config["direction"] == "higher" else "mean_score ASC"
            if sort_by == "bbox_count":
                order_clause = "bbox_count DESC"
            elif sort_by == "feature_name":
                order_clause = "feature_name ASC"
            elif sort_by == "feature_label":
                order_clause = "feature_label ASC"
            elif sort_by == "mean_iou":
                order_clause = "mean_score DESC"

            cursor.execute(
                f"""SELECT feature_label, feature_name, mean_score, std_score, bbox_count
                   FROM feature_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND metric = ? AND percentile = ? AND bbox_count >= ?
                   ORDER BY {order_clause}""",
                (db_model, layer, resolved_method, metric, query_percentile, min_count),
            )

            features = [
                {
                    "feature_label": row["feature_label"],
                    "feature_name": row["feature_name"],
                    "mean_score": row["mean_score"],
                    "std_score": row["std_score"],
                    "bbox_count": row["bbox_count"],
                }
                for row in cursor.fetchall()
            ]

            return {
                "model": model,  # Return original name for display
                "layer": layer,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "method": resolved_method,
                "features": features,
                "total_feature_types": len(features),
            }

    def get_head_ranking(
        self,
        model: str,
        layer: str,
        metric: AnalysisMetricName = "iou",
        percentile: int = 90,
        variant: Q3Variant = "frozen",
    ) -> dict[str, Any]:
        """Get metric-specific Q3 head ranking rows."""
        method, db_model, unsupported_reason = self._resolve_q3_model_context(model, variant)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = self._metric_query_percentile(metric, percentile)

        if unsupported_reason:
            return {
                "model": model,
                "variant": variant,
                "layer": layer,
                "method": method,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "supported": False,
                "reason": unsupported_reason,
                "heads": [],
            }

        with self.get_connection() as conn:
            if not self._table_exists(conn, "head_summary_metrics"):
                return {
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "method": method,
                    "metric": metric,
                    "direction": metric_config["direction"],
                    "percentile": percentile,
                    "supported": False,
                    "reason": "Q3 head metrics are not available. Run generate_metrics_cache.py --per-head first.",
                    "heads": [],
                }
            cursor = conn.cursor()
            cursor.execute(
                """SELECT head, mean_score, std_score, mean_rank, top1_count, top3_count, image_count
                   FROM head_summary_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND metric = ? AND percentile = ?
                   ORDER BY mean_rank ASC, head ASC""",
                (db_model, layer, method, metric, query_percentile),
            )
            heads = [
                {
                    "head": row["head"],
                    "mean_score": row["mean_score"],
                    "std_score": row["std_score"],
                    "mean_rank": row["mean_rank"],
                    "top1_count": row["top1_count"],
                    "top3_count": row["top3_count"],
                    "image_count": row["image_count"],
                }
                for row in cursor.fetchall()
            ]

        return {
            "model": model,
            "variant": variant,
            "layer": layer,
            "method": method,
            "metric": metric,
            "direction": metric_config["direction"],
            "percentile": percentile,
            "supported": True,
            "reason": None,
            "heads": heads,
        }

    def get_image_head_ranking(
        self,
        image_id: str,
        model: str,
        layer: str,
        metric: AnalysisMetricName = "iou",
        percentile: int = 90,
        variant: Q3Variant = "frozen",
        bbox_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Get metric-specific Q3 head ranking rows for one image."""
        method, db_model, unsupported_reason = self._resolve_q3_model_context(model, variant)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = self._metric_query_percentile(metric, percentile)
        selection: dict[str, str | int | None] = {
            "mode": "union",
            "bbox_index": None,
            "bbox_label": None,
        }
        num_heads = MODEL_NUM_HEADS.get(resolve_model_name(model), 0)

        if bbox_index is not None:
            annotation = self._get_annotation(image_id)
            if annotation is None:
                return None
            selection = {
                "mode": "bbox",
                "bbox_index": bbox_index,
                "bbox_label": self._get_bbox_label(annotation, bbox_index),
            }

        if unsupported_reason:
            return {
                "image_id": image_id,
                "model": model,
                "variant": variant,
                "layer": layer,
                "method": method,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "selection": selection,
                "supported": False,
                "reason": unsupported_reason,
                "heads": [],
            }

        if bbox_index is not None:
            heads: list[dict[str, Any]] = []
            for head in range(num_heads):
                metrics = self._compute_bbox_metrics(
                    image_id=image_id,
                    model=db_model,
                    layer=layer,
                    bbox_index=bbox_index,
                    percentile=percentile,
                    method=method,
                    annotation=annotation,
                    head=head,
                )
                if metrics is None:
                    continue

                score = metrics.get(metric)
                if score is None:
                    continue

                heads.append(
                    {
                        "head": head,
                        "score": score,
                    }
                )

            if not heads:
                return {
                    "image_id": image_id,
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "method": method,
                    "metric": metric,
                    "direction": metric_config["direction"],
                    "percentile": percentile,
                    "selection": selection,
                    "supported": False,
                    "reason": "Q3 per-head attention is not cached for this image selection. Run generate_attention_cache.py --per-head first.",
                    "heads": [],
                }

            heads.sort(
                key=lambda entry: (
                    -entry["score"] if metric_config["direction"] == "higher" else entry["score"],
                    entry["head"],
                )
            )

            return {
                "image_id": image_id,
                "model": model,
                "variant": variant,
                "layer": layer,
                "method": method,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "selection": selection,
                "supported": True,
                "reason": None,
                "heads": heads,
            }

        with self.get_connection() as conn:
            if not self._table_exists(conn, "head_image_metrics"):
                return {
                    "image_id": image_id,
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "method": method,
                    "metric": metric,
                    "direction": metric_config["direction"],
                    "percentile": percentile,
                    "selection": selection,
                    "supported": False,
                    "reason": "Q3 head metrics are not available. Run generate_metrics_cache.py --per-head first.",
                    "heads": [],
                }

            cursor = conn.cursor()
            order_direction = "DESC" if metric_config["direction"] == "higher" else "ASC"
            cursor.execute(
                f"""SELECT head, score
                    FROM head_image_metrics
                    WHERE image_id = ? AND model = ? AND layer = ? AND method = ? AND metric = ? AND percentile = ?
                    ORDER BY score {order_direction}, head ASC""",
                (image_id, db_model, layer, method, metric, query_percentile),
            )
            rows = cursor.fetchall()

        if not rows:
            return {
                "image_id": image_id,
                "model": model,
                "variant": variant,
                "layer": layer,
                "method": method,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "selection": selection,
                "supported": False,
                "reason": "No Q3 per-head image ranking rows are available for this image yet.",
                "heads": [],
            }

        return {
            "image_id": image_id,
            "model": model,
            "variant": variant,
            "layer": layer,
            "method": method,
            "metric": metric,
            "direction": metric_config["direction"],
            "percentile": percentile,
            "selection": selection,
            "supported": True,
            "reason": None,
            "heads": [
                {
                    "head": row["head"],
                    "score": row["score"],
                }
                for row in rows
            ],
        }

    def get_head_feature_matrix(
        self,
        model: str,
        layer: str,
        metric: AnalysisMetricName = "iou",
        percentile: int = 90,
        variant: Q3Variant = "frozen",
    ) -> dict[str, Any]:
        """Get the Q3 head-by-feature matrix for one metric."""
        method, db_model, unsupported_reason = self._resolve_q3_model_context(model, variant)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = self._metric_query_percentile(metric, percentile)
        num_heads = MODEL_NUM_HEADS.get(resolve_model_name(model), 0)

        if unsupported_reason:
            return {
                "model": model,
                "variant": variant,
                "layer": layer,
                "method": method,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "supported": False,
                "reason": unsupported_reason,
                "heads": list(range(num_heads)),
                "features": [],
                "total_feature_types": 0,
            }

        with self.get_connection() as conn:
            if not self._table_exists(conn, "head_feature_metrics"):
                return {
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "method": method,
                    "metric": metric,
                    "direction": metric_config["direction"],
                    "percentile": percentile,
                    "supported": False,
                    "reason": "Q3 head metrics are not available. Run generate_metrics_cache.py --per-head first.",
                    "heads": list(range(num_heads)),
                    "features": [],
                    "total_feature_types": 0,
                }
            cursor = conn.cursor()
            cursor.execute(
                """SELECT feature_label, feature_name, bbox_count, head, mean_score
                   FROM head_feature_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND metric = ? AND percentile = ?
                   ORDER BY feature_name ASC, head ASC""",
                (db_model, layer, method, metric, query_percentile),
            )
            rows = cursor.fetchall()

        matrix: dict[int, dict[str, Any]] = {}
        for row in rows:
            feature_label = row["feature_label"]
            entry = matrix.setdefault(
                feature_label,
                {
                    "feature_label": feature_label,
                    "feature_name": row["feature_name"],
                    "bbox_count": row["bbox_count"],
                    "scores": [None] * num_heads,
                },
            )
            head_idx = row["head"]
            if 0 <= head_idx < num_heads:
                entry["scores"][head_idx] = row["mean_score"]

        features = sorted(
            matrix.values(),
            key=lambda row: row["feature_name"].lower(),
        )

        return {
            "model": model,
            "variant": variant,
            "layer": layer,
            "method": method,
            "metric": metric,
            "direction": metric_config["direction"],
            "percentile": percentile,
            "supported": True,
            "reason": None,
            "heads": list(range(num_heads)),
            "features": features,
            "total_feature_types": len(features),
        }

    def get_head_exemplars(
        self,
        model: str,
        layer: str,
        head: int,
        metric: AnalysisMetricName = "iou",
        percentile: int = 90,
        variant: Q3Variant = "frozen",
        feature_label: int | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        """Return representative image candidates for a selected Q3 head."""
        method, db_model, unsupported_reason = self._resolve_q3_model_context(model, variant)
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        query_percentile = self._metric_query_percentile(metric, percentile)
        feature_name = image_service.get_feature_name(feature_label) if feature_label is not None else None

        if unsupported_reason:
            return {
                "model": model,
                "variant": variant,
                "layer": layer,
                "metric": metric,
                "direction": metric_config["direction"],
                "percentile": percentile,
                "head": head,
                "feature_label": feature_label,
                "feature_name": feature_name,
                "supported": False,
                "reason": unsupported_reason,
                "candidates": [],
            }

        order_direction = "DESC" if metric_config["direction"] == "higher" else "ASC"
        with self.get_connection() as conn:
            exemplar_table = "head_feature_image_metrics" if feature_label is not None else "head_image_metrics"
            if not self._table_exists(conn, exemplar_table):
                unavailable_reason = (
                    "Q3 head-feature exemplars are not available. Re-run generate_metrics_cache.py --per-head to populate per-image feature rows."
                    if feature_label is not None
                    else "Q3 head metrics are not available. Run generate_metrics_cache.py --per-head first."
                )
                return {
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "metric": metric,
                    "direction": metric_config["direction"],
                    "percentile": percentile,
                    "head": head,
                    "feature_label": feature_label,
                    "feature_name": feature_name,
                    "supported": False,
                    "reason": unavailable_reason,
                    "candidates": [],
                }
            cursor = conn.cursor()
            if feature_label is not None:
                cursor.execute(
                    f"""SELECT image_id, score, feature_name, default_bbox_index
                        FROM head_feature_image_metrics
                        WHERE model = ? AND layer = ? AND method = ? AND head = ? AND metric = ? AND percentile = ? AND feature_label = ?
                        ORDER BY score {order_direction}, image_id ASC""",
                    (db_model, layer, method, head, metric, query_percentile, feature_label),
                )
            else:
                cursor.execute(
                    f"""SELECT image_id, score
                        FROM head_image_metrics
                        WHERE model = ? AND layer = ? AND method = ? AND head = ? AND metric = ? AND percentile = ?
                        ORDER BY score {order_direction}, image_id ASC""",
                    (db_model, layer, method, head, metric, query_percentile),
                )
            rows = cursor.fetchall()

        if feature_label is not None and rows and not feature_name:
            feature_name = rows[0]["feature_name"]

        candidates: list[dict[str, Any]] = []
        for row in rows:
            image_id = row["image_id"]
            annotation = image_service.get_annotation(image_id)
            if annotation is None:
                continue

            matching_bbox_indices = [
                index
                for index, bbox in enumerate(annotation.bboxes)
                if feature_label is not None and bbox.label == feature_label
            ]
            if feature_label is not None and not matching_bbox_indices:
                continue

            candidates.append(
                {
                    "image_id": image_id,
                    "score": row["score"],
                    "thumbnail_url": f"/api/images/{image_id}/thumbnail",
                    "style_names": image_service.get_style_names(list(annotation.styles)),
                    "matching_bbox_indices": matching_bbox_indices,
                    "default_bbox_index": (
                        row["default_bbox_index"]
                        if feature_label is not None
                        else (matching_bbox_indices[0] if matching_bbox_indices else None)
                    ),
                }
            )
            if len(candidates) >= limit:
                break

        reason: str | None = None
        if feature_label is not None and not candidates:
            feature_descriptor = feature_name or f"feature {feature_label}"
            reason = f"No Q3 head-feature exemplar rows are available for {feature_descriptor} at this selection yet."

        return {
            "model": model,
            "variant": variant,
            "layer": layer,
            "metric": metric,
            "direction": metric_config["direction"],
            "percentile": percentile,
            "head": head,
            "feature_label": feature_label,
            "feature_name": feature_name,
            "supported": True,
            "reason": reason,
            "candidates": candidates,
        }

    def _compute_image_mse_from_cache(
        self,
        image_id: str,
        model: str,
        layer: str,
        method: str,
    ) -> float | None:
        """Compute MSE directly from cached attention for legacy DB rows."""
        return self._compute_image_continuous_metric_from_cache(
            metric_name="mse",
            image_id=image_id,
            model=model,
            layer=layer,
            method=method,
        )

    def _compute_image_kl_from_cache(
        self,
        image_id: str,
        model: str,
        layer: str,
        method: str,
    ) -> float | None:
        """Compute KL divergence directly from cached attention for legacy DB rows."""
        return self._compute_image_continuous_metric_from_cache(
            metric_name="kl",
            image_id=image_id,
            model=model,
            layer=layer,
            method=method,
        )

    def _compute_image_emd_from_cache(
        self,
        image_id: str,
        model: str,
        layer: str,
        method: str,
    ) -> float | None:
        """Compute EMD directly from cached attention for legacy DB rows."""
        return self._compute_image_continuous_metric_from_cache(
            metric_name="emd",
            image_id=image_id,
            model=model,
            layer=layer,
            method=method,
        )

    def _compute_image_continuous_metric_from_cache(
        self,
        metric_name: Literal["mse", "kl", "emd"],
        image_id: str,
        model: str,
        layer: str,
        method: str,
    ) -> float | None:
        """Compute a threshold-free continuous metric from cached attention."""
        from app.backend.services.attention_service import attention_service
        from app.backend.services.image_service import image_service
        from ssl_attention.metrics import compute_image_emd, compute_image_kl, compute_image_mse

        annotation = image_service.get_annotation(image_id)
        if annotation is None:
            return None

        cache_model = resolve_model_name(model)
        try:
            attention = attention_service.cache.load(cache_model, layer, image_id, variant=method)
        except KeyError:
            return None

        if attention.dim() == 1:
            grid_rows, grid_cols = attention_service.get_attention_grid(cache_model)
            attention = attention.reshape(grid_rows, grid_cols)

        if metric_name == "kl":
            return compute_image_kl(attention=attention, annotation=annotation)
        if metric_name == "emd":
            return compute_image_emd(attention=attention, annotation=annotation)
        return compute_image_mse(attention=attention, annotation=annotation)

    def _compute_bbox_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        bbox_index: int,
        percentile: int,
        method: str | None,
        annotation: Any,
        head: int | None = None,
    ) -> dict[str, Any] | None:
        """Compute bbox metrics from cached attention for a single layer."""
        from ssl_attention.metrics.continuous import (
            compute_emd,
            compute_kl_divergence,
            compute_mse,
            gaussian_bbox_heatmap,
        )
        from ssl_attention.metrics.iou import compute_coverage, compute_iou

        resolved_method = method if method else resolve_default_method(model)

        if not 0 <= bbox_index < len(annotation.bboxes):
            raise ValueError(
                f"bbox_index {bbox_index} out of range. "
                f"Image has {len(annotation.bboxes)} bboxes (0-{len(annotation.bboxes) - 1})."
            )

        attention_tensor = self._load_attention_tensor(
            model=model,
            layer=layer,
            image_id=image_id,
            method=resolved_method,
            head=head,
        )
        if attention_tensor is None:
            return None

        bbox = annotation.bboxes[bbox_index]
        height, width = attention_tensor.shape[-2:]
        bbox_mask = bbox.to_mask(height, width)
        bbox_heatmap = gaussian_bbox_heatmap(bbox, height, width, device=attention_tensor.device)
        iou, attention_area, annotation_area = compute_iou(attention_tensor, bbox_mask, percentile)
        coverage = compute_coverage(attention_tensor, bbox_mask)
        mse = compute_mse(attention_tensor, bbox_heatmap)
        kl = compute_kl_divergence(attention_tensor, bbox_heatmap)
        emd = compute_emd(attention_tensor, bbox_heatmap)

        return {
            "image_id": image_id,
            "model": model,
            "layer": layer,
            "percentile": percentile,
            "method": resolved_method,
            "iou": iou,
            "coverage": coverage,
            "mse": mse,
            "kl": kl,
            "emd": emd,
            "attention_area": attention_area,
            "annotation_area": annotation_area,
        }

    def _get_annotation(self, image_id: str) -> Any | None:
        """Load image annotation without introducing a module-level import cycle."""
        from app.backend.services.image_service import image_service

        return image_service.get_annotation(image_id)

    def _table_has_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> bool:
        """Check whether a SQLite table currently exposes a given column."""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return any(row[1] == column_name for row in cursor.fetchall())

    def _table_exists(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        """Check whether a SQLite table exists."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def _load_attention_tensor(
        self,
        model: str,
        layer: str,
        image_id: str,
        method: str,
        head: int | None = None,
    ) -> Any | None:
        """Load cached attention and normalize it to a 2D tensor."""
        from app.backend.services.attention_service import attention_service

        cache_model = resolve_model_name(model)
        cache_variant = attention_service.resolve_variant(method, head)
        try:
            attention = attention_service.cache.load(cache_model, layer, image_id, variant=cache_variant)
        except KeyError:
            return None

        if attention.dim() == 1:
            grid_rows, grid_cols = attention_service.get_attention_grid(cache_model)
            attention = attention.reshape(grid_rows, grid_cols)

        return attention

    def _get_bbox_label(self, annotation: Any, bbox_index: int) -> str:
        """Resolve a human-readable label for a bbox selection."""
        from app.backend.services.image_service import image_service

        if not 0 <= bbox_index < len(annotation.bboxes):
            raise ValueError(
                f"bbox_index {bbox_index} out of range. "
                f"Image has {len(annotation.bboxes)} bboxes (0-{len(annotation.bboxes) - 1})."
            )

        bbox = annotation.bboxes[bbox_index]
        return image_service.get_feature_name(bbox.label) or f"Feature {bbox.label}"

    def _initialize_layer_points(self, model: str) -> dict[str, dict[str, Any]]:
        """Create stable layer entries for the full model depth."""
        num_layers = get_model_num_layers(model)
        return {
            f"layer{layer_index}": {
                "layer": layer_index,
                "layer_key": f"layer{layer_index}",
                "values": self._empty_image_detail_metric_values(),
            }
            for layer_index in range(num_layers)
        }

    def _empty_image_detail_metric_values(self) -> dict[str, float | None]:
        """Build a blank values map for all image-detail metrics."""
        return {str(metric["key"]): None for metric in IMAGE_DETAIL_METRICS}

    def _build_image_layer_progression_response(
        self,
        image_id: str,
        model: str,
        method: str,
        percentile: int,
        selection: dict[str, Any],
        layer_points: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the API payload for image-detail layer progression."""
        ordered_layers = [layer_points[f"layer{layer_index}"] for layer_index in range(get_model_num_layers(model))]
        return {
            "image_id": image_id,
            "model": model,
            "method": method,
            "percentile": percentile,
            "selection": selection,
            "metrics": [dict(metric) for metric in IMAGE_DETAIL_METRICS],
            "layers": ordered_layers,
        }

    def _resolve_q3_model_context(
        self,
        model: str,
        variant: Q3Variant,
    ) -> tuple[str | None, str, str | None]:
        """Resolve the automatic Q3 method plus storage model key."""
        base_model = resolve_model_name(model)
        if MODEL_NUM_HEADS.get(base_model, 0) <= 0:
            return None, base_model, f"Q3 per-head analysis is not supported for model '{model}'."

        available_methods = [method.value for method in MODEL_METHODS.get(base_model, [])]
        supported_methods = [method for method in available_methods if method in PER_HEAD_METHODS]
        if not supported_methods:
            return None, base_model, f"Q3 per-head analysis is not supported for model '{model}'."

        method = supported_methods[0]
        if variant == "frozen":
            return method, base_model, None
        return method, f"{base_model}_finetuned_{variant}", None

    def _metric_query_percentile(
        self,
        metric: AnalysisMetricName,
        percentile: int,
    ) -> int:
        """Map threshold-free metrics to the shared reference percentile key."""
        metric_config = IMAGE_DETAIL_METRICS_BY_KEY[metric]
        if metric_config["percentile_dependent"]:
            return percentile
        return BREAKDOWN_THRESHOLD_FREE_REFERENCE_PERCENTILE

    def _compute_mse_aggregate_from_images(
        self,
        model: str,
        layer: str,
        percentile: int,
        method: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Derive aggregate MSE stats from per-image rows when the DB is not backfilled."""
        return self._compute_continuous_aggregate_from_images(
            model=model,
            layer=layer,
            percentile=percentile,
            method=method,
            value_key="mse",
        )

    def _compute_kl_aggregate_from_images(
        self,
        model: str,
        layer: str,
        percentile: int,
        method: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Derive aggregate KL stats from per-image rows when the DB is not backfilled."""
        return self._compute_continuous_aggregate_from_images(
            model=model,
            layer=layer,
            percentile=percentile,
            method=method,
            value_key="kl",
        )

    def _compute_emd_aggregate_from_images(
        self,
        model: str,
        layer: str,
        percentile: int,
        method: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Derive aggregate EMD stats from per-image rows when the DB is not backfilled."""
        return self._compute_continuous_aggregate_from_images(
            model=model,
            layer=layer,
            percentile=percentile,
            method=method,
            value_key="emd",
        )

    def _compute_continuous_aggregate_from_images(
        self,
        model: str,
        layer: str,
        percentile: int,
        method: str,
        value_key: Literal["mse", "kl", "emd"],
    ) -> tuple[float | None, float | None, float | None]:
        """Derive aggregate continuous-metric stats from per-image rows."""
        import torch

        image_metrics = self.get_all_image_metrics(
            model=model,
            layer=layer,
            percentile=percentile,
            method=method,
        )
        values = [row[value_key] for row in image_metrics if row[value_key] is not None]
        if not values:
            return None, None, None

        tensor_values = torch.tensor(values, dtype=torch.float32)
        return (
            tensor_values.mean().item(),
            tensor_values.std().item(),
            tensor_values.median().item(),
        )


# Global instance
metrics_service = MetricsService()
