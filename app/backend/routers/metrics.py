"""Metrics query endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

from app.backend.config import AVAILABLE_MODELS, get_model_num_layers, resolve_model_name
from app.backend.schemas import (
    FeatureBreakdownSchema,
    HeadExemplarResponse,
    HeadFeatureMatrixResponse,
    HeadRankingResponse,
    ImageHeadRankingResponse,
    ImageLayerProgressionSchema,
    IoUResultSchema,
    LayerProgressionSchema,
    LeaderboardEntry,
    Q2ImageDeltasResponse,
    Q2SummaryResponse,
    StyleBreakdownSchema,
)
from app.backend.services.image_service import image_service
from app.backend.services.metrics_service import metrics_service
from app.backend.validators import (
    resolve_ranking_mode_request,
    validate_attention_method,
    validate_layer_for_model,
    validate_method,
    validate_model,
)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/q2_summary", response_model=Q2SummaryResponse)
async def get_q2_summary(
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    percentile: Annotated[int | None, Query(ge=50, le=95)] = None,
    model: Annotated[str | None, Query()] = None,
    strategy: Annotated[str | None, Query(description="Fine-tuning strategy")] = None,
) -> Q2SummaryResponse:
    """Get a metric-generic strategy-aware Q2 analysis summary."""
    if model is not None:
        validate_model(model)

    data = metrics_service.get_q2_summary(
        metric=metric,
        percentile=percentile,
        model=model,
        strategy=strategy,
    )
    if not data:
        raise HTTPException(
            status_code=503,
            detail=(
                "Q2 summary not available. Run "
                "experiments/scripts/analyze_q2_metrics.py first."
            ),
        )
    return Q2SummaryResponse(**data)


@router.get("/q2_image_deltas", response_model=Q2ImageDeltasResponse)
async def get_q2_image_deltas(
    model: Annotated[str, Query()],
    strategy: Annotated[Literal["linear_probe", "lora", "full"], Query()],
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    top_k: Annotated[int, Query(ge=3, le=30)] = 12,
) -> Q2ImageDeltasResponse:
    """Get image-level IoU deltas for one Q2 model/strategy slice."""
    validate_model(model)
    data = metrics_service.get_q2_image_deltas(
        model=model,
        strategy=strategy,
        percentile=percentile,
        top_k=top_k,
    )
    if not data:
        raise HTTPException(
            status_code=404,
            detail="Q2 image deltas not available for this selection.",
        )
    return Q2ImageDeltasResponse(**data)


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "mse", "kl", "emd"], Query()] = "iou",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
    ranking_mode: Annotated[Literal["default_method", "best_available"] | None, Query()] = None,
) -> list[LeaderboardEntry]:
    """Get model rankings by best score for the selected metric.

    Returns models ranked by their best score at the given percentile.
    """
    resolved_method = validate_attention_method(method)
    resolved_ranking_mode = resolve_ranking_mode_request(resolved_method, ranking_mode)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available. Run generate_metrics_cache.py first.",
        )

    if resolved_method is not None:
        data = metrics_service.get_leaderboard(percentile, metric=metric, method=resolved_method)
    else:
        data = metrics_service.get_leaderboard(
            percentile,
            metric=metric,
            ranking_mode=resolved_ranking_mode or "default_method",
        )
    return [LeaderboardEntry(**entry) for entry in data]


@router.get("/summary")
async def get_summary() -> dict:
    """Get pre-computed metrics summary.

    Returns overall statistics including leaderboard and per-model best layers.
    """
    summary = metrics_service.get_summary()
    if not summary:
        raise HTTPException(
            status_code=503,
            detail="Metrics summary not available. Run generate_metrics_cache.py first.",
        )
    return summary


@router.get("/{image_id}", response_model=IoUResultSchema)
async def get_image_metrics(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> IoUResultSchema:
    """Get alignment metrics for a specific image.

    Returns IoU, coverage, continuous metrics, and area statistics.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile, method=resolved_method)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for {image_id} with {model}/{layer_key}",
        )

    return IoUResultSchema(**result)


@router.get("/{image_id}/progression", response_model=ImageLayerProgressionSchema)
async def get_image_layer_progression(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    bbox_index: Annotated[int | None, Query(ge=0)] = None,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> ImageLayerProgressionSchema:
    """Get extensible per-image metric progression across all layers."""
    validate_model(model)
    resolved_method = validate_method(model, method)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Annotation not found for {image_id}")

    try:
        if bbox_index is None:
            if not metrics_service.db_exists:
                raise HTTPException(
                    status_code=503,
                    detail="Metrics database not available.",
                )
            data = metrics_service.get_image_layer_progression(
                image_id=image_id,
                model=model,
                percentile=percentile,
                method=resolved_method,
            )
        else:
            data = metrics_service.get_bbox_layer_progression(
                image_id=image_id,
                model=model,
                bbox_index=bbox_index,
                percentile=percentile,
                method=resolved_method,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not data:
        detail = (
            f"No bbox progression found for {image_id} at bbox_index {bbox_index}."
            if bbox_index is not None
            else f"No image progression found for {image_id}."
        )
        raise HTTPException(status_code=404, detail=detail)

    return ImageLayerProgressionSchema(**data)


@router.get("/{image_id}/all_models")
async def get_image_metrics_all_models(
    image_id: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get metrics for an image across all models.

    Useful for model comparison on a single image.
    Note: Layer validation is done per-model in the loop since models have different layer counts.
    When method is specified, only models supporting that method are included.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    layer_key = f"layer{layer}"
    results = {}

    for model in AVAILABLE_MODELS:
        # Skip models that don't have this layer
        num_layers = get_model_num_layers(resolve_model_name(model))
        if layer >= num_layers:
            continue

        # Skip models that don't support the requested method
        try:
            resolved_method = validate_method(model, method)
        except HTTPException:
            continue

        result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile, method=resolved_method)
        if result:
            results[model] = result

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    return {
        "image_id": image_id,
        "layer": layer_key,
        "percentile": percentile,
        "models": results,
    }


@router.get("/{image_id}/bbox/{bbox_index}", response_model=IoUResultSchema)
async def get_bbox_metrics(
    image_id: str,
    bbox_index: int,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> IoUResultSchema:
    """Get alignment metrics for a specific bounding box.

    Computes IoU, coverage, and continuous metrics on-the-fly for a single
    bbox against the attention map (rather than the union of all bboxes).
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Annotation not found for {image_id}")

    try:
        result = metrics_service.get_bbox_metrics(
            image_id=image_id,
            model=model,
            layer=layer_key,
            bbox_index=bbox_index,
            percentile=percentile,
            method=resolved_method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Attention not cached for {model}/{layer_key}/{resolved_method}/{image_id}. Run precompute first.",
        )

    return IoUResultSchema(**result)


@router.get("/model/{model}/progression", response_model=LayerProgressionSchema)
async def get_layer_progression(
    model: str,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> LayerProgressionSchema:
    """Get metric progression across all layers for a model.

    Shows how attention alignment evolves through transformer layers.
    """
    validate_model(model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_layer_progression(model, percentile, method=resolved_method, metric=metric)
    return LayerProgressionSchema(**data)


@router.get("/model/{model}/style_breakdown", response_model=StyleBreakdownSchema)
async def get_style_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> StyleBreakdownSchema:
    """Get metric breakdown by architectural style.

    Shows how well the model attends to different architectural styles.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_style_breakdown(
        model,
        layer_key,
        percentile,
        metric=metric,
        method=resolved_method,
    )
    return StyleBreakdownSchema(**data)


@router.get("/model/{model}/feature_breakdown", response_model=FeatureBreakdownSchema)
async def get_feature_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    min_count: Annotated[int, Query(ge=0)] = 0,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    sort_by: Annotated[str, Query(enum=["mean_score", "mean_iou", "bbox_count", "feature_name", "feature_label"])] = "mean_score",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> FeatureBreakdownSchema:
    """Get metric breakdown by architectural feature type.

    Shows how well the model attends to different architectural features
    (e.g., windows, doors, arches) across all 106 feature types.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_feature_breakdown(
        model,
        layer_key,
        percentile,
        metric=metric,
        sort_by=sort_by,
        min_count=min_count,
        method=resolved_method,
    )
    return FeatureBreakdownSchema(**data)


@router.get("/model/{model}/head_ranking", response_model=HeadRankingResponse)
async def get_head_ranking(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    variant: Annotated[Literal["frozen", "linear_probe", "lora", "full"], Query()] = "frozen",
) -> HeadRankingResponse:
    """Get the Q3 per-head ranking summary for one model/layer/metric."""
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_head_ranking(
        model=model,
        layer=layer_key,
        percentile=percentile,
        metric=metric,
        variant=variant,
    )
    return HeadRankingResponse(**data)


@router.get("/{image_id}/head_ranking", response_model=ImageHeadRankingResponse)
async def get_image_head_ranking(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    variant: Annotated[Literal["frozen", "linear_probe", "lora", "full"], Query()] = "frozen",
    bbox_index: Annotated[int | None, Query(ge=0)] = None,
) -> ImageHeadRankingResponse:
    """Get the Q3 per-head ranking for one image and optional bbox selection."""
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    try:
        data = metrics_service.get_image_head_ranking(
            image_id=image_id,
            model=model,
            layer=layer_key,
            percentile=percentile,
            metric=metric,
            variant=variant,
            bbox_index=bbox_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image annotation not found for {image_id}.",
        )

    return ImageHeadRankingResponse(**data)


@router.get("/model/{model}/head_feature_matrix", response_model=HeadFeatureMatrixResponse)
async def get_head_feature_matrix(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    variant: Annotated[Literal["frozen", "linear_probe", "lora", "full"], Query()] = "frozen",
) -> HeadFeatureMatrixResponse:
    """Get the Q3 head-by-feature matrix for one model/layer/metric."""
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_head_feature_matrix(
        model=model,
        layer=layer_key,
        percentile=percentile,
        metric=metric,
        variant=variant,
    )
    return HeadFeatureMatrixResponse(**data)


@router.get("/model/{model}/head_exemplars", response_model=HeadExemplarResponse)
async def get_head_exemplars(
    model: str,
    head: Annotated[int, Query(ge=0, le=11)] = 0,
    layer: Annotated[int, Query(ge=0)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    variant: Annotated[Literal["frozen", "linear_probe", "lora", "full"], Query()] = "frozen",
    feature_label: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=24)] = 12,
) -> HeadExemplarResponse:
    """Get representative image candidates for a selected Q3 head."""
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_head_exemplars(
        model=model,
        layer=layer_key,
        head=head,
        percentile=percentile,
        metric=metric,
        variant=variant,
        feature_label=feature_label,
        limit=limit,
    )
    return HeadExemplarResponse(**data)


@router.get("/model/{model}/aggregate")
async def get_aggregate_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get aggregate metrics for a model/layer combination.

    Returns mean, std, median IoU across all images.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_aggregate_metrics(model, layer_key, percentile, method=resolved_method)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Aggregate metrics not found for {model}/{layer_key}",
        )

    return result


@router.get("/model/{model}/all_images")
async def get_all_images_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    sort_by: Annotated[str, Query(enum=["iou", "coverage"])] = "iou",
    limit: Annotated[int, Query(ge=1, le=200)] = 139,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get metrics for all images for a model/layer.

    Returns list of images sorted by IoU or coverage.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    results = metrics_service.get_all_image_metrics(model, layer_key, percentile, method=resolved_method)

    # Sort
    if sort_by == "coverage":
        results.sort(key=lambda x: x["coverage"], reverse=True)
    # Already sorted by IoU from DB

    # Limit
    results = results[:limit]

    return {
        "model": model,
        "layer": layer_key,
        "percentile": percentile,
        "count": len(results),
        "images": results,
    }
