"""Comparison endpoints for model vs model analysis."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

from app.backend.config import AVAILABLE_MODELS, get_model_num_layers, resolve_model_name
from app.backend.schemas import (
    AllModelsSummaryModelEntry,
    AllModelsSummarySchema,
    ComparisonVariantSchema,
    ImageMetricSelectionSchema,
    IoUResultSchema,
    LeaderboardEntry,
    ModelComparisonSchema,
    VariantComparisonSchema,
    VariantShiftMapSchema,
)
from app.backend.services.attention_service import attention_service
from app.backend.services.image_service import image_service
from app.backend.services.metrics_service import metrics_service
from app.backend.validators import (
    model_supports_method,
    resolve_default_method,
    resolve_ranking_mode_request,
    split_model_variant,
    validate_attention_method,
    validate_layer_for_model,
    validate_method,
    validate_model,
)

router = APIRouter(prefix="/compare", tags=["comparison"])
CompareVariant = Literal["frozen", "linear_probe", "lora", "full"]
ShiftComparedVariant = Literal["linear_probe", "lora", "full"]


def _build_overlay_url(
    image_id: str,
    model: str,
    layer: int,
    method: str,
    show_bboxes: bool,
) -> str:
    return (
        f"/api/attention/{image_id}/overlay?"
        f"model={model}&layer={layer}&method={method}&show_bboxes={str(show_bboxes).lower()}"
    )


def _strategy_variant_key(model: str, strategy: str | None) -> str:
    return f"{model}_finetuned_{strategy}" if strategy else f"{model}_finetuned"


def _finetuned_variant_candidates(base_model: str, strategy: str | None) -> list[tuple[str | None, str]]:
    """Return ordered model-key candidates for a fine-tuned strategy.

    The canonical strategy-aware cache keys are preferred, with legacy
    `{model}_finetuned` kept as a `full` fallback so older artifacts continue
    to work during the transition.
    """
    if strategy is None:
        return [(None, f"{base_model}_finetuned")]

    candidates: list[tuple[str | None, str]] = [(strategy, _strategy_variant_key(base_model, strategy))]
    if strategy == "full":
        candidates.append((strategy, f"{base_model}_finetuned"))
    return candidates


def _resolve_overlay_candidate(
    *,
    candidates: list[tuple[str | None, str]],
    image_id: str,
    layer_key: str,
    layer: int,
    method: str,
    show_bboxes: bool,
) -> tuple[str, str | None, bool, str | None]:
    """Resolve the first available overlay candidate, falling back gracefully."""
    overlay_variant = "overlay_bbox" if show_bboxes else "overlay"
    fallback_model_key = candidates[0][1]
    fallback_strategy = candidates[0][0]

    for candidate_strategy, candidate_model in candidates:
        try:
            resolved_model = validate_model(candidate_model)
        except HTTPException:
            continue

        if candidate_model == fallback_model_key:
            fallback_model_key = resolved_model

        available = image_service.heatmap_exists(
            resolved_model,
            layer_key,
            image_id,
            method=method,
            variant=overlay_variant,
        )
        if available:
            return (
                resolved_model,
                candidate_strategy,
                True,
                _build_overlay_url(image_id, resolved_model, layer, method, show_bboxes),
            )

    return fallback_model_key, fallback_strategy, False, None


def _resolve_attention_candidate(
    *,
    candidates: list[tuple[str | None, str]],
    image_id: str,
    layer_key: str,
    method: str,
) -> tuple[str, str | None, bool]:
    """Resolve the first available raw-attention cache candidate."""
    fallback_model_key = candidates[0][1]
    fallback_strategy = candidates[0][0]

    for candidate_strategy, candidate_model in candidates:
        try:
            resolved_model = validate_model(candidate_model)
        except HTTPException:
            continue

        if candidate_model == fallback_model_key:
            fallback_model_key = resolved_model

        if attention_service.exists(
            resolved_model,
            layer_key,
            image_id,
            method=method,
        ):
            return resolved_model, candidate_strategy, True

    return fallback_model_key, fallback_strategy, False


def _variant_label(variant: CompareVariant) -> str:
    if variant == "frozen":
        return "Frozen (Pretrained)"
    if variant == "linear_probe":
        return "Linear Probe"
    if variant == "lora":
        return "LoRA"
    return "Full Fine-tune"


def _resolve_variant(
    *,
    base_model: str,
    image_id: str,
    layer_key: str,
    layer: int,
    method: str,
    show_bboxes: bool,
    variant: CompareVariant,
) -> ComparisonVariantSchema:
    overlay_variant = "overlay_bbox" if show_bboxes else "overlay"

    if variant == "frozen":
        available = image_service.heatmap_exists(
            base_model,
            layer_key,
            image_id,
            method=method,
            variant=overlay_variant,
        )
        return ComparisonVariantSchema(
            model_key=base_model,
            strategy=None,
            label=_variant_label(variant),
            available=available,
            url=_build_overlay_url(image_id, base_model, layer, method, show_bboxes) if available else None,
        )

    model_key, resolved_strategy, available, url = _resolve_overlay_candidate(
        candidates=_finetuned_variant_candidates(base_model, variant),
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=method,
        show_bboxes=show_bboxes,
    )
    return ComparisonVariantSchema(
        model_key=model_key,
        strategy=resolved_strategy,
        label=_variant_label(variant),
        available=available,
        url=url,
    )


def _resolve_finetuned_variant(
    *,
    base_model: str,
    image_id: str,
    layer_key: str,
    layer: int,
    method: str,
    show_bboxes: bool,
    strategy: str | None,
    auto_select: bool,
) -> tuple[str, str | None, bool, str | None]:
    if strategy:
        validate_model(_strategy_variant_key(base_model, strategy))
        return _resolve_overlay_candidate(
            candidates=_finetuned_variant_candidates(base_model, strategy),
            image_id=image_id,
            layer_key=layer_key,
            layer=layer,
            method=method,
            show_bboxes=show_bboxes,
        )

    if not auto_select:
        return _resolve_overlay_candidate(
            candidates=_finetuned_variant_candidates(base_model, None),
            image_id=image_id,
            layer_key=layer_key,
            layer=layer,
            method=method,
            show_bboxes=show_bboxes,
        )

    auto_candidates = (
        _finetuned_variant_candidates(base_model, "lora")
        + _finetuned_variant_candidates(base_model, "full")
        + _finetuned_variant_candidates(base_model, "linear_probe")
    )
    model_key, resolved_strategy, available, url = _resolve_overlay_candidate(
        candidates=auto_candidates,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=method,
        show_bboxes=show_bboxes,
    )
    if available:
        return model_key, resolved_strategy, available, url
    return f"{base_model}_finetuned", None, False, None


@router.get("/models", response_model=ModelComparisonSchema)
async def compare_models(
    image_id: str,
    models: Annotated[list[str] | None, Query(description="Models to compare")] = None,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    bbox_index: Annotated[int | None, Query(ge=0)] = None,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> ModelComparisonSchema:
    """Compare multiple models on a single image.

    Returns IoU results and heatmap URLs for side-by-side comparison.
    Returns 400 if the requested method is not available for any selected model.
    """
    # Default models if not specified
    if models is None:
        models = ["dinov2", "clip"]

    # Validate models and layer for all requested models
    for model in models:
        validate_model(model)
        validate_layer_for_model(layer, model)

    # Check image exists
    annotation = image_service.get_annotation(image_id)
    if annotation is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if bbox_index is None and not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    layer_key = f"layer{layer}"
    selection = ImageMetricSelectionSchema(
        mode="union",
        bbox_index=None,
        bbox_label=None,
    )
    if bbox_index is not None:
        try:
            selection = ImageMetricSelectionSchema(
                mode="bbox",
                bbox_index=bbox_index,
                bbox_label=metrics_service.get_bbox_label(image_id, bbox_index),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = []
    heatmap_urls = {}
    unavailable_models: dict[str, str] = {}

    for model in models:
        resolved_model = resolve_model_name(model)

        resolved_method = validate_method(model, method)

        if bbox_index is None:
            metrics = metrics_service.get_image_metrics(
                image_id,
                model,
                layer_key,
                percentile,
                method=resolved_method,
            )
        else:
            try:
                metrics = metrics_service.get_bbox_metrics(
                    image_id=image_id,
                    model=model,
                    layer=layer_key,
                    bbox_index=bbox_index,
                    percentile=percentile,
                    method=resolved_method,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if metrics:
            results.append(IoUResultSchema(**metrics))
        elif bbox_index is not None:
            unavailable_models[model] = (
                "Feature-level metrics unavailable because cached attention is missing "
                f"for {model}/{layer_key}/{resolved_method}/{image_id}."
            )

        # Get heatmap URL (include method)
        if image_service.heatmap_exists(resolved_model, layer_key, image_id, method=resolved_method, variant="overlay"):
            heatmap_urls[model] = f"/api/attention/{image_id}/overlay?model={model}&layer={layer}&method={resolved_method}"

    if not results and (bbox_index is None or not unavailable_models):
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    return ModelComparisonSchema(
        image_id=image_id,
        models=models,
        layer=layer_key,
        percentile=percentile,
        selection=selection,
        results=results,
        heatmap_urls=heatmap_urls,
        unavailable_models=unavailable_models,
    )


@router.get("/variants", response_model=VariantComparisonSchema)
async def compare_variants(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    left_variant: Annotated[CompareVariant, Query(description="Left comparison variant")] = "frozen",
    right_variant: Annotated[CompareVariant, Query(description="Right comparison variant")] = "full",
    show_bboxes: Annotated[bool, Query(description="Render overlays with annotation boxes/labels")] = True,
) -> VariantComparisonSchema:
    """Compare any two supported frozen/fine-tuned variants for the same base model."""
    resolved_model = validate_model(model)
    base_model, _, _ = split_model_variant(resolved_model)
    layer_key = validate_layer_for_model(layer, base_model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    method = resolve_default_method(base_model)
    left = _resolve_variant(
        base_model=base_model,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=method,
        show_bboxes=show_bboxes,
        variant=left_variant,
    )
    right = _resolve_variant(
        base_model=base_model,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=method,
        show_bboxes=show_bboxes,
        variant=right_variant,
    )

    note = (
        "One or both overlays are unavailable for this model/layer/image. "
        "Generate fine-tuned attention + heatmap caches for the selected variants first."
    )

    return VariantComparisonSchema(
        image_id=image_id,
        model=base_model,
        layer=layer_key,
        method=method,
        show_bboxes=show_bboxes,
        left=left,
        right=right,
        note=note,
    )


@router.get("/variants/shift", response_model=VariantShiftMapSchema)
async def compare_variant_shift(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    compared_variant: Annotated[
        ShiftComparedVariant,
        Query(description="Non-frozen comparison variant used in shift = compared - frozen"),
    ] = "full",
) -> VariantShiftMapSchema:
    """Return a numeric frozen-vs-variant attention-shift map."""
    resolved_model = validate_model(model)
    base_model, _, _ = split_model_variant(resolved_model)
    layer_key = validate_layer_for_model(layer, base_model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    method = resolve_default_method(base_model)
    baseline_model_key = base_model
    baseline_available = attention_service.exists(
        base_model,
        layer_key,
        image_id,
        method=method,
    )
    compared_model_key, _resolved_strategy, compared_available = _resolve_attention_candidate(
        candidates=_finetuned_variant_candidates(base_model, compared_variant),
        image_id=image_id,
        layer_key=layer_key,
        method=method,
    )

    if not baseline_available or not compared_available:
        reasons = []
        if not baseline_available:
            reasons.append(
                "Frozen attention is not cached for this model/layer/image. "
                "Run generate_attention_cache.py first."
            )
        if not compared_available:
            reasons.append(
                "Compared variant attention is not cached for this model/layer/image. "
                "Generate fine-tuned attention caches for the requested strategy first."
            )

        return VariantShiftMapSchema(
            image_id=image_id,
            model=base_model,
            layer=layer_key,
            method=method,
            available=False,
            reason=" ".join(reasons),
            compared_variant=compared_variant,
            baseline_model_key=baseline_model_key,
            compared_model_key=compared_model_key,
        )

    try:
        shift_payload = attention_service.get_attention_shift(
            image_id=image_id,
            baseline_model=baseline_model_key,
            compared_model=compared_model_key,
            layer=layer,
            method=method,
        )
    except ValueError as exc:
        return VariantShiftMapSchema(
            image_id=image_id,
            model=base_model,
            layer=layer_key,
            method=method,
            available=False,
            reason=str(exc),
            compared_variant=compared_variant,
            baseline_model_key=baseline_model_key,
            compared_model_key=compared_model_key,
        )

    return VariantShiftMapSchema(
        image_id=image_id,
        model=base_model,
        layer=layer_key,
        method=method,
        available=True,
        reason=None,
        compared_variant=compared_variant,
        baseline_model_key=baseline_model_key,
        compared_model_key=compared_model_key,
        shape=shift_payload["shape"],
        shift=shift_payload["shift"],
        min_value=shift_payload["min_value"],
        max_value=shift_payload["max_value"],
        max_abs_value=shift_payload["max_abs_value"],
    )


@router.get("/frozen_vs_finetuned")
async def compare_frozen_vs_finetuned(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    strategy: Annotated[str | None, Query(description="Fine-tuning strategy (linear_probe, lora, full)")] = None,
    show_bboxes: Annotated[bool, Query(description="Render overlays with annotation boxes/labels")] = True,
) -> dict:
    """Compare frozen (pretrained) vs fine-tuned model attention.
    """
    resolved_model = validate_model(model)
    base_model, _, _ = split_model_variant(resolved_model)
    layer_key = validate_layer_for_model(layer, base_model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    method = resolve_default_method(base_model)

    # Frozen model (always available after pre-computation)
    frozen_available = image_service.heatmap_exists(
        base_model,
        layer_key,
        image_id,
        method=method,
        variant="overlay_bbox" if show_bboxes else "overlay",
    )

    finetuned_model, resolved_strategy, finetuned_available, finetuned_url = _resolve_finetuned_variant(
        base_model=base_model,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=method,
        show_bboxes=show_bboxes,
        strategy=strategy,
        auto_select=True,
    )

    return {
        "image_id": image_id,
        "model": base_model,
        "strategy": resolved_strategy,
        "layer": layer_key,
        "show_bboxes": show_bboxes,
        "frozen": {
            "available": frozen_available,
            "url": _build_overlay_url(image_id, base_model, layer, method, show_bboxes) if frozen_available else None,
        },
        "finetuned": {
            "available": finetuned_available,
            "url": finetuned_url,
            "note": (
                "Fine-tuned overlay is unavailable for this model/layer/image. "
                "Generate fine-tuned attention + heatmap caches first."
            ),
        },
    }


@router.get("/finetuned_vs_finetuned")
async def compare_finetuned_variants(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    strategy_a: Annotated[str, Query(description="Left strategy (linear_probe, lora, full)")] = "linear_probe",
    strategy_b: Annotated[str, Query(description="Right strategy (linear_probe, lora, full)")] = "full",
    show_bboxes: Annotated[bool, Query(description="Render overlays with annotation boxes/labels")] = True,
) -> dict:
    """Compare two fine-tuned strategy variants for the same base model."""
    resolved_model = validate_model(model)
    base_model, _, _ = split_model_variant(resolved_model)
    layer_key = validate_layer_for_model(layer, base_model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    left_model, resolved_strategy_a, left_available, left_url = _resolve_finetuned_variant(
        base_model=base_model,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=resolve_default_method(base_model),
        show_bboxes=show_bboxes,
        strategy=strategy_a,
        auto_select=False,
    )
    right_model, resolved_strategy_b, right_available, right_url = _resolve_finetuned_variant(
        base_model=base_model,
        image_id=image_id,
        layer_key=layer_key,
        layer=layer,
        method=resolve_default_method(base_model),
        show_bboxes=show_bboxes,
        strategy=strategy_b,
        auto_select=False,
    )
    method = resolve_default_method(base_model)

    note = (
        "One or both fine-tuned overlays are unavailable for this model/layer/image. "
        "Generate fine-tuned attention + heatmap caches for both strategies first."
    )

    return {
        "image_id": image_id,
        "model": base_model,
        "layer": layer_key,
        "method": method,
        "show_bboxes": show_bboxes,
        "left": {
            "model_key": left_model,
            "strategy": resolved_strategy_a,
            "label": f"Fine-tuned ({resolved_strategy_a})" if resolved_strategy_a else "Fine-tuned",
            "available": left_available,
            "url": left_url,
        },
        "right": {
            "model_key": right_model,
            "strategy": resolved_strategy_b,
            "label": f"Fine-tuned ({resolved_strategy_b})" if resolved_strategy_b else "Fine-tuned",
            "available": right_available,
            "url": right_url,
        },
        "note": note,
    }


@router.get("/layers")
async def compare_layers(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get IoU progression across layers for layer comparison.

    Used for layer progression animation and analysis.
    """
    resolved_model = validate_model(model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    # Get per-model layer count
    num_layers = get_model_num_layers(resolved_model)
    resolved_method = validate_method(model, method)

    layers_data = []
    for layer in range(num_layers):
        layer_key = f"layer{layer}"
        metrics = metrics_service.get_image_metrics(image_id, model, layer_key, percentile, method=resolved_method)

        if metrics:
            has_heatmap = image_service.heatmap_exists(resolved_model, layer_key, image_id, method=resolved_method, variant="overlay")
            layers_data.append({
                "layer": layer,
                "layer_key": layer_key,
                "iou": metrics["iou"],
                "coverage": metrics["coverage"],
                "heatmap_url": f"/api/attention/{image_id}/overlay?model={model}&layer={layer}&method={resolved_method}" if has_heatmap else None,
            })

    if not layers_data:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    # Find best layer
    best = max(layers_data, key=lambda x: x["iou"])

    return {
        "image_id": image_id,
        "model": model,
        "percentile": percentile,
        "layers": layers_data,
        "best_layer": best["layer"],
        "best_iou": best["iou"],
    }


@router.get("/all_models_summary", response_model=AllModelsSummarySchema)
async def compare_all_models_summary(
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    metric: Annotated[Literal["iou", "coverage", "mse", "kl", "emd"], Query()] = "iou",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
    ranking_mode: Annotated[Literal["default_method", "best_available"] | None, Query()] = None,
) -> AllModelsSummarySchema:
    """Get summary comparison of all models for the selected metric.

    Returns best layer and score for each model.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    resolved_method = validate_attention_method(method)
    resolved_ranking_mode = resolve_ranking_mode_request(resolved_method, ranking_mode)
    excluded_models = (
        [model for model in AVAILABLE_MODELS if not model_supports_method(model, resolved_method)]
        if resolved_method is not None
        else []
    )

    if resolved_method is not None:
        leaderboard = metrics_service.get_leaderboard(
            percentile,
            metric=metric,
            method=resolved_method,
        )
    else:
        leaderboard = metrics_service.get_leaderboard(
            percentile,
            metric=metric,
            ranking_mode=resolved_ranking_mode or "default_method",
        )

    # Get layer progression for each model
    models_data: dict[str, AllModelsSummaryModelEntry] = {}
    for entry in leaderboard:
        model = entry["model"]
        progression = metrics_service.get_layer_progression(
            model,
            percentile,
            method=entry["method_used"],
            metric=metric,
        )
        models_data[model] = AllModelsSummaryModelEntry(
            rank=entry["rank"],
            best_layer=entry["best_layer"],
            best_score=entry["score"],
            method_used=entry["method_used"],
            layer_progression=dict(zip(progression["layers"], progression["scores"], strict=True)),
        )

    return AllModelsSummarySchema(
        percentile=percentile,
        metric=metric,
        ranking_mode=resolved_ranking_mode,
        method=resolved_method,
        excluded_models=excluded_models,
        models=models_data,
        leaderboard=[LeaderboardEntry(**entry) for entry in leaderboard],
    )
