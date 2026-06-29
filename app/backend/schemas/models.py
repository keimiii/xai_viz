"""Pydantic models for API request/response validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AnalysisMetric = Literal["iou", "coverage", "mse", "kl", "emd"]
DashboardMetric = Literal["iou", "coverage", "mse", "kl", "emd"]
CompareVariant = Literal["frozen", "linear_probe", "lora", "full"]
ShiftComparedVariant = Literal["linear_probe", "lora", "full"]


class BoundingBoxSchema(BaseModel):
    """A single bounding box annotation."""

    left: float = Field(..., ge=0, le=1, description="Left edge (0-1)")
    top: float = Field(..., ge=0, le=1, description="Top edge (0-1)")
    width: float = Field(..., ge=0, le=1, description="Width (0-1)")
    height: float = Field(..., ge=0, le=1, description="Height (0-1)")
    label: int = Field(..., ge=0, description="Feature type index")
    label_name: str | None = Field(None, description="Human-readable feature name")


class ImageAnnotationSchema(BaseModel):
    """Complete annotation for an image."""

    image_id: str
    styles: list[str]
    style_names: list[str]
    num_bboxes: int
    bboxes: list[BoundingBoxSchema]


class ImageListItem(BaseModel):
    """Summary info for image list view."""

    image_id: str
    thumbnail_url: str
    styles: list[str]
    style_names: list[str]
    num_bboxes: int


class ImageDetailSchema(BaseModel):
    """Detailed image information."""

    image_id: str
    image_url: str
    thumbnail_url: str
    annotation: ImageAnnotationSchema
    available_models: list[str]


class IoUResultSchema(BaseModel):
    """Per-image alignment metrics."""

    image_id: str
    model: str
    layer: str
    percentile: int
    iou: float
    coverage: float
    mse: float
    kl: float
    emd: float
    attention_area: float
    annotation_area: float
    method: str | None = None


class ImageMetricDescriptorSchema(BaseModel):
    """Descriptor for one image-detail metric series."""

    key: AnalysisMetric
    label: str
    direction: Literal["higher", "lower"]
    default_enabled: bool
    percentile_dependent: bool


class ImageMetricSelectionSchema(BaseModel):
    """Current metric selection context for image-detail progression."""

    mode: Literal["union", "bbox"]
    bbox_index: int | None = None
    bbox_label: str | None = None


class ImageLayerMetricPointSchema(BaseModel):
    """Metric values for a single layer in image-detail progression."""

    layer: int
    layer_key: str
    values: dict[str, float | None]


class ImageLayerProgressionSchema(BaseModel):
    """Extensible per-image metric progression across layers."""

    image_id: str
    model: str
    method: str
    percentile: int
    selection: ImageMetricSelectionSchema
    metrics: list[ImageMetricDescriptorSchema]
    layers: list[ImageLayerMetricPointSchema]


class MetricsQueryParams(BaseModel):
    """Query parameters for metrics endpoints."""

    model: str = "dinov2"
    layer: str = "layer0"  # Safe default for all models (some have only 4 layers)
    percentile: int = 90


class LeaderboardEntry(BaseModel):
    """Model ranking entry for a selected metric."""

    rank: int
    model: str
    metric: DashboardMetric
    score: float
    best_layer: str
    method_used: str


class LayerProgressionSchema(BaseModel):
    """Metric progression across layers."""

    model: str
    metric: DashboardMetric
    percentile: int
    layers: list[str]
    scores: list[float]
    best_layer: str
    best_score: float
    method: str | None = None


class StyleBreakdownSchema(BaseModel):
    """Metric breakdown by architectural style."""

    model: str
    layer: str
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    styles: dict[str, float]
    style_counts: dict[str, int]
    method: str | None = None


class ModelComparisonSchema(BaseModel):
    """Comparison data for multiple models."""

    image_id: str
    models: list[str]
    layer: str
    percentile: int
    selection: ImageMetricSelectionSchema
    results: list[IoUResultSchema]
    heatmap_urls: dict[str, str]  # model -> heatmap URL
    unavailable_models: dict[str, str] = Field(
        default_factory=dict,
        description="Per-model reasons why scoped metrics are unavailable",
    )


class AllModelsSummaryModelEntry(BaseModel):
    """Summary stats for one model at a selected metric/percentile."""

    rank: int
    best_layer: str
    best_score: float
    method_used: str
    layer_progression: dict[str, float]


class AllModelsSummarySchema(BaseModel):
    """Summary comparison across all models for a selected metric."""

    percentile: int
    metric: DashboardMetric
    ranking_mode: Literal["default_method", "best_available"] | None = None
    method: str | None = None
    excluded_models: list[str] = Field(default_factory=list)
    models: dict[str, AllModelsSummaryModelEntry]
    leaderboard: list[LeaderboardEntry]


class BboxInput(BaseModel):
    """Input for similarity computation - a single bounding box."""

    left: float = Field(..., ge=0, le=1, description="Left edge (0-1)")
    top: float = Field(..., ge=0, le=1, description="Top edge (0-1)")
    width: float = Field(..., ge=0, le=1, description="Width (0-1)")
    height: float = Field(..., ge=0, le=1, description="Height (0-1)")
    label: str | None = Field(None, description="Optional label for the feature")


class SimilarityResponse(BaseModel):
    """Response containing cosine similarity values for all patches."""

    similarity: list[float] = Field(..., description="Similarity values for each patch")
    patch_grid: list[int] = Field(..., description="Grid dimensions [rows, cols]")
    min_similarity: float = Field(..., description="Minimum similarity value")
    max_similarity: float = Field(..., description="Maximum similarity value")
    bbox_patch_indices: list[int] = Field(
        ..., description="Indices of patches within the bbox"
    )


class FeatureMetricEntry(BaseModel):
    """Metric summary for a single architectural feature type."""

    feature_label: int = Field(..., description="Feature type index (0-105)")
    feature_name: str = Field(..., description="Human-readable feature name")
    mean_score: float = Field(..., description="Mean metric score across all bboxes of this type")
    std_score: float = Field(..., description="Standard deviation of the metric")
    bbox_count: int = Field(..., description="Number of bboxes of this type")


class FeatureBreakdownSchema(BaseModel):
    """Metric breakdown by architectural feature type."""

    model: str
    layer: str
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    features: list[FeatureMetricEntry]
    total_feature_types: int = Field(..., description="Total number of feature types returned")
    method: str | None = None


class HeadRankingEntrySchema(BaseModel):
    """Aggregate Q3 ranking stats for one attention head."""

    head: int
    mean_score: float
    std_score: float
    mean_rank: float
    top1_count: int
    top3_count: int
    image_count: int


class HeadRankingResponse(BaseModel):
    """Metric-specific Q3 head ranking payload."""

    model: str
    variant: CompareVariant
    layer: str
    method: str | None = None
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    supported: bool = True
    reason: str | None = None
    heads: list[HeadRankingEntrySchema] = Field(default_factory=list)


class ImageHeadRankingEntrySchema(BaseModel):
    """Image-scoped Q3 score for one attention head."""

    head: int
    score: float


class ImageHeadRankingResponse(BaseModel):
    """Metric-specific Q3 ranking payload for one image."""

    image_id: str
    model: str
    variant: CompareVariant
    layer: str
    method: str | None = None
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    selection: ImageMetricSelectionSchema
    supported: bool = True
    reason: str | None = None
    heads: list[ImageHeadRankingEntrySchema] = Field(default_factory=list)


class HeadFeatureMatrixRowSchema(BaseModel):
    """One feature row in the Q3 head-by-feature matrix."""

    feature_label: int
    feature_name: str
    bbox_count: int
    scores: list[float | None]


class HeadFeatureMatrixResponse(BaseModel):
    """Metric-specific Q3 feature matrix payload."""

    model: str
    variant: CompareVariant
    layer: str
    method: str | None = None
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    supported: bool = True
    reason: str | None = None
    heads: list[int] = Field(default_factory=list)
    features: list[HeadFeatureMatrixRowSchema] = Field(default_factory=list)
    total_feature_types: int = 0


class HeadExemplarCandidateSchema(BaseModel):
    """One candidate image for drilling into a selected Q3 head."""

    image_id: str
    score: float
    thumbnail_url: str
    style_names: list[str] = Field(default_factory=list)
    matching_bbox_indices: list[int] = Field(default_factory=list)
    default_bbox_index: int | None = None


class HeadExemplarResponse(BaseModel):
    """Representative image candidates for one Q3 head drill-down."""

    model: str
    variant: CompareVariant
    layer: str
    metric: AnalysisMetric
    direction: Literal["higher", "lower"]
    percentile: int
    head: int
    feature_label: int | None = None
    feature_name: str | None = None
    supported: bool = True
    reason: str | None = None
    candidates: list[HeadExemplarCandidateSchema] = Field(default_factory=list)


class RawAttentionResponse(BaseModel):
    """Raw attention values for client-side rendering."""

    attention: list[float] = Field(..., description="Flattened attention values (row-major order)")
    shape: list[int] = Field(..., description="Grid dimensions [rows, cols]")
    min_value: float = Field(..., description="Minimum attention value")
    max_value: float = Field(..., description="Maximum attention value")


class ComparisonVariantSchema(BaseModel):
    """One side of a variant comparison."""

    model_key: str
    strategy: str | None = None
    label: str
    available: bool
    url: str | None = None


class VariantComparisonSchema(BaseModel):
    """Comparison payload for two selected model variants."""

    image_id: str
    model: str
    layer: str
    method: str
    show_bboxes: bool = True
    left: ComparisonVariantSchema
    right: ComparisonVariantSchema
    note: str


class VariantShiftMapSchema(BaseModel):
    """Numeric attention-shift payload for frozen-vs-variant comparisons."""

    image_id: str
    model: str
    layer: str
    method: str
    available: bool = False
    reason: str | None = None
    baseline_variant: Literal["frozen"] = "frozen"
    compared_variant: ShiftComparedVariant
    baseline_model_key: str
    compared_model_key: str
    operation: str = "compared_variant_attention - frozen_attention"
    shape: list[int] = Field(default_factory=list)
    shift: list[float] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    max_abs_value: float | None = None


class Q2SummaryRowSchema(BaseModel):
    """One aggregate fine-tuning result row for Q2."""

    model_name: str
    strategy_id: str
    metric: AnalysisMetric
    label: str
    direction: Literal["higher", "lower"]
    percentile_dependent: bool
    percentile: int | None = None
    method: str
    frozen_mean: float
    finetuned_mean: float
    mean_delta: float
    std_delta: float
    delta_ci_lower: float
    delta_ci_upper: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None = None
    significant: bool
    test_name: str
    num_images: int


class Q2StrategyComparisonSchema(BaseModel):
    """One within-model strategy comparison row for Q2."""

    model_name: str
    metric: AnalysisMetric
    percentile: int | None = None
    strategy_a: str
    strategy_b: str
    mean_delta_difference: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None = None
    significant: bool
    test_name: str


class Q2SummaryResponse(BaseModel):
    """Metric-generic Q2 analysis payload."""

    metric: AnalysisMetric
    label: str
    direction: Literal["higher", "lower"]
    percentile_dependent: bool
    selected_percentile: int | None = None
    experiment_id: str | None = None
    split_id: str | None = None
    analysis_git_commit_sha: str | None = None
    analyzed_layer: int
    evaluation_image_count: int | None = None
    checkpoint_selection_rule: str | None = None
    result_set_scope: str | None = None
    timestamp: str | None = None
    rows: list[Q2SummaryRowSchema]
    strategy_comparisons: list[Q2StrategyComparisonSchema]


class Q2ImageDeltaEntrySchema(BaseModel):
    """One per-image Q2 IoU delta entry."""

    image_id: str
    delta_iou: float
    style_names: list[str] = Field(default_factory=list)


class Q2ImageDeltasResponse(BaseModel):
    """Image-level Q2 deltas for a model/strategy/percentile slice."""

    model_name: str
    strategy_id: Literal["linear_probe", "lora", "full"]
    percentile: int
    method: str | None = None
    mean_delta_iou: float | None = None
    num_images: int | None = None
    top_positive: list[Q2ImageDeltaEntrySchema] = Field(default_factory=list)
    top_negative: list[Q2ImageDeltaEntrySchema] = Field(default_factory=list)
