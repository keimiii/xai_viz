/**
 * Generic variant comparison slider for frozen and fine-tuned strategies.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ReactCompareSlider } from 'react-compare-slider';
import { attentionAPI, comparisonAPI, imagesAPI, metricsAPI } from '../../api/client';
import { useQ2Summary } from '../../hooks/useMetrics';
import { Card, CardContent } from '../ui/Card';
import { InteractiveBboxOverlay } from '../attention/InteractiveBboxOverlay';
import { LayerSlider } from '../attention/LayerSlider';
import { useModels } from '../../hooks/useAttention';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import {
  computeSimilarityStats,
  renderDivergingHeatmap,
  renderDivergingHeatmapLegend,
  renderHeatmap,
  renderHeatmapLegend,
} from '../../utils/renderHeatmap';
import {
  ANALYSIS_METRIC_METADATA,
  formatMetricValue,
  metricImprovementTone,
} from '../../constants/metricMetadata';
import type {
  AnalysisMetric,
  BoundingBox,
  CompareVariantId,
  ComparisonVariant,
  IoUResult,
  Q2StrategyComparison,
  Q2SummaryRow,
  ShiftComparedVariantId,
} from '../../types';

type ViewMode = 'side-by-side' | 'slider' | 'shift-map';

interface VariantCompareProps {
  imageId: string;
  model: string;
  layer: number;
  percentile: number;
  metric: AnalysisMetric;
  leftVariant: CompareVariantId;
  rightVariant: CompareVariantId;
  isPlaying: boolean;
  onPlayingChange: (isPlaying: boolean) => void;
  onLayerChange: (layer: number, options?: { replace?: boolean }) => void;
  bboxes?: BoundingBox[];
  showBboxes?: boolean;
}

interface CompareCanvasProps {
  imageSrc: string;
  imageAlt: string;
  overlaySrc?: string | null;
  overlayAlt?: string;
  imageClassName?: string;
  overlayClassName?: string;
}

interface NormalizedVariantComparison {
  left: ComparisonVariant;
  right: ComparisonVariant;
  note: string;
  linearProbeInvolved: boolean;
}

function CompareCanvas({
  imageSrc,
  imageAlt,
  overlaySrc,
  overlayAlt,
  imageClassName,
  overlayClassName,
}: CompareCanvasProps) {
  const baseImageClasses = ['absolute inset-0 h-full w-full object-cover', imageClassName]
    .filter(Boolean)
    .join(' ');
  const overlayClasses = [
    'pointer-events-none absolute inset-0 h-full w-full object-cover',
    overlayClassName,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg bg-gray-950">
      <img
        src={imageSrc}
        alt={imageAlt}
        className={baseImageClasses}
      />
      {overlaySrc && (
        <img
          src={overlaySrc}
          alt={overlayAlt || imageAlt}
          className={overlayClasses}
        />
      )}
    </div>
  );
}

function getVariantId(variant: ComparisonVariant): CompareVariantId {
  if (!variant.strategy) {
    return 'frozen';
  }
  if (variant.strategy === 'linear_probe' || variant.strategy === 'lora' || variant.strategy === 'full') {
    return variant.strategy;
  }
  return 'full';
}

function isShiftComparedVariantId(
  variant: CompareVariantId | null | undefined
): variant is ShiftComparedVariantId {
  return variant === 'linear_probe' || variant === 'lora' || variant === 'full';
}

function getShiftComparedVariantId(
  leftVariant: CompareVariantId | null,
  rightVariant: CompareVariantId | null
): ShiftComparedVariantId | null {
  if (leftVariant === 'frozen' && isShiftComparedVariantId(rightVariant)) {
    return rightVariant;
  }
  if (rightVariant === 'frozen' && isShiftComparedVariantId(leftVariant)) {
    return leftVariant;
  }
  return null;
}

function findStrategyPair(
  comparisons: Q2StrategyComparison[],
  leftStrategy: string,
  rightStrategy: string
) {
  return comparisons.find(
    (comparison) =>
      (comparison.strategy_a === leftStrategy && comparison.strategy_b === rightStrategy) ||
      (comparison.strategy_a === rightStrategy && comparison.strategy_b === leftStrategy)
  );
}

function findStrategySummary(rows: Q2SummaryRow[], strategyId: string | null | undefined) {
  if (!strategyId) {
    return null;
  }
  return rows.find((row) => row.strategy_id === strategyId) ?? null;
}

function toneClassFromDelta(metric: AnalysisMetric, delta: number | null | undefined) {
  const tone = metricImprovementTone(ANALYSIS_METRIC_METADATA[metric].direction, delta);
  if (tone === 'positive') {
    return 'border-green-200 bg-green-50 text-green-700';
  }
  if (tone === 'negative') {
    return 'border-red-200 bg-red-50 text-red-700';
  }
  return 'border-gray-200 bg-gray-100 text-gray-700';
}

function getMetricResult(metrics: IoUResult, metric: AnalysisMetric): number {
  return metrics[metric as keyof IoUResult] as number;
}

function VariantMetricSummary({
  metrics,
  selectedMetric,
}: {
  metrics: IoUResult;
  selectedMetric: AnalysisMetric;
}) {
  const orderedMetrics: AnalysisMetric[] = [
    selectedMetric,
    ...(['iou', 'coverage', 'mse', 'kl', 'emd'] as AnalysisMetric[]).filter((metric) => metric !== selectedMetric),
  ];

  return (
    <div className="space-y-1 text-sm">
      {orderedMetrics.map((metric) => {
        const isSelected = metric === selectedMetric;
        const classes = isSelected ? 'font-medium text-gray-900' : 'text-gray-700';
        return (
          <p key={metric} className={classes}>
            {ANALYSIS_METRIC_METADATA[metric].shortLabel}: {formatMetricValue(metric, getMetricResult(metrics, metric))}
          </p>
        );
      })}
    </div>
  );
}

function ExperimentSummaryCard({
  title,
  summary,
  analyzedLayer,
}: {
  title: string;
  summary: Q2SummaryRow;
  analyzedLayer: number | null;
}) {
  const metric = summary.metric;
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const significanceLabel =
    summary.corrected_p_value !== null
      ? (summary.significant ? 'Holm-corrected significant' : 'Not significant after Holm correction')
      : (summary.significant ? 'Significant' : 'Not significant');

  return (
    <Card>
      <CardContent className="space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Experiment summary</p>
          <p className="text-sm font-medium text-gray-900">{title}</p>
          <p className="text-xs text-gray-600">
            {metricMetadata.optionLabel}
            {analyzedLayer !== null ? ` · layer ${analyzedLayer}` : ''}
            {' · '}
            {summary.num_images} annotated images
            {' · '}
            {metricMetadata.direction === 'higher' ? 'higher is better' : 'lower is better'}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-gray-500">Mean delta</p>
            <p className="text-2xl font-semibold text-gray-900">{formatMetricValue(metric, summary.mean_delta, { signed: true })}</p>
            <p className="text-xs text-gray-600">
              CI [{formatMetricValue(metric, summary.delta_ci_lower, { signed: true })}, {formatMetricValue(metric, summary.delta_ci_upper, { signed: true })}]
            </p>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="font-medium">Frozen:</span> {formatMetricValue(metric, summary.frozen_mean)}
            </p>
            <p>
              <span className="font-medium">Fine-tuned:</span> {formatMetricValue(metric, summary.finetuned_mean)}
            </p>
            <p>
              <span className="font-medium">Effect size:</span> {summary.cohens_d.toFixed(3)}
            </p>
            <p>
              <span className="font-medium">Significance:</span>{' '}
              {significanceLabel}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PairwiseSummaryCard({
  comparison,
  analyzedLayer,
}: {
  comparison: Q2StrategyComparison;
  analyzedLayer: number | null;
}) {
  const metric = comparison.metric;
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const improvementCopy =
    metricMetadata.direction === 'higher'
      ? `Positive means ${comparison.strategy_a} improves more than ${comparison.strategy_b}.`
      : `Negative means ${comparison.strategy_a} improves more than ${comparison.strategy_b}.`;
  const significanceLabel =
    comparison.corrected_p_value !== null
      ? (comparison.significant ? 'Holm-corrected significant' : 'Not significant after Holm correction')
      : (comparison.significant ? 'Significant' : 'Not significant');

  return (
    <Card>
      <CardContent className="space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Experiment summary</p>
          <p className="text-sm font-medium text-gray-900">Cross-strategy comparison</p>
          <p className="text-xs text-gray-600">
            {comparison.strategy_a} vs {comparison.strategy_b} · {metricMetadata.optionLabel}
            {analyzedLayer !== null ? ` · layer ${analyzedLayer}` : ''}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-gray-500">Delta difference</p>
            <p className="text-2xl font-semibold text-gray-900">
              {formatMetricValue(metric, comparison.mean_delta_difference, { signed: true })}
            </p>
            <p className="text-xs text-gray-600">{improvementCopy}</p>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="font-medium">Effect size:</span> {comparison.cohens_d.toFixed(3)}
            </p>
            <p>
              <span className="font-medium">p-value:</span> {(comparison.corrected_p_value ?? comparison.p_value).toFixed(4)}
            </p>
            <p>
              <span className="font-medium">Significance:</span>{' '}
              {significanceLabel}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function VariantCompare({
  imageId,
  model,
  layer,
  percentile,
  metric,
  leftVariant,
  rightVariant,
  isPlaying,
  onPlayingChange,
  onLayerChange,
  bboxes = [],
  showBboxes = true,
}: VariantCompareProps) {
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [q2SummaryExpanded, setQ2SummaryExpanded] = useState(false);
  const { data: modelsData, isLoading: modelsLoading } = useModels();
  const heatmapOpacity = useHeatmapOpacity();
  const heatmapStyle = useHeatmapStyle();
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const maxLayers = modelsData?.num_layers_per_model?.[model] ?? modelsData?.num_layers ?? 12;
  const effectiveLayer = Math.min(layer, Math.max(maxLayers - 1, 0));
  const percentileCopy = metricMetadata.thresholdFree
    ? `${metricMetadata.optionLabel} is threshold-free`
    : `Top ${100 - percentile}% attention`;

  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: [
      'variant-compare',
      imageId,
      model,
      effectiveLayer,
      leftVariant,
      rightVariant,
      showBboxes,
    ],
    queryFn: async (): Promise<NormalizedVariantComparison> => {
      const response = await comparisonAPI.compareVariants(
        imageId,
        model,
        effectiveLayer,
        leftVariant,
        rightVariant,
        showBboxes
      );
      return {
        left: response.left,
        right: response.right,
        note: response.note,
        linearProbeInvolved:
          getVariantId(response.left) === 'linear_probe' || getVariantId(response.right) === 'linear_probe',
      };
    },
    enabled: Boolean(imageId && model && !modelsLoading),
  });

  const { data: q2Summary, isLoading: q2SummaryLoading } = useQ2Summary(metric, percentile, model);
  const leftComparedVariant = data ? getVariantId(data.left) : null;
  const rightComparedVariant = data ? getVariantId(data.right) : null;
  const shiftComparedVariant = getShiftComparedVariantId(leftComparedVariant, rightComparedVariant);
  const shiftSupported = shiftComparedVariant !== null;
  const shiftResetKey = `${leftComparedVariant ?? 'none'}|${rightComparedVariant ?? 'none'}`;
  const [previousShiftResetKey, setPreviousShiftResetKey] = useState(shiftResetKey);
  if (shiftResetKey !== previousShiftResetKey) {
    setPreviousShiftResetKey(shiftResetKey);
    if (viewMode === 'shift-map' && !shiftSupported) {
      setViewMode('side-by-side');
    }
  }

  const leftUrl = data?.left.url ?? null;
  const rightUrl = data?.right.url ?? null;
  const sliderAvailable =
    Boolean(data?.left.available) &&
    Boolean(data?.right.available) &&
    typeof leftUrl === 'string' &&
    typeof rightUrl === 'string';
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;
  const originalUrl = imagesAPI.getImageUrl(imageId, 224);
  const legendUrl = useMemo(() => renderHeatmapLegend(200, 16), []);

  const featureOptions = useMemo(
    () =>
      bboxes.map((bbox, index) => ({
        index,
        label: bbox.label_name || `Feature ${bbox.label}`,
      })),
    [bboxes]
  );

  const {
    data: bboxMetrics,
    isLoading: bboxMetricsLoading,
    error: bboxMetricsError,
  } = useQuery({
    queryKey: [
      'variant-compare-bbox-metrics',
      imageId,
      effectiveLayer,
      percentile,
      selectedBboxIndex,
      data?.left.model_key,
      data?.right.model_key,
    ],
    queryFn: async () => {
      if (selectedBboxIndex === null || !data) {
        return null;
      }

      const [left, right] = await Promise.all([
        metricsAPI.getBboxMetrics(imageId, data.left.model_key, effectiveLayer, selectedBboxIndex, percentile),
        metricsAPI.getBboxMetrics(imageId, data.right.model_key, effectiveLayer, selectedBboxIndex, percentile),
      ]);

      return { left, right };
    },
    enabled: selectedBboxIndex !== null && Boolean(data),
  });

  const {
    data: similarityData,
    isLoading: similarityLoading,
    error: similarityError,
  } = useQuery({
    queryKey: [
      'variant-compare-similarity',
      imageId,
      data?.left.model_key,
      data?.right.model_key,
      effectiveLayer,
      selectedBbox,
    ],
    queryFn: async () => {
      if (!selectedBbox || !data) {
        return null;
      }

      const bboxPayload = {
        left: selectedBbox.left,
        top: selectedBbox.top,
        width: selectedBbox.width,
        height: selectedBbox.height,
        label: selectedBbox.label_name || undefined,
      };

      const [left, right] = await Promise.all([
        attentionAPI.getSimilarity(imageId, bboxPayload, data.left.model_key, effectiveLayer),
        attentionAPI.getSimilarity(imageId, bboxPayload, data.right.model_key, effectiveLayer),
      ]);

      return { left, right };
    },
    enabled: Boolean(selectedBbox) && Boolean(data),
    retry: false,
  });

  const {
    data: shiftData,
    isLoading: shiftLoading,
    error: shiftError,
  } = useQuery({
    queryKey: [
      'variant-compare-shift',
      imageId,
      model,
      effectiveLayer,
      shiftComparedVariant,
    ],
    queryFn: async () => {
      if (!shiftComparedVariant) {
        return null;
      }

      return comparisonAPI.compareVariantShift(
        imageId,
        model,
        effectiveLayer,
        shiftComparedVariant
      );
    },
    enabled: Boolean(imageId && model && shiftComparedVariant && viewMode === 'shift-map'),
    retry: false,
  });

  const leftSimilarity = similarityData?.left?.similarity;
  const leftPatchGrid = similarityData?.left?.patch_grid as [number, number] | undefined;
  const rightSimilarity = similarityData?.right?.similarity;
  const rightPatchGrid = similarityData?.right?.patch_grid as [number, number] | undefined;

  const leftSimilarityHeatmapUrl = useMemo(() => {
    if (!leftSimilarity || !leftPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: leftSimilarity,
        patchGrid: leftPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [heatmapOpacity, heatmapStyle, leftPatchGrid, leftSimilarity]);

  const rightSimilarityHeatmapUrl = useMemo(() => {
    if (!rightSimilarity || !rightPatchGrid) {
      return null;
    }
    try {
      return renderHeatmap({
        similarity: rightSimilarity,
        patchGrid: rightPatchGrid,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [heatmapOpacity, heatmapStyle, rightPatchGrid, rightSimilarity]);

  const shiftLegendUrl = useMemo(() => renderDivergingHeatmapLegend(200, 16), []);
  const shiftHeatmapUrl = useMemo(() => {
    if (!shiftData?.available || !shiftData.shape.length || shiftData.max_abs_value === null) {
      return null;
    }

    try {
      return renderDivergingHeatmap({
        values: shiftData.shift,
        patchGrid: shiftData.shape as [number, number],
        opacity: heatmapOpacity,
        maxAbsValue: shiftData.max_abs_value,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [heatmapOpacity, heatmapStyle, shiftData]);

  const leftSimilarityStats = useMemo(() => {
    if (!leftSimilarity) {
      return null;
    }
    return computeSimilarityStats(leftSimilarity);
  }, [leftSimilarity]);

  const rightSimilarityStats = useMemo(() => {
    if (!rightSimilarity) {
      return null;
    }
    return computeSimilarityStats(rightSimilarity);
  }, [rightSimilarity]);

  const showSimilarityHeatmaps =
    Boolean(selectedBbox) &&
    Boolean(leftSimilarityHeatmapUrl) &&
    Boolean(rightSimilarityHeatmapUrl) &&
    !similarityLoading &&
    !similarityError;

  const q2Rows = q2Summary?.rows ?? [];
  const q2StrategyComparisons = q2Summary?.strategy_comparisons ?? [];
  const q2AnalyzedLayer = q2Summary?.analyzed_layer ?? null;
  const leftSummary = findStrategySummary(q2Rows, data?.left.strategy);
  const rightSummary = findStrategySummary(q2Rows, data?.right.strategy);
  const methodPairSummary =
    data?.left.strategy && data?.right.strategy
      ? findStrategyPair(q2StrategyComparisons, data.left.strategy, data.right.strategy)
      : null;

  const frozenVariant = leftComparedVariant === 'frozen' ? data?.left : rightComparedVariant === 'frozen' ? data?.right : null;
  const comparedVariantDetails = leftComparedVariant === 'frozen' ? data?.right : rightComparedVariant === 'frozen' ? data?.left : null;

  const hasQ2Content =
    !q2SummaryLoading &&
    Boolean(
      (leftComparedVariant && leftComparedVariant !== 'frozen' && leftSummary) ||
      (rightComparedVariant && rightComparedVariant !== 'frozen' && rightSummary) ||
      methodPairSummary
    );

  const deltaValue =
    bboxMetrics && typeof bboxMetrics.right[metric] === 'number' && typeof bboxMetrics.left[metric] === 'number'
      ? bboxMetrics.right[metric] - bboxMetrics.left[metric]
      : null;
  const deltaIoU =
    bboxMetrics && typeof bboxMetrics.right.iou === 'number' && typeof bboxMetrics.left.iou === 'number'
      ? bboxMetrics.right.iou - bboxMetrics.left.iou
      : null;
  const deltaCoverage =
    bboxMetrics &&
    typeof bboxMetrics.right.coverage === 'number' &&
    typeof bboxMetrics.left.coverage === 'number'
      ? bboxMetrics.right.coverage - bboxMetrics.left.coverage
      : null;
  const deltaMse =
    bboxMetrics && typeof bboxMetrics.right.mse === 'number' && typeof bboxMetrics.left.mse === 'number'
      ? bboxMetrics.right.mse - bboxMetrics.left.mse
      : null;
  const deltaKl =
    bboxMetrics && typeof bboxMetrics.right.kl === 'number' && typeof bboxMetrics.left.kl === 'number'
      ? bboxMetrics.right.kl - bboxMetrics.left.kl
      : null;
  const deltaEmd =
    bboxMetrics && typeof bboxMetrics.right.emd === 'number' && typeof bboxMetrics.left.emd === 'number'
      ? bboxMetrics.right.emd - bboxMetrics.left.emd
      : null;
  const deltaTone = toneClassFromDelta(metric, deltaValue);

  const experimentSummary = (
    <div className="space-y-3">
      {q2SummaryLoading && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
          Loading experiment summary...
        </div>
      )}

      {hasQ2Content && (
        <Card>
          <button
            type="button"
            onClick={() => setQ2SummaryExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-gray-50"
            aria-expanded={q2SummaryExpanded}
          >
            <span className="text-sm font-medium text-gray-900">
              Experiment summary
              {q2AnalyzedLayer !== null ? ` (aggregate at layer ${q2AnalyzedLayer})` : ' (aggregate)'}
            </span>
            <span className="text-gray-500" aria-hidden>
              {q2SummaryExpanded ? '▼' : '▶'}
            </span>
          </button>
          {q2SummaryExpanded && (
            <CardContent className="border-t border-gray-200 pt-3">
              <div className="space-y-3">
                {leftSummary && getVariantId(data!.left) !== 'frozen' && (
                  <ExperimentSummaryCard
                    title={`${data?.left.label ?? 'Left variant'} vs frozen baseline`}
                    summary={leftSummary}
                    analyzedLayer={q2AnalyzedLayer}
                  />
                )}
                {rightSummary && rightComparedVariant !== 'frozen' && (
                  <ExperimentSummaryCard
                    title={`${data?.right.label ?? 'Right variant'} vs frozen baseline`}
                    summary={rightSummary}
                    analyzedLayer={q2AnalyzedLayer}
                  />
                )}
                {methodPairSummary && (
                  <PairwiseSummaryCard
                    comparison={methodPairSummary}
                    analyzedLayer={q2AnalyzedLayer}
                  />
                )}
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );

  if (modelsLoading || isLoading) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
        Loading comparison availability...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load comparison data.
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent>
          <LayerSlider
            currentLayer={effectiveLayer}
            maxLayers={maxLayers}
            onChange={(nextLayer) => onLayerChange(nextLayer, { replace: isPlaying })}
            isPlaying={isPlaying}
            onPlayingChange={onPlayingChange}
            playSpeed={400}
          />
        </CardContent>
      </Card>

      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-700">View:</span>
        <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Side by side
          </button>
          <button
            type="button"
            onClick={() => setViewMode('slider')}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              viewMode === 'slider'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Slider
          </button>
          {shiftSupported && (
            <button
              type="button"
              onClick={() => setViewMode('shift-map')}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                viewMode === 'shift-map'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Shift map
            </button>
          )}
        </div>
      </div>

      {experimentSummary}

      {viewMode === 'side-by-side' && (
        <>
          {!sliderAvailable ? (
            <>
              {leftUrl && (
                <div className="relative">
                  <img
                    src={leftUrl}
                    alt={`${data.left.label} attention`}
                    className="h-auto w-full rounded-lg"
                  />
                  <div className="absolute bottom-2 left-2 rounded bg-black/50 px-2 py-1 text-xs text-white">
                    {data.left.label}
                  </div>
                </div>
              )}
              <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
                <h4 className="font-medium text-yellow-800">Comparison overlay unavailable</h4>
                <p className="mt-1 text-sm text-yellow-700">{data.note}</p>
                <p className="mt-1 text-sm text-yellow-700">
                  Side-by-side and slider comparison both require cached overlays for the compared variants.
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardContent className="p-0">
                    <div className="border-b border-gray-200 px-3 py-2">
                      <p className="text-sm font-medium text-gray-900">{data.left.label}</p>
                      <p className="text-xs text-gray-500">
                        {showSimilarityHeatmaps ? 'Feature-local similarity' : 'Global overlay'} · {percentileCopy}
                      </p>
                    </div>
                    <div className="relative aspect-square w-full overflow-hidden bg-gray-950">
                      <CompareCanvas
                        imageSrc={showSimilarityHeatmaps ? originalUrl : leftUrl!}
                        imageAlt={showSimilarityHeatmaps ? `${data.left.label} similarity heatmap` : `${data.left.label} attention`}
                        overlaySrc={showSimilarityHeatmaps ? leftSimilarityHeatmapUrl : null}
                        overlayAlt={`${data.left.label} similarity overlay`}
                      />
                      {similarityLoading && selectedBbox && (
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        </div>
                      )}
                      {showBboxes && bboxes.length > 0 && (
                        <InteractiveBboxOverlay
                          bboxes={bboxes}
                          selectedIndex={selectedBboxIndex}
                          onBboxClick={(_bbox, index) => setSelectedBboxIndex(selectedBboxIndex === index ? null : index)}
                        />
                      )}
                    </div>
                    <div className="border-t border-gray-200 px-3 py-2 text-sm text-gray-600">
                      {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                        <VariantMetricSummary metrics={bboxMetrics.left} selectedMetric={metric} />
                      ) : selectedBbox && bboxMetricsLoading ? (
                        <span>Loading metrics…</span>
                      ) : (
                        <span>Click a feature to see local metrics for this variant.</span>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-0">
                    <div className="border-b border-gray-200 px-3 py-2">
                      <p className="text-sm font-medium text-gray-900">{data.right.label}</p>
                      <p className="text-xs text-gray-500">
                        {showSimilarityHeatmaps ? 'Feature-local similarity' : 'Global overlay'} · {percentileCopy}
                      </p>
                    </div>
                    <div className="relative aspect-square w-full overflow-hidden bg-gray-950">
                      <CompareCanvas
                        imageSrc={showSimilarityHeatmaps ? originalUrl : rightUrl!}
                        imageAlt={showSimilarityHeatmaps ? `${data.right.label} similarity heatmap` : `${data.right.label} attention`}
                        overlaySrc={showSimilarityHeatmaps ? rightSimilarityHeatmapUrl : null}
                        overlayAlt={`${data.right.label} similarity overlay`}
                      />
                      {similarityLoading && selectedBbox && (
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        </div>
                      )}
                      {showBboxes && bboxes.length > 0 && (
                        <InteractiveBboxOverlay
                          bboxes={bboxes}
                          selectedIndex={selectedBboxIndex}
                          onBboxClick={(_bbox, index) => setSelectedBboxIndex(selectedBboxIndex === index ? null : index)}
                        />
                      )}
                    </div>
                    <div className="border-t border-gray-200 px-3 py-2 text-sm text-gray-600">
                      {selectedBbox && bboxMetrics && !bboxMetricsLoading ? (
                        <VariantMetricSummary metrics={bboxMetrics.right} selectedMetric={metric} />
                      ) : selectedBbox && bboxMetricsLoading ? (
                        <span>Loading metrics…</span>
                      ) : (
                        <span>Click a feature to see local metrics for this variant.</span>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
              {data.linearProbeInvolved && (
                <p className="text-center text-xs text-amber-700">
                  Linear probe is the no-backbone-change baseline, so attention differences are expected to stay small.
                </p>
              )}
              {similarityError && selectedBbox && (
                <p className="text-center text-xs text-amber-700">
                  Similarity heatmaps are unavailable for this selection. Run feature-cache generation for both compared variants to enable bbox-local inspection.
                </p>
              )}
            </>
          )}
        </>
      )}

      {viewMode === 'shift-map' && shiftSupported && (
        <>
          <div className="rounded-lg border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-sky-900">
            This map always shows {comparedVariantDetails?.label ?? 'Adapted variant'} minus{' '}
            {frozenVariant?.label ?? 'Frozen'} using cached numeric heatmaps. Red means more attention
            after fine-tuning, blue means less. The selected metric ({metricMetadata.optionLabel}) and
            percentile stay visible for the Q2 summary and feature-local delta cards below, but they do
            not change this shift map. The photo background is shown in grayscale and dimmed so the
            signed shift colors stand out.
          </div>
          <Card>
            <CardContent className="p-0">
              <div className="border-b border-gray-200 px-3 py-2">
                <p className="text-sm font-medium text-gray-900">Attention shift map</p>
                <p className="text-xs text-gray-500">
                  {comparedVariantDetails?.label ?? 'Adapted variant'} minus {frozenVariant?.label ?? 'Frozen'} · raw cached heatmaps on a dimmed grayscale photo
                </p>
              </div>
              {shiftLoading ? (
                <div className="p-4 text-sm text-gray-600">Loading attention shift map...</div>
              ) : shiftError ? (
                <div className="m-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                  Failed to load the attention shift map for this comparison.
                </div>
              ) : shiftData && !shiftData.available ? (
                <div className="m-4 rounded-lg border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-800">
                  <p className="font-medium">Attention shift unavailable</p>
                  <p className="mt-1">{shiftData.reason}</p>
                </div>
              ) : (
                <>
                  <div className="relative aspect-square w-full overflow-hidden bg-gray-950">
                    <CompareCanvas
                      imageSrc={originalUrl}
                      imageAlt={`${comparedVariantDetails?.label ?? 'Adapted variant'} minus ${frozenVariant?.label ?? 'Frozen'} shift map`}
                      imageClassName="grayscale brightness-50 contrast-75"
                      overlaySrc={shiftHeatmapUrl}
                      overlayAlt="Attention shift heatmap"
                    />
                    {showBboxes && bboxes.length > 0 && (
                      <InteractiveBboxOverlay
                        bboxes={bboxes}
                        selectedIndex={selectedBboxIndex}
                        onBboxClick={(_bbox, index) => setSelectedBboxIndex((current) => (current === index ? null : index))}
                      />
                    )}
                  </div>
                  <div className="flex flex-col gap-2 border-t border-gray-200 px-3 py-3 text-xs text-gray-600 sm:flex-row sm:items-center sm:justify-between">
                    <div className="space-y-1">
                      <p>
                        Shift = {shiftData?.operation ?? 'compared_variant_attention - frozen_attention'}
                      </p>
                      <p>
                        Range {shiftData?.min_value?.toFixed(3) ?? 'n/a'} to {shiftData?.max_value?.toFixed(3) ?? 'n/a'}
                        {shiftData?.max_abs_value !== null && shiftData?.max_abs_value !== undefined
                          ? ` · symmetric color scale ±${shiftData.max_abs_value.toFixed(3)}`
                          : ''}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span>Less attention</span>
                      <img src={shiftLegendUrl} alt="Attention shift scale" className="h-4 rounded" />
                      <span>More attention</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
          {data.linearProbeInvolved && (
            <p className="text-center text-xs text-amber-700">
              Linear probe is the no-backbone-change baseline, so attention differences are expected to stay small.
            </p>
          )}
        </>
      )}

      {viewMode === 'slider' && (
        <>
          {!sliderAvailable ? (
            <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
              <h4 className="font-medium text-yellow-800">Comparison overlay unavailable</h4>
              <p className="mt-1 text-sm text-yellow-700">{data.note}</p>
              <p className="mt-1 text-sm text-yellow-700">
                Generate cached overlays for both selected variants to enable the slider view.
              </p>
            </div>
          ) : (
            <>
              <div className="flex justify-between text-sm text-gray-600">
                <span>{data.left.label}</span>
                <span>{data.right.label}</span>
              </div>
              <div className="rounded-lg border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm text-sky-900">
                Clicking a bounding box switches the slider from cached global overlays to bbox-conditioned
                similarity heatmaps. The selected metric is {metricMetadata.optionLabel.toLowerCase()}, and
                {metricMetadata.direction === 'higher' ? ' higher values are better.' : ' lower values are better.'}
              </div>
              <div className="relative mx-auto w-full max-w-3xl">
                <ReactCompareSlider
                  itemOne={
                    <CompareCanvas
                      imageSrc={showSimilarityHeatmaps ? originalUrl : leftUrl!}
                      imageAlt={showSimilarityHeatmaps ? `${data.left.label} similarity heatmap` : `${data.left.label} attention`}
                      overlaySrc={showSimilarityHeatmaps ? leftSimilarityHeatmapUrl : null}
                      overlayAlt={`${data.left.label} similarity overlay`}
                    />
                  }
                  itemTwo={
                    <CompareCanvas
                      imageSrc={showSimilarityHeatmaps ? originalUrl : rightUrl!}
                      imageAlt={showSimilarityHeatmaps ? `${data.right.label} similarity heatmap` : `${data.right.label} attention`}
                      overlaySrc={showSimilarityHeatmaps ? rightSimilarityHeatmapUrl : null}
                      overlayAlt={`${data.right.label} similarity overlay`}
                    />
                  }
                  className="aspect-square overflow-hidden rounded-lg"
                  position={50}
                />
                {similarityLoading && selectedBbox && (
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20">
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  </div>
                )}
                {showBboxes && bboxes.length > 0 && (
                  <InteractiveBboxOverlay
                    bboxes={bboxes}
                    selectedIndex={selectedBboxIndex}
                    onBboxClick={(_bbox, index) => setSelectedBboxIndex((current) => (current === index ? null : index))}
                  />
                )}
              </div>
              <p className="text-center text-xs text-gray-500">
                {showSimilarityHeatmaps
                  ? 'Drag slider to compare bbox-conditioned similarity heatmaps'
                  : `Drag slider to compare ${data.left.label.toLowerCase()} vs ${data.right.label.toLowerCase()} attention${showBboxes ? ' with annotated boxes' : ''}`}
              </p>
              {data.linearProbeInvolved && (
                <p className="text-center text-xs text-amber-700">
                  Linear probe is the no-backbone-change baseline, so attention differences are expected to stay small.
                </p>
              )}
              {similarityError && selectedBbox && (
                <p className="text-center text-xs text-amber-700">
                  Similarity heatmaps are unavailable for this selection, so the view stays on the global overlays. Run
                  feature-cache generation for both compared variants to enable bbox-local inspection.
                </p>
              )}
            </>
          )}
        </>
      )}

      {featureOptions.length > 0 && (
        <div className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-gray-900">Feature-local metrics</p>
              <p className="text-xs text-gray-600">
                Click a box on the image or choose a feature below to compare local alignment for the selected metric.
              </p>
            </div>
            {selectedBboxIndex !== null && (
              <button
                type="button"
                onClick={() => setSelectedBboxIndex(null)}
                className="text-xs font-medium text-primary-600 hover:underline"
              >
                Clear selection
              </button>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            {featureOptions.map((feature) => {
              const selected = selectedBboxIndex === feature.index;
              return (
                <button
                  key={`${feature.label}-${feature.index}`}
                  type="button"
                  onClick={() => setSelectedBboxIndex((current) => (current === feature.index ? null : feature.index))}
                  className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                    selected
                      ? 'border-primary-600 bg-primary-50 text-primary-700'
                      : 'border-gray-200 bg-gray-50 text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {feature.label}
                </button>
              );
            })}
          </div>

          {selectedBbox && (
            <div className="grid gap-3 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                <p className="text-sm font-medium text-gray-900">
                  Selected feature: {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
                </p>
                <p className="mt-1 text-xs text-gray-600">
                  Bounding box #{selectedBboxIndex! + 1} at layer {effectiveLayer}. {viewMode === 'shift-map'
                    ? `The main view stays on the frozen-vs-variant shift map while the lower panels keep ${metricMetadata.optionLabel} and the full local metric bundle visible for context.`
                    : viewMode === 'slider'
                      ? `The slider now uses this expert region as the similarity query. The highlighted delta card tracks ${metricMetadata.optionLabel}, while the detailed panels keep all local metrics visible for context.`
                      : `The comparison panels now use this expert region as the similarity query. The highlighted delta card tracks ${metricMetadata.optionLabel}, while the detailed panels keep all local metrics visible for context.`}
                </p>
              </div>

              <div className={`rounded-lg border p-3 ${deltaTone}`}>
                <p className="text-xs font-semibold uppercase tracking-wide">Feature-local delta</p>
                {bboxMetricsLoading ? (
                  <p className="mt-2 text-sm">Loading feature metrics...</p>
                ) : bboxMetricsError ? (
                  <p className="mt-2 text-sm">
                    Failed to load bbox metrics for this comparison. Check that both compared metrics are cached.
                  </p>
                ) : bboxMetrics ? (
                  <>
                    <p className="mt-2 text-2xl font-semibold">
                      {formatMetricValue(metric, deltaValue, { signed: true })} {metricMetadata.shortLabel}
                    </p>
                    <p className="mt-1 text-xs">
                      {metricMetadata.direction === 'higher'
                        ? 'Positive means the right variant scores higher for this feature.'
                        : 'Negative means the right variant scores lower for this feature.'}
                    </p>
                    <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
                      <p>IoU Δ: {formatMetricValue('iou', deltaIoU, { signed: true })}</p>
                      <p>Coverage Δ: {formatMetricValue('coverage', deltaCoverage, { signed: true })}</p>
                      <p>MSE Δ: {formatMetricValue('mse', deltaMse, { signed: true })}</p>
                      <p>KL Δ: {formatMetricValue('kl', deltaKl, { signed: true })}</p>
                      <p>EMD Δ: {formatMetricValue('emd', deltaEmd, { signed: true })}</p>
                    </div>
                  </>
                ) : (
                  <p className="mt-2 text-sm">Select a feature to see local change.</p>
                )}
              </div>
            </div>
          )}

          {selectedBbox && showSimilarityHeatmaps && (
            <div className="flex items-center justify-between gap-4 rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600">
              <div className="space-y-1">
                <p className="font-medium text-gray-900">{data.left.label} similarity</p>
                <p>
                  {leftSimilarityStats
                    ? `min ${leftSimilarityStats.min.toFixed(2)} | max ${leftSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span>Low similarity</span>
                <img src={legendUrl} alt="Similarity scale" className="h-4 rounded" />
                <span>High similarity</span>
              </div>
              <div className="space-y-1 text-right">
                <p className="font-medium text-gray-900">{data.right.label} similarity</p>
                <p>
                  {rightSimilarityStats
                    ? `min ${rightSimilarityStats.min.toFixed(2)} | max ${rightSimilarityStats.max.toFixed(2)}`
                    : 'No stats available'}
                </p>
              </div>
            </div>
          )}

          {selectedBbox && bboxMetrics && !bboxMetricsLoading && !bboxMetricsError && (
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{data.left.label}</p>
                <div className="mt-2">
                  <VariantMetricSummary metrics={bboxMetrics.left} selectedMetric={metric} />
                </div>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{data.right.label}</p>
                <div className="mt-2">
                  <VariantMetricSummary metrics={bboxMetrics.right} selectedMetric={metric} />
                </div>
              </div>
            </div>
          )}

          {!selectedBbox && (
            <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-3 text-sm text-gray-600">
              Global overlays often look similar. Selecting a feature switches the view to bbox-conditioned
              similarity heatmaps and feature-local metrics so you can compare the same architectural cue across
              the chosen variants.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export { VariantCompare as FrozenVsFinetuned };
