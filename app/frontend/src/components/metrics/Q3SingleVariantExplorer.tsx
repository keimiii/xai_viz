import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { useHeadExemplars, useHeadFeatureMatrix, useHeadRanking } from '../../hooks/useMetrics';
import {
  ANALYSIS_METRIC_METADATA,
  formatMetricValue,
} from '../../constants/metricMetadata';
import { buildImageDetailQ3Href } from '../../constants/q3Routing';
import { Q3_DEFAULTS } from '../../constants/q3Scope';
import { Q3ExemplarPicker, type Q3ExemplarPickerRequest } from './Q3ExemplarPicker';
import type {
  AnalysisMetric,
  CompareVariantId,
  HeadExemplarCandidate,
  MetricDirection,
} from '../../types';

const ITEMS_PER_PAGE = 20;
const EXEMPLAR_LIMIT = 12;

interface HeatmapRange {
  min: number;
  max: number;
}

interface HeatmapCellPreview {
  head: number;
  featureLabel: number;
  featureName: string;
  score: number;
}

export interface Q3ExplorerFocus {
  head: number | null;
  featureLabel: number | null;
}

interface Q3SingleVariantExplorerProps {
  model: string;
  variant: CompareVariantId;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
  focus?: Q3ExplorerFocus;
  onActiveFocusChange?: (focus: Q3ExplorerFocus) => void;
  autoScrollSelectionIntoView?: boolean;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function interpolateColor(start: number, end: number, ratio: number): number {
  return Math.round(start + (end - start) * ratio);
}

function getHeatmapIntensity(score: number, range: HeatmapRange | null, direction: MetricDirection): number {
  if (!range || range.max === range.min) {
    return 1;
  }

  const normalized = clamp01((score - range.min) / (range.max - range.min));
  return direction === 'higher' ? normalized : 1 - normalized;
}

function getHeatmapCellStyle(score: number | null, range: HeatmapRange | null, direction: MetricDirection) {
  if (score === null) {
    return {
      backgroundColor: '#f8fafc',
      borderColor: '#e2e8f0',
    };
  }

  const intensity = getHeatmapIntensity(score, range, direction);
  return {
    backgroundColor: `rgb(${interpolateColor(240, 12, intensity)}, ${interpolateColor(249, 74, intensity)}, ${interpolateColor(255, 110, intensity)})`,
    borderColor: `rgb(${interpolateColor(186, 8, intensity)}, ${interpolateColor(230, 145, intensity)}, ${interpolateColor(253, 178, intensity)})`,
  };
}

function getHeatmapDotClass(score: number | null, range: HeatmapRange | null, direction: MetricDirection): string {
  if (score === null) {
    return 'border-slate-300 bg-slate-300';
  }

  const intensity = getHeatmapIntensity(score, range, direction);
  return intensity >= 0.55
    ? 'border-white/80 bg-white/80'
    : 'border-slate-500/30 bg-slate-700/20';
}

function getMetricTone(metric: AnalysisMetric, value: number | null): string {
  if (value === null || value === undefined) {
    return 'text-gray-400';
  }

  const direction = ANALYSIS_METRIC_METADATA[metric].direction;
  if (direction === 'higher') {
    if (value >= 0.6) return 'text-green-700 bg-green-50';
    if (value >= 0.4) return 'text-amber-700 bg-amber-50';
    return 'text-rose-700 bg-rose-50';
  }

  if (value <= 0.05) return 'text-green-700 bg-green-50';
  if (value <= 0.15) return 'text-amber-700 bg-amber-50';
  return 'text-rose-700 bg-rose-50';
}

function formatHoverReadout(metric: AnalysisMetric, preview: HeatmapCellPreview): string {
  const metricLabel = ANALYSIS_METRIC_METADATA[metric].shortLabel;
  return `${preview.featureName} · H${preview.head} · ${metricLabel}: ${formatMetricValue(metric, preview.score)}`;
}

export function Q3SingleVariantExplorer({
  model,
  variant,
  layer,
  metric,
  percentile,
  focus,
  onActiveFocusChange,
  autoScrollSelectionIntoView = true,
}: Q3SingleVariantExplorerProps) {
  const navigate = useNavigate();
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const effectivePercentile = metricMetadata.thresholdFree ? 90 : percentile;
  const [searchQuery, setSearchQuery] = useState('');
  const [showCount, setShowCount] = useState(ITEMS_PER_PAGE);
  const [drilldownRequest, setDrilldownRequest] = useState<Q3ExemplarPickerRequest | null>(null);
  const [hoveredCell, setHoveredCell] = useState<HeatmapCellPreview | null>(null);
  const heatmapContainerRef = useRef<HTMLDivElement | null>(null);
  const exemplarPanelRef = useRef<HTMLDivElement | null>(null);
  const heatmapHeaderRefs = useRef(new Map<number, HTMLTableCellElement>());
  const lastReportedFocusRef = useRef<string | null>(null);
  const isFocusControlled = focus !== undefined;

  const rankingQuery = useHeadRanking(model, layer, effectivePercentile, metric, variant);
  const matrixQuery = useHeadFeatureMatrix(model, layer, effectivePercentile, metric, variant);

  const localActiveDrilldownRequest = useMemo(() => {
    if (!drilldownRequest) {
      return null;
    }
    if (
      drilldownRequest.model !== model
      || drilldownRequest.variant !== variant
      || drilldownRequest.layer !== layer
      || drilldownRequest.metric !== metric
      || drilldownRequest.percentile !== effectivePercentile
    ) {
      return null;
    }
    return drilldownRequest;
  }, [drilldownRequest, model, variant, layer, metric, effectivePercentile]);

  const focusDerivedDrilldownRequest = useMemo(() => {
    if (!focus || focus.head === null) {
      return null;
    }

    if (focus.featureLabel !== null) {
      const feature = (matrixQuery.data?.features ?? []).find((entry) => entry.feature_label === focus.featureLabel);
      const headIndex = (matrixQuery.data?.heads ?? []).indexOf(focus.head);
      const score = feature && headIndex >= 0 ? feature.scores[headIndex] : null;

      if (feature && headIndex >= 0 && score !== null) {
        return {
          origin: 'feature' as const,
          model,
          variant,
          layer,
          head: focus.head,
          metric,
          percentile: effectivePercentile,
          featureLabel: focus.featureLabel,
          featureName: feature.feature_name,
          score,
        };
      }
    }

    const rankingEntry = (rankingQuery.data?.heads ?? []).find((entry) => entry.head === focus.head);
    if (!rankingEntry) {
      return null;
    }

    return {
      origin: 'ranking' as const,
      model,
      variant,
      layer,
      head: focus.head,
      metric,
      percentile: effectivePercentile,
      score: rankingEntry.mean_score,
    };
  }, [
    focus,
    matrixQuery.data?.features,
    matrixQuery.data?.heads,
    rankingQuery.data?.heads,
    model,
    variant,
    layer,
    metric,
    effectivePercentile,
  ]);

  const activeDrilldownRequest = isFocusControlled
    ? focusDerivedDrilldownRequest
    : localActiveDrilldownRequest;

  useEffect(() => {
    if (isFocusControlled || !onActiveFocusChange) {
      return;
    }

    const nextFocus = {
      head: localActiveDrilldownRequest?.head ?? null,
      featureLabel: localActiveDrilldownRequest?.origin === 'feature'
        ? (localActiveDrilldownRequest.featureLabel ?? null)
        : null,
    };
    const nextFocusKey = JSON.stringify(nextFocus);
    if (lastReportedFocusRef.current === nextFocusKey) {
      return;
    }

    lastReportedFocusRef.current = nextFocusKey;
    onActiveFocusChange(nextFocus);
  }, [isFocusControlled, localActiveDrilldownRequest, onActiveFocusChange]);

  const activeHoveredCell = useMemo(() => {
    if (!hoveredCell) {
      return null;
    }
    const feature = (matrixQuery.data?.features ?? []).find((entry) => entry.feature_label === hoveredCell.featureLabel);
    const headIndex = (matrixQuery.data?.heads ?? []).indexOf(hoveredCell.head);
    if (!feature || headIndex < 0) {
      return null;
    }
    const score = feature.scores[headIndex];
    if (score === null) {
      return null;
    }
    return {
      ...hoveredCell,
      featureName: feature.feature_name,
      score,
    };
  }, [hoveredCell, matrixQuery.data?.features, matrixQuery.data?.heads]);

  const exemplarQuery = useHeadExemplars(
    activeDrilldownRequest?.model ?? Q3_DEFAULTS.model,
    activeDrilldownRequest?.head ?? null,
    activeDrilldownRequest?.layer ?? Q3_DEFAULTS.layer,
    activeDrilldownRequest?.percentile ?? effectivePercentile,
    activeDrilldownRequest?.metric ?? metric,
    activeDrilldownRequest?.variant ?? variant,
    {
      featureLabel: activeDrilldownRequest?.featureLabel,
      limit: EXEMPLAR_LIMIT,
      enabled: activeDrilldownRequest !== null,
    },
  );

  const filteredFeatures = useMemo(() => {
    const features = matrixQuery.data?.features ?? [];
    if (!searchQuery.trim()) {
      return features;
    }
    const query = searchQuery.toLowerCase();
    return features.filter((feature) => feature.feature_name.toLowerCase().includes(query));
  }, [matrixQuery.data?.features, searchQuery]);

  const heatmapRange = useMemo<HeatmapRange | null>(() => {
    const scores = (matrixQuery.data?.features ?? []).flatMap((feature) =>
      feature.scores.filter((score): score is number => score !== null),
    );
    if (scores.length === 0) {
      return null;
    }
    return {
      min: Math.min(...scores),
      max: Math.max(...scores),
    };
  }, [matrixQuery.data?.features]);

  const visibleFeatures = filteredFeatures.slice(0, showCount);
  const hasMore = showCount < filteredFeatures.length;
  const heatmapDirection = matrixQuery.data?.direction ?? metricMetadata.direction;
  const selectedHead = activeDrilldownRequest?.head ?? null;
  const selectedFeatureLabel = activeDrilldownRequest?.origin === 'feature'
    ? (activeDrilldownRequest.featureLabel ?? null)
    : null;

  useEffect(() => {
    if (!autoScrollSelectionIntoView || !activeDrilldownRequest || selectedHead === null) {
      return undefined;
    }

    const animationFrameId = window.requestAnimationFrame(() => {
      const heatmapContainer = heatmapContainerRef.current;
      const selectedHeader = heatmapHeaderRefs.current.get(selectedHead);

      if (heatmapContainer && selectedHeader) {
        const containerRect = heatmapContainer.getBoundingClientRect();
        const headerRect = selectedHeader.getBoundingClientRect();
        const maxScrollLeft = Math.max(0, heatmapContainer.scrollWidth - heatmapContainer.clientWidth);
        const centeredScrollLeft = heatmapContainer.scrollLeft
          + (headerRect.left - containerRect.left)
          - (heatmapContainer.clientWidth / 2)
          + (headerRect.width / 2);
        const clampedScrollLeft = Math.min(Math.max(centeredScrollLeft, 0), maxScrollLeft);

        heatmapContainer.scrollTo({
          left: clampedScrollLeft,
          behavior: 'smooth',
        });
      }

      exemplarPanelRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    });

    return () => window.cancelAnimationFrame(animationFrameId);
  }, [activeDrilldownRequest, selectedHead, autoScrollSelectionIntoView]);

  const exemplarError = exemplarQuery.error instanceof Error ? exemplarQuery.error.message : null;

  const openRankingDrilldown = (head: number, score: number | null) => {
    const nextRequest: Q3ExemplarPickerRequest = {
      origin: 'ranking',
      model,
      variant,
      layer,
      head,
      metric,
      percentile: effectivePercentile,
      score,
    };

    if (isFocusControlled) {
      onActiveFocusChange?.({ head, featureLabel: null });
      return;
    }

    setDrilldownRequest(nextRequest);
  };

  const openFeatureDrilldown = (head: number, featureLabel: number, featureName: string, score: number) => {
    const nextRequest: Q3ExemplarPickerRequest = {
      origin: 'feature',
      model,
      variant,
      layer,
      head,
      metric,
      percentile: effectivePercentile,
      featureLabel,
      featureName,
      score,
    };

    if (isFocusControlled) {
      onActiveFocusChange?.({ head, featureLabel });
      return;
    }

    setDrilldownRequest(nextRequest);
  };

  const handleSelectCandidate = (candidate: HeadExemplarCandidate) => {
    if (!activeDrilldownRequest) {
      return;
    }

    navigate(
      buildImageDetailQ3Href(candidate.image_id, {
        model: activeDrilldownRequest.model,
        variant: activeDrilldownRequest.variant,
        layer: activeDrilldownRequest.layer,
        head: activeDrilldownRequest.head,
        metric: activeDrilldownRequest.metric,
        mode: 'head_attention',
        showBboxes: true,
        bboxIndex: activeDrilldownRequest.origin === 'feature' ? candidate.default_bbox_index : null,
        featureLabel: activeDrilldownRequest.featureLabel ?? null,
        featureName: activeDrilldownRequest.featureName ?? null,
      }),
    );
  };

  const hoverReadout = activeHoveredCell
    ? `Hover: ${formatHoverReadout(metric, activeHoveredCell)}`
    : 'Hover or focus a heatmap cell to inspect the exact score before loading representative images.';

  const selectionSummary = (() => {
    if (!activeDrilldownRequest) {
      return 'Select a head or click a heatmap cell to load representative images inline below the matrix.';
    }
    if (activeDrilldownRequest.origin === 'feature') {
      const featureLabel = activeDrilldownRequest.featureName ?? `feature ${activeDrilldownRequest.featureLabel}`;
      const scoreText = activeDrilldownRequest.score === null || activeDrilldownRequest.score === undefined
        ? ''
        : ` · ${metricMetadata.shortLabel} ${formatMetricValue(metric, activeDrilldownRequest.score)}`;
      return `Selected cell: ${featureLabel} · H${activeDrilldownRequest.head}${scoreText}.`;
    }
    const scoreText = activeDrilldownRequest.score === null || activeDrilldownRequest.score === undefined
      ? ''
      : ` · ${metricMetadata.shortLabel} ${formatMetricValue(metric, activeDrilldownRequest.score)}`;
    return `Selected head: Head ${activeDrilldownRequest.head}${scoreText}. Its heatmap column is highlighted and representative images appear below.`;
  })();

  const registerHeatmapHeaderRef = (head: number) => (node: HTMLTableCellElement | null) => {
    if (node) {
      heatmapHeaderRefs.current.set(head, node);
      return;
    }
    heatmapHeaderRefs.current.delete(head);
  };

  return (
    <>
      {metricMetadata.thresholdFree && (
        <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-900">
          {metricMetadata.infoBanner}
        </div>
      )}

      {!rankingQuery.data?.supported || !matrixQuery.data?.supported ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          {rankingQuery.data?.reason || matrixQuery.data?.reason || 'Per-head analysis is not available for this selection.'}
        </div>
      ) : (rankingQuery.data?.heads.length ?? 0) === 0 && (matrixQuery.data?.features.length ?? 0) === 0 ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
          No Q3 per-head rows are available for this selection yet. Run the per-head precompute commands first.
        </div>
      ) : (
        <>
          <div className="rounded-lg border border-sky-100 bg-sky-50 px-4 py-3 text-sm text-sky-900">
            Darker cells indicate better-performing head-feature pairs for the current metric. Hover for the exact value, then click to load representative images inline.
          </div>

          <div
            className="grid grid-cols-1 gap-6 xl:grid-cols-[24rem_minmax(0,1fr)]"
            data-testid="q3-single-variant-analysis"
          >
            <div className="rounded-lg border border-gray-200">
              <div className="border-b border-gray-100 px-4 py-3">
                <div className="font-medium text-gray-900">Head Ranking</div>
                <div className="text-xs text-gray-500">
                  {metricMetadata.direction === 'higher' ? 'Higher scores rank better.' : 'Lower scores rank better.'}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-gray-200 text-left text-xs text-gray-500">
                    <tr>
                      <th className="px-4 py-2 font-medium">Head</th>
                      <th className="px-4 py-2 font-medium text-right">{metricMetadata.shortLabel}</th>
                      <th className="px-4 py-2 font-medium text-right">Mean Rank</th>
                      <th className="px-4 py-2 font-medium text-right">Top-1</th>
                      <th className="px-4 py-2 font-medium text-right">Inspect</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {(rankingQuery.data?.heads ?? []).map((entry) => {
                      const isSelectedRankingHead = activeDrilldownRequest?.origin === 'ranking' && activeDrilldownRequest.head === entry.head;
                      return (
                        <tr
                          key={entry.head}
                          data-testid={`q3-head-ranking-row-${entry.head}`}
                          className={isSelectedRankingHead ? 'bg-primary-50/50' : undefined}
                        >
                          <td className="px-4 py-2 font-medium text-gray-900">Head {entry.head}</td>
                          <td className="px-4 py-2 text-right">
                            <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${getMetricTone(metric, entry.mean_score)}`}>
                              {formatMetricValue(metric, entry.mean_score)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right text-gray-600">{entry.mean_rank.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-gray-600">{entry.top1_count}</td>
                          <td className="px-4 py-2 text-right">
                            <button
                              type="button"
                              onClick={() => openRankingDrilldown(entry.head, entry.mean_score)}
                              className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                                isSelectedRankingHead
                                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                                  : 'border-primary-200 bg-white text-primary-700 hover:border-primary-300 hover:bg-primary-50'
                              }`}
                            >
                              Inspect exemplar
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="rounded-lg border border-gray-200">
              <div className="border-b border-gray-100 px-4 py-3">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <div className="font-medium text-gray-900">Head × Feature Heatmap</div>
                    <div className="text-xs text-gray-500">
                      Showing {visibleFeatures.length} of {filteredFeatures.length} feature types
                    </div>
                  </div>
                  <input
                    type="text"
                    placeholder="Search features..."
                    value={searchQuery}
                    onChange={(event) => {
                      setSearchQuery(event.target.value);
                      setShowCount(ITEMS_PER_PAGE);
                    }}
                    className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 lg:max-w-xs"
                  />
                </div>

                <div className="mt-4 grid gap-3">
                  <div
                    className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
                    data-testid="q3-heatmap-hover-readout"
                  >
                    {hoverReadout}
                  </div>
                  <div
                    className="rounded-lg border border-primary-100 bg-primary-50 px-3 py-2 text-sm text-primary-900"
                    data-testid="q3-heatmap-selection-summary"
                  >
                    {selectionSummary}
                  </div>
                  <div
                    className="flex items-center gap-2 text-xs text-slate-500"
                    data-testid="q3-heatmap-legend"
                  >
                    <span className="font-medium text-slate-700">Heatmap legend</span>
                    <span>lighter</span>
                    <span className="h-2 w-16 rounded-full bg-gradient-to-r from-slate-200 via-orange-200 to-orange-500" />
                    <span>darker</span>
                  </div>
                </div>
              </div>

              <div
                ref={heatmapContainerRef}
                className="overflow-x-auto"
                data-testid="q3-heatmap-scroll-container"
              >
                <table className="min-w-full border-separate border-spacing-0 text-sm">
                  <thead className="sticky top-0 z-10 bg-white">
                    <tr>
                      <th className="sticky left-0 z-20 border-b border-r border-gray-200 bg-white px-4 py-3 text-left text-xs font-medium uppercase tracking-wide text-gray-500">
                        Feature
                      </th>
                      {(matrixQuery.data?.heads ?? []).map((headIdx) => {
                        const isSelectedColumn = selectedHead === headIdx;
                        return (
                          <th
                            key={headIdx}
                            ref={registerHeatmapHeaderRef(headIdx)}
                            className={`border-b border-gray-200 px-3 py-3 text-center text-xs font-medium uppercase tracking-wide text-gray-500 ${isSelectedColumn ? 'bg-primary-50 text-primary-700' : 'bg-white'}`}
                            data-testid={`q3-heatmap-head-${headIdx}`}
                            data-selected-column={isSelectedColumn ? 'true' : 'false'}
                          >
                            H{headIdx}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleFeatures.map((feature) => {
                      const isSelectedRow = selectedFeatureLabel === feature.feature_label;
                      return (
                      <tr key={feature.feature_label} className="border-b border-gray-100">
                        <th
                          className={`sticky left-0 z-10 border-b border-r border-gray-200 px-4 py-3 text-left ${
                            isSelectedRow ? 'bg-sky-50' : 'bg-white'
                          }`}
                          data-testid={`q3-heatmap-feature-${feature.feature_label}`}
                        >
                          <div className="font-medium text-gray-900">{feature.feature_name}</div>
                          <div className="text-xs text-gray-500">{feature.bbox_count} annotations</div>
                        </th>
                        {feature.scores.map((score, headIndex) => {
                          const head = matrixQuery.data?.heads?.[headIndex] ?? headIndex;
                          const isSelectedColumn = selectedHead === head;
                          const isSelectedCell = isSelectedColumn && isSelectedRow;
                          const cellStyle = getHeatmapCellStyle(score, heatmapRange, heatmapDirection);
                          const dotClass = getHeatmapDotClass(score, heatmapRange, heatmapDirection);

                          return (
                            <td
                              key={`${feature.feature_label}-${head}`}
                              className={`border-b border-gray-100 px-2 py-2 text-center transition-colors ${
                                isSelectedColumn
                                  ? isSelectedCell
                                    ? 'border-l border-r border-sky-300 bg-sky-100/80'
                                    : 'border-l border-r border-sky-200 bg-sky-50/80'
                                  : ''
                              }`}
                              data-testid={`q3-heatmap-cell-wrapper-${feature.feature_label}-${head}`}
                              data-selected-column={isSelectedColumn ? 'true' : 'false'}
                            >
                              <button
                                type="button"
                                onMouseEnter={() => {
                                  if (score === null) {
                                    setHoveredCell(null);
                                    return;
                                  }
                                  setHoveredCell({
                                    head,
                                    featureLabel: feature.feature_label,
                                    featureName: feature.feature_name,
                                    score,
                                  });
                                }}
                                onFocus={() => {
                                  if (score === null) {
                                    setHoveredCell(null);
                                    return;
                                  }
                                  setHoveredCell({
                                    head,
                                    featureLabel: feature.feature_label,
                                    featureName: feature.feature_name,
                                    score,
                                  });
                                }}
                                onMouseLeave={() => setHoveredCell(null)}
                                onBlur={() => setHoveredCell(null)}
                                onClick={() => {
                                  if (score === null) {
                                    return;
                                  }
                                  openFeatureDrilldown(head, feature.feature_label, feature.feature_name, score);
                                }}
                                className={`flex h-12 w-16 items-center justify-center rounded-lg border transition focus:outline-none focus:ring-2 focus:ring-primary-500 ${isSelectedCell ? 'ring-2 ring-primary-500 ring-offset-1' : ''} ${score === null ? 'cursor-not-allowed opacity-60' : 'hover:scale-[1.02]'}`}
                                style={cellStyle}
                                data-testid={`q3-heatmap-cell-${feature.feature_label}-${head}`}
                                data-selected-cell={isSelectedCell ? 'true' : 'false'}
                                disabled={score === null}
                              >
                                <span className={`h-3.5 w-3.5 rounded-full border ${dotClass}`} />
                              </button>
                            </td>
                          );
                        })}
                      </tr>
                    );
                    })}
                  </tbody>
                </table>
              </div>

              {hasMore && (
                <div className="border-t border-gray-100 px-4 py-3">
                  <button
                    type="button"
                    onClick={() => setShowCount((current) => current + ITEMS_PER_PAGE)}
                    className="rounded-md border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
                  >
                    Show more features
                  </button>
                </div>
              )}
            </div>
          </div>

          <div ref={exemplarPanelRef} data-testid="q3-exemplar-panel-anchor">
            <Q3ExemplarPicker
              open={activeDrilldownRequest !== null}
              request={activeDrilldownRequest}
              data={exemplarQuery.data}
              isLoading={exemplarQuery.isLoading}
              error={exemplarError}
              onClose={() => {
                if (isFocusControlled) {
                  onActiveFocusChange?.({ head: null, featureLabel: null });
                  return;
                }
                setDrilldownRequest(null);
              }}
              onSelectCandidate={handleSelectCandidate}
            />
          </div>
        </>
      )}
    </>
  );
}
