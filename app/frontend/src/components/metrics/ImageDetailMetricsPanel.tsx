import { useState } from 'react';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Tooltip } from '../ui/Tooltip';
import { useImageLayerProgression } from '../../hooks/useAttention';
import type { ImageDetailMode, ImageMetricDescriptor } from '../../types';
import { GLOSSARY } from '../../constants/glossary';
import { LayerMetricsChart } from './LayerMetricsChart';
import { getMetricDirectionLabel } from './layerChartUtils';

const METRIC_COLORS: Record<string, string> = {
  iou: '#2563eb',
  coverage: '#4d7c0f',
  mse: '#dc2626',
  kl: '#7c3aed',
  emd: '#111827',
};

const DEFAULT_TOGGLE_THEME: MetricToggleTheme = {
  accentColor: '#2563eb',
  checkedClassName: 'border-slate-300 bg-gradient-to-r from-slate-50 via-white to-slate-100 text-slate-800 shadow-sm shadow-slate-200/80',
};

const METRIC_GLOSSARY_KEYS: Record<string, keyof typeof GLOSSARY> = {
  iou: 'IoU Score',
  coverage: 'Coverage',
  mse: 'MSE',
  kl: 'KL',
  emd: 'EMD',
};

const METRIC_TOGGLE_THEMES: Record<string, MetricToggleTheme> = {
  iou: {
    accentColor: METRIC_COLORS.iou,
    checkedClassName: 'border-blue-300 bg-gradient-to-r from-blue-50 via-sky-50 to-blue-100 text-blue-800 shadow-sm shadow-blue-100/80',
  },
  coverage: {
    accentColor: METRIC_COLORS.coverage,
    checkedClassName: 'border-lime-300 bg-gradient-to-r from-lime-50 via-yellow-50 to-lime-100 text-lime-900 shadow-sm shadow-lime-100/80',
  },
  mse: {
    accentColor: METRIC_COLORS.mse,
    checkedClassName: 'border-red-300 bg-gradient-to-r from-rose-50 via-red-50 to-red-100 text-red-800 shadow-sm shadow-red-100/80',
  },
  kl: {
    accentColor: METRIC_COLORS.kl,
    checkedClassName: 'border-violet-300 bg-gradient-to-r from-violet-50 via-fuchsia-50 to-indigo-100 text-violet-900 shadow-sm shadow-violet-100/80',
  },
  emd: {
    accentColor: METRIC_COLORS.emd,
    checkedClassName: 'border-slate-400 bg-gradient-to-r from-slate-50 via-white to-slate-100 text-slate-900 shadow-sm shadow-slate-200/80',
  },
};

interface MetricToggleTheme {
  accentColor: string;
  checkedClassName: string;
}

interface ImageDetailMetricsPanelProps {
  imageId: string;
  model: string;
  percentile: number;
  method: string;
  mode: ImageDetailMode;
  bboxSelectionDrivesOverlay?: boolean;
  selectedBboxIndex: number | null;
  currentLayer: number;
  isPlaying: boolean;
  enabled?: boolean;
}

export function ImageDetailMetricsPanel({
  imageId,
  model,
  percentile,
  method,
  mode,
  bboxSelectionDrivesOverlay = false,
  selectedBboxIndex,
  currentLayer,
  isPlaying,
  enabled = true,
}: ImageDetailMetricsPanelProps) {
  const { data, isLoading, error } = useImageLayerProgression(
    imageId,
    model,
    percentile,
    method,
    selectedBboxIndex,
    enabled,
  );
  const [metricVisibilityOverrides, setMetricVisibilityOverrides] = useState<Record<string, boolean>>({});
  const metrics = data?.metrics ?? [];
  const visibleMetricKeys = metrics
    .filter((metric) => metricVisibilityOverrides[metric.key] ?? metric.default_enabled)
    .map((metric) => metric.key);
  const semanticsCopy = buildSemanticsCopy(metrics);
  const modeCopy = buildModeCopy(mode, bboxSelectionDrivesOverlay, selectedBboxIndex !== null);

  return (
    <div data-testid="metrics-panel">
      <Card>
        <CardHeader className="space-y-2">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="font-semibold">Metrics</h3>
              <p className="mt-1 text-sm text-gray-500">
                The chart tracks annotation-alignment metrics across layers for the current image,
                model, and method. It complements the active viewer instead of acting as a second
                image interpretation.
              </p>
            </div>
            <div
              className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700"
              data-testid="active-layer-indicator"
            >
              {isPlaying ? 'Playing' : 'Focused'}: Layer {currentLayer}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
            data-testid="metrics-mode-note"
          >
            {modeCopy}
          </div>

          {isLoading && <MetricsPanelSkeleton />}

          {!isLoading && error && (
            <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              Failed to load layer metrics.
            </div>
          )}

          {!isLoading && !error && !data && (
            <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              Metrics progression is not available for this image selection.
            </div>
          )}

          {!isLoading && !error && data && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium text-slate-900">
                  Showing {data.selection.mode === 'bbox' ? 'bbox metrics' : 'union metrics'}
                </span>
                {data.selection.bbox_label && (
                  <span className="rounded-full bg-green-50 px-2 py-1 text-xs font-medium text-green-700">
                    {data.selection.bbox_label}
                  </span>
                )}
              </div>

              <div className="space-y-2" data-testid="metric-toggle-group">
                <div className="text-sm font-medium text-slate-800">Visible metrics</div>
                <div className="flex flex-wrap gap-2">
                  {data.metrics.map((metric) => {
                    const checked = metricVisibilityOverrides[metric.key] ?? metric.default_enabled;
                    const label = `${metric.label} (${getMetricDirectionLabel(metric.direction)})`;
                    const toggleTheme = getMetricToggleTheme(metric.key);
                    return (
                      <Tooltip
                        key={metric.key}
                        content={getMetricTooltipContent(metric.key)}
                        align="left"
                        width={320}
                      >
                        <label
                          data-testid={`metric-toggle-${metric.key}`}
                          data-selected={checked ? 'true' : 'false'}
                          className={`flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition-all ${
                            checked
                              ? toggleTheme.checkedClassName
                              : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                          }`}
                        >
                          <input
                            type="checkbox"
                            className="h-4 w-4"
                            style={{ accentColor: toggleTheme.accentColor }}
                            checked={checked}
                            onChange={(event) => {
                              setMetricVisibilityOverrides((previous) => ({
                                ...previous,
                                [metric.key]: event.target.checked,
                              }));
                            }}
                          />
                          <span
                            aria-hidden="true"
                            className={`h-2.5 w-2.5 rounded-full ${checked ? 'opacity-100' : 'opacity-70'}`}
                            style={{ backgroundColor: toggleTheme.accentColor }}
                          />
                          <span>{label}</span>
                        </label>
                      </Tooltip>
                    );
                  })}
                </div>
              </div>

              {visibleMetricKeys.length > 0 ? (
                <>
                  <div
                    className="text-xs font-medium uppercase tracking-[0.12em] text-slate-500"
                    data-testid="chart-reveal-status"
                  >
                    {isPlaying ? `Revealing layers 0-${currentLayer}` : 'Showing full layer history'}
                  </div>
                  <LayerMetricsChart
                    layers={data.layers}
                    metrics={data.metrics}
                    visibleMetricKeys={visibleMetricKeys}
                    currentLayer={currentLayer}
                    isPlaying={isPlaying}
                    metricColors={METRIC_COLORS}
                  />
                </>
              ) : (
                <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                  Select at least one metric to render the chart.
                </div>
              )}

              {semanticsCopy && (
                <div
                  className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
                  data-testid="metrics-semantics-note"
                >
                  {semanticsCopy}
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function MetricsPanelSkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      <div className="h-10 rounded bg-slate-100" />
      <div className="h-80 rounded bg-slate-100" />
      <div className="h-12 rounded bg-slate-100" />
    </div>
  );
}

function joinMetricLabels(metrics: ImageMetricDescriptor[]): string {
  const labels = metrics.map((metric) => metric.label);

  if (labels.length === 1) {
    return labels[0];
  }

  if (labels.length === 2) {
    return `${labels[0]} and ${labels[1]}`;
  }

  return `${labels.slice(0, -1).join(', ')}, and ${labels[labels.length - 1]}`;
}

function buildSemanticsCopy(metrics: ImageMetricDescriptor[]) {
  if (!metrics.length) {
    return null;
  }

  const percentileSensitive = metrics.filter((metric) => metric.percentile_dependent);
  const thresholdFree = metrics.filter((metric) => !metric.percentile_dependent);

  if (!percentileSensitive.length || !thresholdFree.length) {
    return null;
  }

  return (
    <>
      Percentile changes update {joinMetricLabels(percentileSensitive)}.{' '}
      {joinMetricLabels(thresholdFree)} remain fixed for the same image, model, and method
      because they are threshold-free.
    </>
  );
}

function getMetricToggleTheme(metricKey: string): MetricToggleTheme {
  return METRIC_TOGGLE_THEMES[metricKey] ?? {
    ...DEFAULT_TOGGLE_THEME,
    accentColor: METRIC_COLORS[metricKey] ?? DEFAULT_TOGGLE_THEME.accentColor,
  };
}

function getMetricTooltipContent(metricKey: string): string {
  const glossaryKey = METRIC_GLOSSARY_KEYS[metricKey];
  return glossaryKey ? GLOSSARY[glossaryKey] : 'Metric details are not available.';
}

function buildModeCopy(mode: ImageDetailMode, bboxSelectionDrivesOverlay: boolean, hasSelectedBbox: boolean) {
  if (mode === 'feature_similarity') {
    return hasSelectedBbox
      ? 'Feature Similarity mode is active. The viewer is showing bbox-conditioned similarity, while this chart still provides contextual annotation-alignment metrics for the same selection.'
      : 'Feature Similarity mode is active. Select a bounding box to drive the viewer overlay; this chart remains contextual annotation-alignment data for the current image and method.';
  }

  if (bboxSelectionDrivesOverlay) {
    return hasSelectedBbox
      ? 'Head Attention mode is active. In Image Detail, selecting a bounding box swaps the viewer to a bbox-conditioned focused overlay and the chart to bbox-scoped metrics.'
      : 'Head Attention mode is active. The viewer is showing the global attention overlay. Select a bounding box to switch both the viewer and chart into bbox-scoped inspection.';
  }

  return hasSelectedBbox
    ? 'Head Attention mode is active. The viewer stays focused on attention, while the selected bounding box only scopes contextual chart labels and metrics.'
    : 'Head Attention mode is active. Use this chart to read layer-by-layer annotation alignment while the viewer stays focused on fused or per-head attention overlays.';
}
