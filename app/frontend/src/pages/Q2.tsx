import { useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useModels } from '../hooks/useAttention';
import { useQ2ImageDeltas, useQ2Summary } from '../hooks/useMetrics';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import {
  ANALYSIS_METRIC_METADATA,
  ANALYSIS_METRIC_OPTIONS,
  formatMetricValue,
  isAnalysisMetric,
} from '../constants/metricMetadata';
import type { AnalysisMetric, CompareVariantId, Q2SummaryRow, Q2StrategyComparison } from '../types';

const STRATEGY_OPTIONS = [
  { value: 'all', label: 'All Strategies' },
  { value: 'linear_probe', label: 'Linear Probe' },
  { value: 'lora', label: 'LoRA' },
  { value: 'full', label: 'Full Fine-tune' },
];

const PERCENTILES = [90, 80, 70, 60, 50];

function getMetricSortValue(metric: AnalysisMetric, row: Q2SummaryRow) {
  const direction = ANALYSIS_METRIC_METADATA[metric].direction;
  return direction === 'higher' ? -row.mean_delta : row.mean_delta;
}

function getPairwiseSortValue(metric: AnalysisMetric, row: Q2StrategyComparison) {
  const direction = ANALYSIS_METRIC_METADATA[metric].direction;
  return direction === 'higher' ? -row.mean_delta_difference : row.mean_delta_difference;
}

export function Q2Page() {
  const [searchParams, setSearchParams] = useSearchParams();
  const metricParam = searchParams.get('metric');
  const metric: AnalysisMetric = isAnalysisMetric(metricParam) ? metricParam : 'iou';
  const percentile = Number(searchParams.get('percentile') || '90');
  const selectedModel = searchParams.get('model') || 'all';
  const selectedStrategy = searchParams.get('strategy') || 'all';

  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const thresholdFree = metricMetadata.thresholdFree;

  const { data: modelsData } = useModels();
  const modelOptions = useMemo(() => {
    const models = (modelsData?.models || []).filter((modelName) => modelName !== 'resnet50');
    return [{ value: 'all', label: 'All Models' }, ...models.map((modelName) => ({ value: modelName, label: modelName }))];
  }, [modelsData]);

  const { data, isLoading, error } = useQ2Summary(
    metric,
    percentile,
    selectedModel === 'all' ? undefined : selectedModel,
    selectedStrategy === 'all' ? undefined : selectedStrategy,
  );

  const rows = useMemo(
    () => [...(data?.rows || [])].sort((left, right) => getMetricSortValue(metric, left) - getMetricSortValue(metric, right)),
    [data?.rows, metric],
  );
  const pairwise = useMemo(
    () => [...(data?.strategy_comparisons || [])].sort(
      (left, right) => getPairwiseSortValue(metric, left) - getPairwiseSortValue(metric, right),
    ),
    [data?.strategy_comparisons, metric],
  );
  const analyzedLayer = data?.analyzed_layer ?? 11;

  const buildSearchParams = (overrides?: Record<string, string>) => {
    const next = new URLSearchParams(searchParams);
    next.set('metric', metric);
    next.set('percentile', String(percentile));
    next.set('model', selectedModel);
    next.set('strategy', selectedStrategy);
    if (overrides) {
      Object.entries(overrides).forEach(([key, value]) => next.set(key, value));
    }
    return next;
  };

  const compareModel = selectedModel === 'all' ? 'dinov2' : selectedModel;
  const compareRightVariant: CompareVariantId =
    selectedStrategy === 'linear_probe' || selectedStrategy === 'lora' || selectedStrategy === 'full'
      ? selectedStrategy
      : 'full';
  const q2ImageDeltaModel = selectedModel === 'all' ? undefined : selectedModel;
  const q2ImageDeltaStrategy =
    selectedStrategy === 'linear_probe' || selectedStrategy === 'lora' || selectedStrategy === 'full'
      ? selectedStrategy
      : undefined;
  const imageDeltaSelectionMessage = !q2ImageDeltaModel && !q2ImageDeltaStrategy
    ? 'Select a specific model and strategy to inspect per-image deltas.'
    : !q2ImageDeltaModel
      ? 'Select a specific model to inspect per-image deltas.'
      : !q2ImageDeltaStrategy
        ? 'Select a specific strategy to inspect per-image deltas.'
        : null;
  const { data: imageDeltaData, isLoading: isImageDeltaLoading } = useQ2ImageDeltas(
    q2ImageDeltaModel,
    q2ImageDeltaStrategy,
    percentile,
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Q2 Strategy-Aware Attention Shift</h1>
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="text-sm font-bold uppercase tracking-[0.14em] text-slate-600">
            Question it answers for Q2
          </div>
          <p className="mt-2 max-w-5xl text-base font-semibold leading-7 text-slate-900">
            How does attention change after adaptation to the style-classification task, and does the strategy matter?
          </p>
        </div>
      </div>

      <Card>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            <Select
              value={metric}
              onChange={(value) => setSearchParams(buildSearchParams({ metric: value }))}
              options={ANALYSIS_METRIC_OPTIONS}
              label="Metric"
            />
            <Select
              value={String(percentile)}
              onChange={(value) => setSearchParams(buildSearchParams({ percentile: value }))}
              options={PERCENTILES.map((p) => ({ value: String(p), label: `Top ${100 - p}%` }))}
              label="Percentile"
              disabled={thresholdFree}
            />
            <Select
              value={selectedModel}
              onChange={(value) => setSearchParams(buildSearchParams({ model: value }))}
              options={modelOptions}
              label="Model"
            />
            <Select
              value={selectedStrategy}
              onChange={(value) => setSearchParams(buildSearchParams({ strategy: value }))}
              options={STRATEGY_OPTIONS}
              label="Strategy"
            />
          </div>
          {thresholdFree && (
            <p className="mt-3 text-sm text-slate-600">
              {metricMetadata.optionLabel} is threshold-free, so percentile stays visible for consistency but only affects IoU-based analysis.
            </p>
          )}
          <p className="mt-3 text-sm text-slate-600">
            This Q2 summary was computed at layer {analyzedLayer}. The compare link opens that same layer so the image-level playback starts from the aggregate analysis layer.
          </p>
          {data?.experiment_id && (
            <p className="mt-2 text-sm text-slate-600">
              Active experiment: <span className="font-medium">{data.experiment_id}</span>
              {data.result_set_scope ? ` • ${data.result_set_scope}` : ''}
              {data.evaluation_image_count ? ` • ${data.evaluation_image_count} annotated images` : ''}
            </p>
          )}
          {data?.checkpoint_selection_rule && (
            <p className="mt-2 text-sm text-slate-600">
              Checkpoint rule: {data.checkpoint_selection_rule}
            </p>
          )}
        </CardContent>
      </Card>

      {thresholdFree && metricMetadata.infoBanner && (
        <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-900">
          {metricMetadata.infoBanner}
        </div>
      )}

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Model × Strategy Delta</h3>
            <Link
              to={`/compare?type=variants&model=${compareModel}&metric=${metric}&left_variant=frozen&right_variant=${compareRightVariant}&layer=${analyzedLayer}&percentile=${percentile}`}
              className="text-sm text-primary-600 hover:underline"
            >
              Open Variant Compare
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading && <div className="text-sm text-gray-500">Loading Q2 summary...</div>}
          {error && <div className="text-sm text-red-600">Failed to load Q2 summary.</div>}
          {!isLoading && !error && rows.length === 0 && (
            <div className="text-sm text-gray-500">No Q2 rows available for current filters.</div>
          )}
          {rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b text-left">
                    <th className="py-2 pr-3">Model</th>
                    <th className="py-2 pr-3">Strategy</th>
                    <th className="py-2 pr-3">Frozen mean</th>
                    <th className="py-2 pr-3">Fine-tuned mean</th>
                    <th className="py-2 pr-3">Delta</th>
                    <th className="py-2 pr-3">95% CI</th>
                    <th className="py-2 pr-3">Effect size</th>
                    <th className="py-2 pr-3">Images</th>
                    <th className="py-2">Significant</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={`${row.model_name}-${row.strategy_id}-${row.metric}-${row.percentile ?? 'all'}`} className="border-b last:border-0">
                      <td className="py-2 pr-3">{row.model_name}</td>
                      <td className="py-2 pr-3">{row.strategy_id}</td>
                      <td className="py-2 pr-3">{formatMetricValue(metric, row.frozen_mean)}</td>
                      <td className="py-2 pr-3">{formatMetricValue(metric, row.finetuned_mean)}</td>
                      <td className="py-2 pr-3 font-medium">{formatMetricValue(metric, row.mean_delta, { signed: true })}</td>
                      <td className="py-2 pr-3">
                        [{formatMetricValue(metric, row.delta_ci_lower, { signed: true })}, {formatMetricValue(metric, row.delta_ci_upper, { signed: true })}]
                      </td>
                      <td className="py-2 pr-3">{row.cohens_d.toFixed(3)}</td>
                      <td className="py-2 pr-3">{row.num_images}</td>
                      <td className="py-2">{row.significant ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="font-semibold">Image-Level Delta Drilldown (IoU)</h3>
        </CardHeader>
        <CardContent>
          {imageDeltaSelectionMessage && (
            <p className="text-sm text-gray-500">
              {imageDeltaSelectionMessage}
            </p>
          )}
          {!imageDeltaSelectionMessage && isImageDeltaLoading && (
            <p className="text-sm text-gray-500">Loading image-level deltas...</p>
          )}
          {!imageDeltaSelectionMessage && !isImageDeltaLoading && !imageDeltaData && (
            <p className="text-sm text-gray-500">No image-level deltas available for this selection.</p>
          )}
          {imageDeltaData && (
            <div className="space-y-4">
              <p className="text-sm text-slate-600">
                Mean ΔIoU {formatMetricValue('iou', imageDeltaData.mean_delta_iou ?? 0, { signed: true })} over{' '}
                {imageDeltaData.num_images ?? 0} images.
              </p>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div>
                  <h4 className="mb-2 text-sm font-medium text-green-700">Top Improved</h4>
                  <div className="space-y-2 text-sm">
                    {imageDeltaData.top_positive.map((entry) => (
                      <div key={`pos-${entry.image_id}`} className="rounded border border-green-100 p-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-xs">{entry.image_id}</span>
                          <span className="font-medium text-green-700">
                            {formatMetricValue('iou', entry.delta_iou, { signed: true })}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-slate-600">{entry.style_names.join(', ') || 'Unknown style'}</div>
                        <div className="mt-1 flex gap-3 text-xs">
                          <Link to={`/image/${entry.image_id}`} className="text-primary-600 hover:underline">Image Detail</Link>
                          <Link
                            to={`/compare?type=variants&image_id=${entry.image_id}&model=${q2ImageDeltaModel}&metric=iou&left_variant=frozen&right_variant=${q2ImageDeltaStrategy}&layer=${analyzedLayer}&percentile=${percentile}`}
                            className="text-primary-600 hover:underline"
                          >
                            Compare
                          </Link>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="mb-2 text-sm font-medium text-rose-700">Most Regressed</h4>
                  <div className="space-y-2 text-sm">
                    {imageDeltaData.top_negative.map((entry) => (
                      <div key={`neg-${entry.image_id}`} className="rounded border border-rose-100 p-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-xs">{entry.image_id}</span>
                          <span className="font-medium text-rose-700">
                            {formatMetricValue('iou', entry.delta_iou, { signed: true })}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-slate-600">{entry.style_names.join(', ') || 'Unknown style'}</div>
                        <div className="mt-1 flex gap-3 text-xs">
                          <Link to={`/image/${entry.image_id}`} className="text-primary-600 hover:underline">Image Detail</Link>
                          <Link
                            to={`/compare?type=variants&image_id=${entry.image_id}&model=${q2ImageDeltaModel}&metric=iou&left_variant=frozen&right_variant=${q2ImageDeltaStrategy}&layer=${analyzedLayer}&percentile=${percentile}`}
                            className="text-primary-600 hover:underline"
                          >
                            Compare
                          </Link>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="font-semibold">Cross-Strategy Paired Comparisons</h3>
        </CardHeader>
        <CardContent>
          {pairwise.length === 0 && (
            <div className="text-sm text-gray-500">No cross-strategy comparisons available.</div>
          )}
          {pairwise.length > 0 && (
            <div className="space-y-2 text-sm">
              {pairwise.map((row, index) => (
                <div key={`${row.model_name}-${row.strategy_a}-${row.strategy_b}-${row.percentile ?? 'all'}-${index}`} className="flex justify-between border-b pb-1">
                  <span>{row.model_name} - {row.strategy_a} vs {row.strategy_b}</span>
                  <span className="font-medium">
                    Delta difference {formatMetricValue(metric, row.mean_delta_difference, { signed: true })} ({row.significant ? 'sig' : 'ns'})
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
