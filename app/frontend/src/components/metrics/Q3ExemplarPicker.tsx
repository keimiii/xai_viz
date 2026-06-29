import { ANALYSIS_METRIC_METADATA, formatMetricValue } from '../../constants/metricMetadata';
import type { AnalysisMetric, CompareVariantId, HeadExemplarCandidate, HeadExemplarResponse } from '../../types';

interface Q3ExemplarPickerProps {
  open: boolean;
  request: Q3ExemplarPickerRequest | null;
  data?: HeadExemplarResponse;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
  onSelectCandidate: (candidate: HeadExemplarCandidate) => void;
}

export interface Q3ExemplarPickerRequest {
  origin: 'ranking' | 'feature';
  model: string;
  variant: CompareVariantId;
  layer: number;
  head: number;
  metric: AnalysisMetric;
  percentile: number;
  featureLabel?: number;
  featureName?: string | null;
  score?: number | null;
}

export function Q3ExemplarPicker({
  open,
  request,
  data,
  isLoading,
  error,
  onClose,
  onSelectCandidate,
}: Q3ExemplarPickerProps) {
  if (!open || !request) {
    return null;
  }

  const metricLabel = ANALYSIS_METRIC_METADATA[request.metric].shortLabel;
  const variantLabel = formatVariantLabel(request.variant);
  const title = request.origin === 'feature'
    ? 'Representative images for the selected heatmap cell'
    : `Representative images for Head ${request.head}`;

  return (
    <div
      className="rounded-2xl border border-slate-200 bg-white shadow-sm"
      data-testid="q3-exemplar-panel"
    >
      <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold text-slate-900">
            {title}
          </h3>
          <p className="text-sm text-slate-600">
            {request.origin === 'feature'
              ? `Inspect Head ${request.head} on one image that contains ${request.featureName ?? `feature ${request.featureLabel}`}.`
              : `Inspect Head ${request.head} on one representative image before drilling into Image Detail Q3.`}
          </p>
          <p className="text-xs text-slate-500">
            {request.model} · {variantLabel} · Layer {request.layer} · {metricLabel}
          </p>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="rounded-md border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
        >
          Clear selection
        </button>
      </div>

      <div className="px-5 py-4">
        {isLoading && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="animate-pulse rounded-xl border border-slate-200 p-4">
                <div className="aspect-square rounded-lg bg-slate-200" />
                <div className="mt-3 h-4 w-2/3 rounded bg-slate-200" />
                <div className="mt-2 h-3 w-1/2 rounded bg-slate-200" />
              </div>
            ))}
          </div>
        )}

        {!isLoading && error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            Failed to load exemplar candidates. {error}
          </div>
        )}

        {!isLoading && !error && data?.supported === false && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            {data.reason ?? 'Exemplar candidates are not available for this selection.'}
          </div>
        )}

        {!isLoading && !error && data?.supported !== false && (data?.candidates.length ?? 0) === 0 && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
            {data?.reason ?? 'No representative images are available for this selection yet.'}
          </div>
        )}

        {!isLoading && !error && data?.supported !== false && (data?.candidates.length ?? 0) > 0 && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {data?.candidates.map((candidate, index) => (
              <div
                key={`${candidate.image_id}-${index}`}
                className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
                data-testid={`q3-exemplar-card-${index}`}
              >
                <img
                  src={candidate.thumbnail_url}
                  alt={candidate.image_id}
                  className="aspect-square w-full object-cover"
                />
                <div className="space-y-3 p-4">
                  <div className="space-y-1">
                    <div className="font-medium text-slate-900">{candidate.image_id}</div>
                    <div className="text-sm text-slate-600">
                      {metricLabel}: {formatMetricValue(request.metric, candidate.score)}
                    </div>
                  </div>

                  {candidate.style_names.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {candidate.style_names.map((style) => (
                        <span
                          key={style}
                          className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700"
                        >
                          {style}
                        </span>
                      ))}
                    </div>
                  )}

                  {request.origin === 'feature' && (
                    <div className="rounded-lg border border-emerald-100 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
                      {candidate.matching_bbox_indices.length > 0
                        ? `${candidate.matching_bbox_indices.length} matching annotation${candidate.matching_bbox_indices.length === 1 ? '' : 's'} found for ${request.featureName ?? `feature ${request.featureLabel}`}.`
                        : `No matching annotation was found for ${request.featureName ?? `feature ${request.featureLabel}`}.`}
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => onSelectCandidate(candidate)}
                    data-testid={`q3-exemplar-open-${candidate.image_id}`}
                    className="w-full rounded-md bg-primary-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-700"
                  >
                    Open in Image Detail Q3
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function formatVariantLabel(variant: CompareVariantId): string {
  switch (variant) {
    case 'linear_probe':
      return 'Linear Probe (Control)';
    case 'lora':
      return 'LoRA';
    case 'full':
      return 'Full Fine-tune';
    default:
      return 'Frozen';
  }
}
