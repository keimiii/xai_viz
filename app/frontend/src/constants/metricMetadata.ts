import type { AnalysisMetric, CompareVariantId, DashboardMetric, MetricDirection } from '../types';

type AxisMode = 'unit' | 'auto';

export interface AnalysisMetricMetadata {
  optionLabel: string;
  shortLabel: string;
  chartLabel: string;
  direction: MetricDirection;
  thresholdFree: boolean;
  deltaDigits: number;
  hint: (percentile: number) => string;
  infoBanner?: string;
  axisMode: AxisMode;
}

export const ANALYSIS_METRIC_METADATA: Record<AnalysisMetric, AnalysisMetricMetadata> = {
  iou: {
    optionLabel: 'IoU',
    shortLabel: 'IoU',
    chartLabel: 'IoU',
    direction: 'higher',
    thresholdFree: false,
    deltaDigits: 3,
    hint: (percentile) => `Top ${100 - percentile}% threshold`,
    axisMode: 'unit',
  },
  coverage: {
    optionLabel: 'Coverage',
    shortLabel: 'Coverage',
    chartLabel: 'Coverage (higher better)',
    direction: 'higher',
    thresholdFree: true,
    deltaDigits: 3,
    hint: () => 'Higher is better',
    infoBanner:
      'Coverage measures how much attention energy lands inside the annotated regions and is threshold-free, so changing the percentile does not change the coverage score.',
    axisMode: 'unit',
  },
  mse: {
    optionLabel: 'MSE',
    shortLabel: 'MSE',
    chartLabel: 'MSE (lower better)',
    direction: 'lower',
    thresholdFree: true,
    deltaDigits: 4,
    hint: () => 'Lower is better',
    infoBanner:
      'MSE compares each attention heatmap against the Gaussian soft-union ground truth and is threshold-free, so changing the percentile keeps the scores the same.',
    axisMode: 'unit',
  },
  kl: {
    optionLabel: 'KL',
    shortLabel: 'KL',
    chartLabel: 'KL divergence (lower better)',
    direction: 'lower',
    thresholdFree: true,
    deltaDigits: 4,
    hint: () => 'Lower is better',
    infoBanner:
      'KL divergence reports KL(GT || attention) after both heatmaps are converted into smoothed probability distributions, so changing the percentile keeps the scores the same.',
    axisMode: 'auto',
  },
  emd: {
    optionLabel: 'EMD',
    shortLabel: 'EMD',
    chartLabel: 'EMD (lower better)',
    direction: 'lower',
    thresholdFree: true,
    deltaDigits: 4,
    hint: () => 'Lower is better',
    infoBanner:
      'EMD reports Earth Mover\'s Distance / Wasserstein-1 on a shared 8x8 support after both heatmaps are resized and normalized, so changing the percentile keeps the scores the same.',
    axisMode: 'auto',
  },
};

export const DASHBOARD_METRIC_METADATA: Record<DashboardMetric, AnalysisMetricMetadata> = {
  iou: ANALYSIS_METRIC_METADATA.iou,
  coverage: ANALYSIS_METRIC_METADATA.coverage,
  mse: ANALYSIS_METRIC_METADATA.mse,
  kl: ANALYSIS_METRIC_METADATA.kl,
  emd: ANALYSIS_METRIC_METADATA.emd,
};

export const ANALYSIS_METRIC_OPTIONS = (Object.entries(ANALYSIS_METRIC_METADATA) as Array<
  [AnalysisMetric, AnalysisMetricMetadata]
>).map(([value, metadata]) => ({
  value,
  label: metadata.optionLabel,
}));

export const DASHBOARD_METRIC_OPTIONS = (Object.entries(DASHBOARD_METRIC_METADATA) as Array<
  [DashboardMetric, AnalysisMetricMetadata]
>).map(([value, metadata]) => ({
  value,
  label: metadata.optionLabel,
}));

export const COMPARE_VARIANT_OPTIONS: Array<{ value: CompareVariantId; label: string }> = [
  { value: 'frozen', label: 'Frozen' },
  { value: 'linear_probe', label: 'Linear Probe' },
  { value: 'lora', label: 'LoRA' },
  { value: 'full', label: 'Full Fine-tune' },
];

export function isAnalysisMetric(value: string | null | undefined): value is AnalysisMetric {
  return Boolean(value && value in ANALYSIS_METRIC_METADATA);
}

export function isDashboardMetric(value: string | null | undefined): value is DashboardMetric {
  return Boolean(value && value in DASHBOARD_METRIC_METADATA);
}

export function isCompareVariantId(value: string | null | undefined): value is CompareVariantId {
  return value === 'frozen' || value === 'linear_probe' || value === 'lora' || value === 'full';
}

export function metricImprovementTone(
  direction: MetricDirection,
  delta: number | null | undefined,
): 'positive' | 'negative' | 'neutral' {
  if (delta === null || delta === undefined || Number.isNaN(delta) || delta === 0) {
    return 'neutral';
  }

  if (direction === 'higher') {
    return delta > 0 ? 'positive' : 'negative';
  }

  return delta < 0 ? 'positive' : 'negative';
}

export function formatMetricValue(
  metric: AnalysisMetric,
  value: number | null | undefined,
  options?: { signed?: boolean },
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a';
  }

  const digits = ANALYSIS_METRIC_METADATA[metric].deltaDigits;
  const prefix = options?.signed ? (value >= 0 ? '+' : '') : '';
  return `${prefix}${value.toFixed(digits)}`;
}
