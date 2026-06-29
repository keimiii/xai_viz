import type { DashboardMetric } from '../types';

export type ContinuousDashboardMetric = 'mse' | 'kl' | 'emd';
export type DashboardBaselineKey =
  | 'random'
  | 'center_gaussian'
  | 'saliency_prior'
  | 'sobel_edge';

interface DashboardBaselineStyle {
  label: string;
  stroke: string;
}

export interface DashboardBaselineReference extends DashboardBaselineStyle {
  key: DashboardBaselineKey;
  value: number;
}

const BASELINE_ORDER: DashboardBaselineKey[] = [
  'random',
  'center_gaussian',
  'saliency_prior',
  'sobel_edge',
];

const BASELINE_STYLES: Record<DashboardBaselineKey, DashboardBaselineStyle> = {
  random: {
    label: 'Random',
    stroke: '#475569',
  },
  center_gaussian: {
    label: 'Center Gaussian',
    stroke: '#a16207',
  },
  saliency_prior: {
    label: 'Saliency Prior',
    stroke: '#0f766e',
  },
  sobel_edge: {
    label: 'Sobel Edge',
    stroke: '#9a3412',
  },
};

const CONTINUOUS_BASELINE_VALUES: Record<
  ContinuousDashboardMetric,
  Record<DashboardBaselineKey, number>
> = {
  mse: {
    random: 0.3192,
    center_gaussian: 0.1770,
    saliency_prior: 0.0957,
    sobel_edge: 0.0376,
  },
  kl: {
    random: 3.3627,
    center_gaussian: 2.6317,
    saliency_prior: 2.6111,
    sobel_edge: 3.2237,
  },
  emd: {
    random: 0.3468,
    center_gaussian: 0.2836,
    saliency_prior: 0.2654,
    sobel_edge: 0.3137,
  },
};

export function getDashboardContinuousBaselines(
  metric: DashboardMetric
): DashboardBaselineReference[] {
  if (metric !== 'mse' && metric !== 'kl' && metric !== 'emd') {
    return [];
  }

  return BASELINE_ORDER.map((key) => ({
    key,
    label: BASELINE_STYLES[key].label,
    stroke: BASELINE_STYLES[key].stroke,
    value: CONTINUOUS_BASELINE_VALUES[metric][key],
  }));
}
