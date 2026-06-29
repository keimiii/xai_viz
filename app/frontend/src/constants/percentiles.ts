/**
 * Shared percentile threshold options.
 * Must match the values precomputed in app/precompute/generate_metrics_cache.py.
 */
export const PERCENTILE_OPTIONS = [
  { value: 90, label: 'Top 10%' },
  { value: 85, label: 'Top 15%' },
  { value: 80, label: 'Top 20%' },
  { value: 75, label: 'Top 25%' },
  { value: 70, label: 'Top 30%' },
  { value: 60, label: 'Top 40%' },
  { value: 50, label: 'Top 50%' },
];
