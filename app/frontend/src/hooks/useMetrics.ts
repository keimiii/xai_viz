/**
 * React Query hooks for metrics data.
 */

import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { metricsAPI, comparisonAPI } from '../api/client';
import type { AnalysisMetric, CompareVariantId, DashboardMetric, RankingMode } from '../types';

export function useStyleBreakdown(
  model: string,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  method?: string
) {
  return useQuery({
    queryKey: ['styleBreakdown', model, layer, percentile, metric, method],
    queryFn: () => metricsAPI.getStyleBreakdown(model, layer, percentile, metric, method),
  });
}

export function useFeatureBreakdown(
  model: string,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  sortBy: 'mean_score' | 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label' = 'mean_score',
  minCount = 0,
  method?: string
) {
  return useQuery({
    queryKey: ['featureBreakdown', model, layer, percentile, metric, sortBy, minCount, method],
    queryFn: () => metricsAPI.getFeatureBreakdown(model, layer, percentile, metric, sortBy, minCount, method),
  });
}

export function useAllModelsSummary(
  percentile: number,
  metric: DashboardMetric,
  options?: { method?: string; rankingMode?: RankingMode }
) {
  return useQuery({
    queryKey: ['allModelsSummary', percentile, metric, options?.method, options?.rankingMode],
    queryFn: () => comparisonAPI.getAllModelsSummary(percentile, metric, options),
  });
}

export function useQ2Summary(
  metric: AnalysisMetric,
  percentile?: number,
  model?: string,
  strategy?: string
) {
  return useQuery({
    queryKey: ['q2Summary', metric, percentile, model, strategy],
    queryFn: () => metricsAPI.getQ2Summary({ metric, percentile, model, strategy }),
  });
}

export function useQ2ImageDeltas(
  model: string | undefined,
  strategy: 'linear_probe' | 'lora' | 'full' | undefined,
  percentile: number,
  topK = 12
) {
  return useQuery({
    queryKey: ['q2ImageDeltas', model, strategy, percentile, topK],
    queryFn: () => metricsAPI.getQ2ImageDeltas({ model: model!, strategy: strategy!, percentile, topK }),
    enabled: !!model && !!strategy,
  });
}

export function useHeadRanking(
  model: string,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  variant: CompareVariantId
) {
  return useQuery({
    queryKey: ['headRanking', model, layer, percentile, metric, variant],
    queryFn: () => metricsAPI.getHeadRanking(model, layer, percentile, metric, variant),
  });
}

export function useImageHeadRanking(
  imageId: string | undefined,
  model: string,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  variant: CompareVariantId,
  options?: {
    bboxIndex?: number | null;
    enabled?: boolean;
  },
) {
  return useQuery({
    queryKey: ['imageHeadRanking', imageId, model, layer, percentile, metric, variant, options?.bboxIndex ?? null],
    queryFn: () => metricsAPI.getImageHeadRanking(imageId!, model, layer, percentile, metric, variant, options),
    enabled: !!imageId && (options?.enabled ?? true),
    placeholderData: keepPreviousData,
  });
}

export function useHeadFeatureMatrix(
  model: string,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  variant: CompareVariantId
) {
  return useQuery({
    queryKey: ['headFeatureMatrix', model, layer, percentile, metric, variant],
    queryFn: () => metricsAPI.getHeadFeatureMatrix(model, layer, percentile, metric, variant),
  });
}

export function useHeadExemplars(
  model: string,
  head: number | null,
  layer: number,
  percentile: number,
  metric: AnalysisMetric,
  variant: CompareVariantId,
  options?: {
    featureLabel?: number;
    limit?: number;
    enabled?: boolean;
  },
) {
  return useQuery({
    queryKey: ['headExemplars', model, head, layer, percentile, metric, variant, options?.featureLabel, options?.limit],
    queryFn: () => metricsAPI.getHeadExemplars(model, head ?? 0, layer, percentile, metric, variant, options),
    enabled: options?.enabled ?? head !== null,
  });
}
