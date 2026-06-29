/**
 * API client for the SSL Attention Visualization backend.
 */

const DEFAULT_API_BASE =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000/api`
    : 'http://127.0.0.1:8000/api';

const API_BASE = import.meta.env.VITE_API_URL || DEFAULT_API_BASE;

class APIError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

function buildHeaders(options?: RequestInit): Headers {
  const headers = new Headers(options?.headers);
  const method = options?.method?.toUpperCase() ?? 'GET';
  const hasBody = options?.body !== undefined && options?.body !== null;
  const shouldAttachJsonHeader =
    hasBody
    && method !== 'GET'
    && method !== 'HEAD'
    && !(options?.body instanceof FormData)
    && !headers.has('Content-Type');

  if (shouldAttachJsonHeader) {
    headers.set('Content-Type', 'application/json');
  }

  return headers;
}

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: buildHeaders(options),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new APIError(response.status, error.detail || 'Request failed');
  }

  return response.json();
}

// Images API
export const imagesAPI = {
  list: (params?: { style?: string; limit?: number; offset?: number }) => {
    const query = new URLSearchParams();
    if (params?.style) query.set('style', params.style);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const queryStr = query.toString();
    return fetchJSON<import('../types').ImageListItem[]>(`/images${queryStr ? `?${queryStr}` : ''}`);
  },

  getStyles: () => fetchJSON<string[]>('/images/styles'),

  getDetail: (imageId: string) =>
    fetchJSON<import('../types').ImageDetail>(`/images/${imageId}`),

  getImageUrl: (imageId: string, size?: number) => {
    const query = size ? `?size=${size}` : '';
    return `${API_BASE}/images/${imageId}/file${query}`;
  },

  getThumbnailUrl: (imageId: string) =>
    `${API_BASE}/images/${imageId}/thumbnail`,

  getWithBboxesUrl: (imageId: string) =>
    `${API_BASE}/images/${imageId}/with_bboxes`,
};

// Similarity response type
export interface SimilarityResponse {
  similarity: number[];
  patch_grid: [number, number];
  min_similarity: number;
  max_similarity: number;
  bbox_patch_indices: number[];
}

// Attention API
export const attentionAPI = {
  getHeatmapUrl: (imageId: string, model: string, layer: number, method?: string) => {
    const params = new URLSearchParams({ model, layer: String(layer) });
    if (method) params.set('method', method);
    return `${API_BASE}/attention/${imageId}/heatmap?${params}`;
  },

  getOverlayUrl: (imageId: string, model: string, layer: number, showBboxes = false, method?: string) => {
    const params = new URLSearchParams({
      model,
      layer: String(layer),
      show_bboxes: String(showBboxes),
    });
    if (method) params.set('method', method);
    return `${API_BASE}/attention/${imageId}/overlay?${params}`;
  },

  getLayerUrls: (imageId: string, model: string, showBboxes = false, method?: string) => {
    const params = new URLSearchParams({
      model,
      show_bboxes: String(showBboxes),
    });
    if (method) params.set('method', method);
    return fetchJSON<{
      image_id: string;
      model: string;
      method: string;
      show_bboxes: boolean;
      layers: Record<string, string>;
    }>(`/attention/${imageId}/layers?${params}`);
  },

  getModels: () => fetchJSON<import('../types').ModelsResponse>('/attention/models'),

  getRawAttention: (imageId: string, model: string, layer: number, method?: string, head?: number | null) => {
    const params = new URLSearchParams({ model, layer: String(layer) });
    if (method) params.set('method', method);
    if (head !== null && head !== undefined) params.set('head', String(head));
    return fetchJSON<import('../types').RawAttentionResponse>(
      `/attention/${imageId}/raw?${params}`
    );
  },

  getSimilarity: (
    imageId: string,
    bbox: { left: number; top: number; width: number; height: number; label?: string },
    model: string,
    layer: number
  ) =>
    fetchJSON<SimilarityResponse>(
      `/attention/${imageId}/similarity?model=${model}&layer=${layer}`,
      {
        method: 'POST',
        body: JSON.stringify(bbox),
      }
    ),
};

// Metrics API
export const metricsAPI = {
  getImageLayerProgression: (
    imageId: string,
    model: string,
    percentile = 90,
    method?: string,
    bboxIndex?: number | null
  ) => {
    const params = new URLSearchParams({ model, percentile: String(percentile) });
    if (method) params.set('method', method);
    if (bboxIndex !== null && bboxIndex !== undefined) params.set('bbox_index', String(bboxIndex));
    return fetchJSON<import('../types').ImageLayerProgression>(
      `/metrics/${imageId}/progression?${params}`
    );
  },

  getImageMetricsAllModels: (imageId: string, layer: number, percentile = 90, method?: string) => {
    const params = new URLSearchParams({ layer: String(layer), percentile: String(percentile) });
    if (method) params.set('method', method);
    return fetchJSON<{
      image_id: string;
      layer: string;
      percentile: number;
      models: Record<string, import('../types').IoUResult>;
    }>(`/metrics/${imageId}/all_models?${params}`);
  },

  getLayerProgression: (
    model: string,
    percentile = 90,
    metric: import('../types').DashboardMetric = 'iou',
    method?: string
  ) => {
    const params = new URLSearchParams({ percentile: String(percentile), metric });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').LayerProgression>(
      `/metrics/model/${model}/progression?${params}`
    );
  },

  getStyleBreakdown: (
    model: string,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    method?: string
  ) => {
    const params = new URLSearchParams({
      layer: String(layer),
      percentile: String(percentile),
      metric,
    });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').StyleBreakdown>(
      `/metrics/model/${model}/style_breakdown?${params}`
    );
  },

  getFeatureBreakdown: (
    model: string,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    sortBy: 'mean_score' | 'mean_iou' | 'bbox_count' | 'feature_name' | 'feature_label' = 'mean_score',
    minCount = 0,
    method?: string
  ) => {
    const params = new URLSearchParams({
      layer: String(layer),
      percentile: String(percentile),
      metric,
      sort_by: sortBy,
      min_count: String(minCount),
    });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').FeatureBreakdown>(
      `/metrics/model/${model}/feature_breakdown?${params}`
    );
  },

  getQ2Summary: (params?: {
    metric?: import('../types').AnalysisMetric;
    percentile?: number;
    model?: string;
    strategy?: string;
  }) => {
    const query = new URLSearchParams();
    if (params?.metric) query.set('metric', params.metric);
    if (params?.percentile !== undefined) query.set('percentile', String(params.percentile));
    if (params?.model) query.set('model', params.model);
    if (params?.strategy) query.set('strategy', params.strategy);
    const queryStr = query.toString();
    return fetchJSON<import('../types').Q2SummaryResponse>(
      `/metrics/q2_summary${queryStr ? `?${queryStr}` : ''}`
    );
  },

  getQ2ImageDeltas: (params: {
    model: string;
    strategy: 'linear_probe' | 'lora' | 'full';
    percentile?: number;
    topK?: number;
  }) => {
    const query = new URLSearchParams({
      model: params.model,
      strategy: params.strategy,
      percentile: String(params.percentile ?? 90),
      top_k: String(params.topK ?? 12),
    });
    return fetchJSON<import('../types').Q2ImageDeltasResponse>(
      `/metrics/q2_image_deltas?${query.toString()}`
    );
  },

  getHeadRanking: (
    model: string,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    variant: import('../types').CompareVariantId = 'frozen',
  ) => {
    const params = new URLSearchParams({
      layer: String(layer),
      percentile: String(percentile),
      metric,
      variant,
    });
    return fetchJSON<import('../types').HeadRankingResponse>(
      `/metrics/model/${model}/head_ranking?${params}`
    );
  },

  getImageHeadRanking: (
    imageId: string,
    model: string,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    variant: import('../types').CompareVariantId = 'frozen',
    options?: {
      bboxIndex?: number | null;
    },
  ) => {
    const params = new URLSearchParams({
      model,
      layer: String(layer),
      percentile: String(percentile),
      metric,
      variant,
    });
    if (options?.bboxIndex !== undefined && options?.bboxIndex !== null) {
      params.set('bbox_index', String(options.bboxIndex));
    }
    return fetchJSON<import('../types').ImageHeadRankingResponse>(
      `/metrics/${imageId}/head_ranking?${params}`
    );
  },

  getHeadFeatureMatrix: (
    model: string,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    variant: import('../types').CompareVariantId = 'frozen',
  ) => {
    const params = new URLSearchParams({
      layer: String(layer),
      percentile: String(percentile),
      metric,
      variant,
    });
    return fetchJSON<import('../types').HeadFeatureMatrixResponse>(
      `/metrics/model/${model}/head_feature_matrix?${params}`
    );
  },

  getHeadExemplars: (
    model: string,
    head: number,
    layer: number,
    percentile = 90,
    metric: import('../types').AnalysisMetric = 'iou',
    variant: import('../types').CompareVariantId = 'frozen',
    options?: {
      featureLabel?: number;
      limit?: number;
    },
  ) => {
    const params = new URLSearchParams({
      head: String(head),
      layer: String(layer),
      percentile: String(percentile),
      metric,
      variant,
    });
    if (options?.featureLabel !== undefined) {
      params.set('feature_label', String(options.featureLabel));
    }
    if (options?.limit !== undefined) {
      params.set('limit', String(options.limit));
    }
    return fetchJSON<import('../types').HeadExemplarResponse>(
      `/metrics/model/${model}/head_exemplars?${params}`
    );
  },

  getBboxMetrics: (
    imageId: string,
    model: string,
    layer: number,
    bboxIndex: number,
    percentile = 90,
    method?: string
  ) => {
    const params = new URLSearchParams({
      model,
      layer: String(layer),
      percentile: String(percentile),
    });
    if (method) params.set('method', method);
    return fetchJSON<import('../types').IoUResult>(
      `/metrics/${imageId}/bbox/${bboxIndex}?${params}`
    );
  },
};

// Comparison API
export const comparisonAPI = {
  compareModels: (
    imageId: string,
    models: string[],
    layer: number,
    percentile = 90,
    method?: string,
    bboxIndex?: number | null
  ) => {
    const params = new URLSearchParams({
      image_id: imageId,
      layer: String(layer),
      percentile: String(percentile),
    });
    for (const model of models) {
      params.append('models', model);
    }
    if (method) {
      params.set('method', method);
    }
    if (bboxIndex !== null && bboxIndex !== undefined) {
      params.set('bbox_index', String(bboxIndex));
    }
    return fetchJSON<import('../types').ModelComparison>(
      `/compare/models?${params.toString()}`
    );
  },

  compareFrozenVsFinetuned: (
    imageId: string,
    model: string,
    layer: number,
    strategy?: string,
    showBboxes = true
  ) => {
    const query = new URLSearchParams({ image_id: imageId, model, layer: String(layer) });
    if (strategy) query.set('strategy', strategy);
    query.set('show_bboxes', String(showBboxes));
    return fetchJSON<{
      image_id: string;
      model: string;
      strategy?: string | null;
      layer: string;
      show_bboxes?: boolean;
      frozen: { available: boolean; url: string | null };
      finetuned: { available: boolean; url: string | null; note: string };
    }>(`/compare/frozen_vs_finetuned?${query}`);
  },

  compareFinetunedVariants: (
    imageId: string,
    model: string,
    layer: number,
    strategyA: string,
    strategyB: string,
    showBboxes = true
  ) => {
    const query = new URLSearchParams({
      image_id: imageId,
      model,
      layer: String(layer),
      strategy_a: strategyA,
      strategy_b: strategyB,
      show_bboxes: String(showBboxes),
    });
    return fetchJSON<import('../types').VariantComparison>(
      `/compare/finetuned_vs_finetuned?${query}`
    );
  },

  compareVariants: (
    imageId: string,
    model: string,
    layer: number,
    leftVariant: import('../types').CompareVariantId,
    rightVariant: import('../types').CompareVariantId,
    showBboxes = true
  ) => {
    const query = new URLSearchParams({
      image_id: imageId,
      model,
      layer: String(layer),
      left_variant: leftVariant,
      right_variant: rightVariant,
      show_bboxes: String(showBboxes),
    });
    return fetchJSON<import('../types').VariantComparison>(
      `/compare/variants?${query.toString()}`
    );
  },

  compareVariantShift: (
    imageId: string,
    model: string,
    layer: number,
    comparedVariant: import('../types').ShiftComparedVariantId
  ) => {
    const query = new URLSearchParams({
      image_id: imageId,
      model,
      layer: String(layer),
      compared_variant: comparedVariant,
    });
    return fetchJSON<import('../types').VariantShiftMap>(
      `/compare/variants/shift?${query.toString()}`
    );
  },

  getAllModelsSummary: (
    percentile = 90,
    metric: import('../types').DashboardMetric = 'iou',
    options?: {
      method?: string;
      rankingMode?: import('../types').RankingMode;
    }
  ) => {
    const params = new URLSearchParams({
      percentile: String(percentile),
      metric,
    });
    if (options?.method) params.set('method', options.method);
    if (options?.rankingMode) params.set('ranking_mode', options.rankingMode);
    return fetchJSON<import('../types').AllModelsSummary>(
      `/compare/all_models_summary?${params.toString()}`
    );
  },
};

export { APIError };
