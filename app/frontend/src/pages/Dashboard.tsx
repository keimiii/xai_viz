/**
 * Dashboard page with overall metrics and leaderboard.
 */

import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar,
} from 'recharts';
import { useViewStore } from '../store/viewStore';
import { useAllModelsSummary, useStyleBreakdown } from '../hooks/useMetrics';
import { useModels } from '../hooks/useAttention';
import { ModelLeaderboard } from '../components/metrics/ModelLeaderboard';
import { FeatureBreakdown } from '../components/metrics/FeatureBreakdown';
import { Q3HeadAnalysis } from '../components/metrics/Q3HeadAnalysis';
import { Q2Page } from './Q2';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { ErrorBoundary } from '../components/ui/ErrorBoundary';
import { PageTabs } from '../components/ui/PageTabs';
import { Select } from '../components/ui/Select';
import { computeFlexibleNumericAxisConfig } from '../components/metrics/layerChartUtils';
import { getAttentionMethodLabel } from '../constants/attentionMethods';
import { parsePageTab } from '../constants/pageTabs';
import { PERCENTILE_OPTIONS } from '../constants/percentiles';
import type {
  AllModelsSummary,
  DashboardMetric,
  LeaderboardEntry,
  PageTab,
  RankingMode,
} from '../types';
import {
  DASHBOARD_METRIC_METADATA,
  DASHBOARD_METRIC_OPTIONS,
} from '../constants/metricMetadata';

const RANKING_MODE_OPTIONS: Array<{ value: RankingMode; label: string }> = [
  { value: 'default_method', label: 'Default method' },
  { value: 'best_available', label: 'Best available' },
];

export function DashboardPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    model,
    layer,
    method,
    percentile,
    setModelWithPreferredMethod,
    setPercentile,
    setMethodsConfig,
    setNumLayersPerModel,
  } = useViewStore();
  const { data: modelsData } = useModels();
  const [metric, setMetric] = useState<DashboardMetric>('iou');
  const [rankingMode, setRankingMode] = useState<RankingMode>('default_method');
  const currentTab = parsePageTab(searchParams.get('tab'));
  const isMainTab = currentTab === 'main';
  const isQ2Tab = currentTab === 'q2';
  const [hasVisitedQ3Tab, setHasVisitedQ3Tab] = useState(currentTab === 'q3');
  const isQ3Tab = currentTab === 'q3';
  const shouldRenderQ3Panel = hasVisitedQ3Tab || isQ3Tab;
  const metricMetadata = DASHBOARD_METRIC_METADATA[metric];
  const metricLabel = metricMetadata.chartLabel;
  const q2AnalysisHref = buildQ2AnalysisHref(metric, percentile, model);

  // Populate store with model config (methods, layer counts) so setModel
  // resolves the correct default method (e.g. gradcam for ResNet-50)
  useEffect(() => {
    if (modelsData?.methods && modelsData?.default_methods) {
      setMethodsConfig(modelsData.methods, modelsData.default_methods);
    }
  }, [modelsData, setMethodsConfig]);

  useEffect(() => {
    if (modelsData?.num_layers_per_model) {
      setNumLayersPerModel(modelsData.num_layers_per_model);
    }
  }, [modelsData, setNumLayersPerModel]);

  const {
    data: summary,
    isLoading: summaryLoading,
    error: summaryError,
  } = useAllModelsSummary(percentile, metric, { rankingMode });
  const { data: styleBreakdown, isLoading: styleLoading } = useStyleBreakdown(
    model,
    layer,
    percentile,
    metric,
    method
  );
  const summaryContextMessage = getSummaryContextMessage(summary, rankingMode);
  const leaderboardEmptyMessage = getLeaderboardEmptyMessage(summary, rankingMode);
  const handleLeaderboardModelSelect = (entry: LeaderboardEntry) => {
    setModelWithPreferredMethod(entry.model, entry.method_used);
  };
  const handleTabChange = (nextTab: PageTab) => {
    if (nextTab === 'q3') {
      setHasVisitedQ3Tab(true);
    }
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('tab', nextTab);
    setSearchParams(nextParams);
  };

  // Collect all unique layers from API response (handles models with different layer counts)
  const allLayers = new Set<string>();
  if (summary?.models) {
    for (const modelData of Object.values(summary.models)) {
      for (const layerKey of Object.keys(modelData.layer_progression)) {
        allLayers.add(layerKey);
      }
    }
  }
  // Sort layers numerically (layer0, layer1, ..., layer11)
  const sortedLayers = Array.from(allLayers).sort((a, b) =>
    parseInt(a.replace('layer', '')) - parseInt(b.replace('layer', ''))
  );

  // Merge into single array by layer
  const chartData: Record<string, number | string>[] = sortedLayers.map((layerKey) => {
    const layerNum = parseInt(layerKey.replace('layer', ''));
    const layerData: Record<string, number | string> = { layer: `L${layerNum}` };
    if (summary?.models) {
      for (const [modelName, modelData] of Object.entries(summary.models)) {
        if (modelData.layer_progression[layerKey] !== undefined) {
          layerData[modelName] = modelData.layer_progression[layerKey];
        }
      }
    }
    return layerData;
  });
  const layerProgressionValues = chartData.flatMap((point) =>
    Object.entries(point)
      .filter(([key]) => key !== 'layer')
      .map(([, value]) => value)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  );
  const yAxisConfig = computeFlexibleNumericAxisConfig(layerProgressionValues);

  // Style breakdown data
  const styleData = styleBreakdown
    ? Object.entries(styleBreakdown.styles).map(([style, score]) => ({
        styleLabel: style,
        score,
        count: styleBreakdown.style_counts[style] || 0,
      }))
    : [];
  const styleAxisConfig = getNumericAxisConfig(
    metricMetadata.axisMode,
    styleData.map((row) => row.score)
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Attention-annotation alignment metrics across models
          </p>
        </div>

        {isMainTab && (
          <div className="flex flex-wrap items-end gap-3">
            <Select
              value={metric}
              onChange={(value) => setMetric(value as DashboardMetric)}
              options={DASHBOARD_METRIC_OPTIONS.map((option) => ({ ...option }))}
              label="Metric"
            />
            <Select
              value={percentile}
              onChange={(v) => setPercentile(Number(v))}
              options={PERCENTILE_OPTIONS}
              label="Threshold"
            />
            <div className="flex flex-col gap-1">
              <label className="text-sm font-medium text-gray-700">Ranking</label>
              <div className="inline-flex rounded-md border border-gray-300 bg-white p-1 shadow-sm">
                {RANKING_MODE_OPTIONS.map((option) => {
                  const isSelected = rankingMode === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      data-testid={`dashboard-ranking-mode-${option.value}`}
                      onClick={() => setRankingMode(option.value)}
                      className={`rounded px-3 py-1.5 text-sm transition-colors ${
                        isSelected
                          ? 'bg-primary-600 text-white'
                          : 'text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      <PageTabs
        label="Dashboard sections"
        activeTab={currentTab}
        onChange={handleTabChange}
        tabs={[
          {
            value: 'main',
            label: 'Overview',
            id: 'dashboard-page-tab-main',
            panelId: 'dashboard-main-panel',
            dataTestId: 'dashboard-page-tab-main',
          },
          {
            value: 'q2',
            label: 'Q2',
            id: 'dashboard-page-tab-q2',
            panelId: 'dashboard-q2-panel',
            dataTestId: 'dashboard-page-tab-q2',
          },
          {
            value: 'q3',
            label: 'Q3',
            id: 'dashboard-page-tab-q3',
            panelId: 'dashboard-q3-panel',
            dataTestId: 'dashboard-page-tab-q3',
          },
        ]}
      />

      <div
        id="dashboard-main-panel"
        role="tabpanel"
        aria-labelledby="dashboard-page-tab-main"
        hidden={!isMainTab}
        className={`space-y-6 ${isMainTab ? '' : 'hidden'}`}
        data-testid="dashboard-main-panel"
      >
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="text-sm font-bold uppercase tracking-[0.14em] text-slate-600">
            Question it answers for Q1
          </div>
          <p className="mt-2 max-w-5xl text-base font-semibold leading-7 text-slate-900">
            Do frozen SSL and baseline vision models attend to the same architectural regions that human experts mark as diagnostically important?
          </p>
        </div>

        {metricMetadata.thresholdFree && metricMetadata.infoBanner && (
          <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-900">
            {metricMetadata.infoBanner}
          </div>
        )}

        {summaryContextMessage && (
          <div
            className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700"
            data-testid="dashboard-method-context"
          >
            {summaryContextMessage}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div>
            <ErrorBoundary resetKeys={[percentile, metric, rankingMode]}>
              <ModelLeaderboard
                leaderboard={summary?.leaderboard}
                percentile={percentile}
                metric={metric}
                isLoading={summaryLoading}
                hasError={!!summaryError}
                emptyMessage={leaderboardEmptyMessage}
                onModelSelect={handleLeaderboardModelSelect}
              />
            </ErrorBoundary>
          </div>

          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <div className="flex justify-between items-center gap-4">
                  <h3 className="font-semibold">Layer Progression (All Models)</h3>
                  <span className="text-xs text-gray-500">{metricLabel}</span>
                </div>
              </CardHeader>
              <CardContent>
                {summaryLoading ? (
                  <div className="h-64 animate-pulse bg-gray-100 rounded" />
                ) : summaryError ? (
                  <div className="h-64 flex items-center justify-center text-sm text-red-500">
                    Failed to load layer progression
                  </div>
                ) : !summary?.leaderboard.length ? (
                  <div className="h-64 flex items-center justify-center text-sm text-gray-500">
                    {leaderboardEmptyMessage}
                  </div>
                ) : (
                  <div className="h-[300px]" data-testid="dashboard-layer-progression-chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="layer"
                          tick={{ fontSize: 12 }}
                          tickMargin={10}
                        />
                        <YAxis
                          domain={yAxisConfig.domain}
                          ticks={yAxisConfig.ticks}
                          tick={{ fontSize: 12 }}
                          tickFormatter={formatAxisTick}
                        />
                        <Tooltip
                          formatter={(value: number) => [value.toFixed(3), metricLabel]}
                          labelStyle={{ color: '#374151', fontWeight: 600 }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        {summary?.leaderboard.map((entry, i) => (
                          <Line
                            key={entry.model}
                            type="monotone"
                            dataKey={entry.model}
                            name={entry.model}
                            stroke={`hsl(${(i * 137.5) % 360}, 70%, 50%)`}
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            activeDot={{ r: 5 }}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <h3 className="font-semibold">{metricMetadata.optionLabel} by Architectural Style</h3>
                <span className="text-xs text-gray-500 capitalize">
                  {model} • {metricMetadata.optionLabel}
                </span>
              </div>
            </CardHeader>
            <CardContent>
              {styleLoading ? (
                <div className="h-48 animate-pulse bg-gray-100 rounded" />
              ) : styleData.length > 0 ? (
                <div className="h-[200px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={styleData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                      <XAxis
                        type="number"
                        domain={styleAxisConfig.domain}
                        ticks={styleAxisConfig.ticks}
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis
                        type="category"
                        dataKey="styleLabel"
                        tick={{ fontSize: 12 }}
                        width={80}
                      />
                      <Tooltip
                        formatter={(value: number, name: string) => {
                          if (name === 'score') return [value.toFixed(3), `Mean ${metricMetadata.optionLabel}`];
                          return [value, 'Images'];
                        }}
                      />
                      <Bar
                        dataKey="score"
                        fill="#3b82f6"
                        radius={[0, 4, 4, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-48 flex items-center justify-center text-gray-500">
                  No style data available
                </div>
              )}
            </CardContent>
          </Card>

          <ErrorBoundary resetKeys={[model, layer, percentile, metric, method]}>
            <FeatureBreakdown
              model={model}
              layer={layer}
              percentile={percentile}
              metric={metric}
              method={method}
            />
          </ErrorBoundary>
        </div>

        <Card>
          <CardHeader>
            <h3 className="font-semibold">Quick Actions</h3>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
              <Link
                to="/"
                className="block p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="font-medium">Browse Images</div>
                <div className="text-sm text-gray-500">
                  View all 139 annotated church images
                </div>
              </Link>

              <Link
                to={`/compare?type=models&layer=${layer}&percentile=${percentile}&metric=${metric}`}
                className="block p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="font-medium">Compare Models</div>
                <div className="text-sm text-gray-500">
                  Side-by-side attention comparison
                </div>
              </Link>

              <Link
                to={q2AnalysisHref}
                className="block p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="font-medium">Q2 Fine-Tuning Analysis</div>
                <div className="text-sm text-gray-500">
                  Strategy-aware summary from the active experiment
                </div>
              </Link>

              <div className="p-3 bg-yellow-50 rounded-lg">
                <div className="font-medium text-yellow-800">Pre-computation Required</div>
                <div className="text-sm text-yellow-700">
                  Run the pre-computation scripts to generate heatmaps and metrics
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div
        id="dashboard-q2-panel"
        role="tabpanel"
        aria-labelledby="dashboard-page-tab-q2"
        hidden={!isQ2Tab}
        className={`space-y-6 ${isQ2Tab ? '' : 'hidden'}`}
        data-testid="dashboard-q2-panel"
      >
        <Q2Page />
      </div>

      {shouldRenderQ3Panel && (
        <div
          id="dashboard-q3-panel"
          role="tabpanel"
          aria-labelledby="dashboard-page-tab-q3"
          hidden={!isQ3Tab}
          className={`space-y-6 ${isQ3Tab ? '' : 'hidden'}`}
          data-testid="dashboard-q3-panel"
        >
          <ErrorBoundary>
            <Q3HeadAnalysis />
          </ErrorBoundary>
        </div>
      )}
    </div>
  );
}

interface YAxisConfig {
  domain: [number, number];
  ticks: number[];
}

function getLeaderboardEmptyMessage(
  summary: AllModelsSummary | undefined,
  rankingMode: RankingMode
): string {
  if (summary?.method) {
    return `No compatible models available for ${getAttentionMethodLabel(summary.method)}.`;
  }

  if (rankingMode === 'best_available') {
    return 'No models have compatible scores for best-available ranking.';
  }

  return 'No models have compatible scores for default-method ranking.';
}

function getSummaryContextMessage(
  summary: AllModelsSummary | undefined,
  rankingMode: RankingMode
): string | null {
  if (!summary) {
    return null;
  }

  if (summary.method) {
    const excludedModelsText = summary.excluded_models.length
      ? ` Excluded models: ${summary.excluded_models.join(', ')}.`
      : '';
    return `Summary panels are using ${getAttentionMethodLabel(summary.method)}.${excludedModelsText}`;
  }

  if ((summary.ranking_mode ?? rankingMode) === 'best_available') {
    return "Leaderboard and layer progression rank each model by its strongest available attention method.";
  }

  return "Leaderboard and layer progression rank each model by its default attention method.";
}

function getNumericAxisConfig(axisMode: 'unit' | 'auto', values: number[]): YAxisConfig {
  if (axisMode === 'unit') {
    return { domain: [0, 1], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
  }

  return computeFlexibleNumericAxisConfig(values);
}

function formatAxisTick(value: number): string {
  if (Number.isInteger(value)) {
    return value.toString();
  }

  return value.toFixed(3).replace(/\.?0+$/, '');
}

function buildQ2AnalysisHref(
  metric: DashboardMetric,
  percentile: number,
  model: string
): string {
  const params = new URLSearchParams({
    metric,
    percentile: String(percentile),
    model,
  });

  return `/q2?${params.toString()}`;
}
