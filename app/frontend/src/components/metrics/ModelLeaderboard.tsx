/**
 * Model leaderboard component showing a ranked score chart and rows.
 */

import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, CardContent, CardHeader } from '../ui/Card';
import type { DashboardMetric, LeaderboardEntry } from '../../types';
import { DASHBOARD_METRIC_METADATA } from '../../constants/metricMetadata';
import { getAttentionMethodLabel } from '../../constants/attentionMethods';
import { getDashboardContinuousBaselines } from '../../constants/dashboardBaselines';
import { computeFlexibleNumericAxisConfig } from './layerChartUtils';

interface ModelLeaderboardProps {
  leaderboard?: LeaderboardEntry[];
  percentile: number;
  metric: DashboardMetric;
  isLoading?: boolean;
  hasError?: boolean;
  emptyMessage?: string;
  onModelSelect?: (entry: LeaderboardEntry) => void;
}

interface LeaderboardChartDatum {
  model: string;
  chartLabel: string;
  score: number;
  bestLayer: string;
  methodUsed: string;
  rank: number;
}

const BAR_FILL_BY_RANK: Record<number, string> = {
  1: '#0f766e',
  2: '#0284c7',
  3: '#2563eb',
};

export function ModelLeaderboard({
  leaderboard,
  percentile,
  metric,
  isLoading = false,
  hasError = false,
  emptyMessage = 'No compatible models available for this method.',
  onModelSelect,
}: ModelLeaderboardProps) {
  const metricMetadata = DASHBOARD_METRIC_METADATA[metric];
  const metricLabel = metricMetadata.shortLabel;
  const metricHint = metricMetadata.hint(percentile);

  const chartData = useMemo<LeaderboardChartDatum[]>(
    () =>
      (leaderboard ?? []).map((entry) => ({
        model: entry.model,
        chartLabel: `#${entry.rank} ${entry.model}`,
        score: entry.score,
        bestLayer: entry.best_layer,
        methodUsed: entry.method_used,
        rank: entry.rank,
      })),
    [leaderboard]
  );
  const baselineReferences = useMemo(
    () => getDashboardContinuousBaselines(metric),
    [metric]
  );
  const xAxisConfig = useMemo(() => {
    const values = [
      ...chartData.map((entry) => entry.score),
      ...baselineReferences.map((reference) => reference.value),
    ];
    return computeFlexibleNumericAxisConfig(values);
  }, [baselineReferences, chartData]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Model Leaderboard</h3>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 rounded bg-gray-200" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (hasError) {
    return (
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Model Leaderboard</h3>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-red-500">Failed to load leaderboard</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Model Leaderboard</h3>
          <span className="text-xs text-gray-500">{metricHint}</span>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {!leaderboard?.length ? (
          <div className="px-4 py-6 text-sm text-gray-500">{emptyMessage}</div>
        ) : (
          <>
            <div className="border-b border-slate-200 px-4 pb-4">
              <div className="h-72" data-testid="leaderboard-score-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 8, right: 20, bottom: 8, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={xAxisConfig.domain}
                      ticks={xAxisConfig.ticks}
                      tickFormatter={formatAxisTick}
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis
                      type="category"
                      dataKey="chartLabel"
                      tick={{ fontSize: 12, fill: '#475569' }}
                      width={96}
                    />
                    <Tooltip
                      content={
                        <LeaderboardScoreTooltip
                          digits={metricMetadata.deltaDigits}
                          metricLabel={metricMetadata.chartLabel}
                        />
                      }
                    />
                    {baselineReferences.map((reference) => (
                      <ReferenceLine
                        key={reference.key}
                        x={reference.value}
                        stroke={reference.stroke}
                        strokeDasharray="6 6"
                        strokeWidth={2}
                      />
                    ))}
                    <Bar dataKey="score" radius={[0, 5, 5, 0]} isAnimationActive={false}>
                      {chartData.map((entry) => (
                        <Cell
                          key={`bar-${entry.model}`}
                          fill={BAR_FILL_BY_RANK[entry.rank] ?? '#60a5fa'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {baselineReferences.length > 0 && (
                <div className="mt-3 space-y-2" data-testid="leaderboard-baseline-legend">
                  <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Documented Baselines
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {baselineReferences.map((reference) => (
                      <div
                        key={reference.key}
                        className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-700"
                        data-testid={`leaderboard-baseline-${reference.key}`}
                      >
                        <span
                          aria-hidden="true"
                          className="inline-block h-0.5 w-4"
                          style={{
                            backgroundColor: reference.stroke,
                            borderTop: `2px dashed ${reference.stroke}`,
                          }}
                        />
                        <span>{reference.label}</span>
                        <span className="font-semibold text-slate-900">
                          {reference.value.toFixed(metricMetadata.deltaDigits)}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-slate-500">
                    Dashed lines in the chart show the same reference values.
                  </p>
                </div>
              )}
            </div>

            <div className="divide-y">
              {leaderboard.map((entry) => (
                <div
                  key={entry.model}
                  data-testid={`leaderboard-row-${entry.model}`}
                  className={`flex items-center gap-3 px-4 py-3 ${
                    onModelSelect ? 'cursor-pointer hover:bg-gray-50' : ''
                  }`}
                  onClick={() => onModelSelect?.(entry)}
                >
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold ${
                      entry.rank === 1
                        ? 'bg-yellow-100 text-yellow-700'
                        : entry.rank === 2
                        ? 'bg-gray-100 text-gray-700'
                        : entry.rank === 3
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-gray-50 text-gray-500'
                    }`}
                  >
                    #{entry.rank}
                  </div>

                  <div className="flex-1">
                    <div className="font-medium capitalize">{entry.model}</div>
                    <div
                      className="text-xs text-gray-500"
                      data-testid={`leaderboard-row-meta-${entry.model}`}
                    >
                      Best: {entry.best_layer} • {getAttentionMethodLabel(entry.method_used)}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-lg font-bold text-primary-600">
                      {entry.score.toFixed(metricMetadata.deltaDigits)}
                    </div>
                    <div className="text-xs text-gray-500">{metricLabel}</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

interface LeaderboardScoreTooltipProps {
  active?: boolean;
  payload?: Array<{ payload?: LeaderboardChartDatum; value?: number }>;
  digits: number;
  metricLabel: string;
}

function LeaderboardScoreTooltip({
  active,
  payload,
  digits,
  metricLabel,
}: LeaderboardScoreTooltipProps) {
  const datum = payload?.[0]?.payload;
  const value = payload?.[0]?.value;

  if (!active || !datum || typeof value !== 'number') {
    return null;
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-lg">
      <div className="font-semibold text-slate-900">{datum.model}</div>
      <div className="mt-2 space-y-1 text-slate-600">
        <div className="flex items-center justify-between gap-4">
          <span>Score</span>
          <span className="font-medium text-slate-900">{value.toFixed(digits)}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span>Best layer</span>
          <span className="font-medium text-slate-900">{datum.bestLayer}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span>Method</span>
          <span className="font-medium text-slate-900">
            {getAttentionMethodLabel(datum.methodUsed)}
          </span>
        </div>
        <div className="pt-1 text-xs text-slate-500">{metricLabel}</div>
      </div>
    </div>
  );
}

function formatAxisTick(value: number): string {
  if (Number.isInteger(value)) {
    return value.toString();
  }

  return value.toFixed(6).replace(/\.?0+$/, '');
}
