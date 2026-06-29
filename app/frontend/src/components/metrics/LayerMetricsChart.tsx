import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {
  ImageLayerMetricPoint,
  ImageMetricDescriptor,
} from '../../types';
import {
  buildLayerChartData,
  computeYAxisConfig,
  getXAxisTicks,
  getRenderedSeriesKey,
} from './layerChartUtils';

interface LayerMetricsChartProps {
  layers: ImageLayerMetricPoint[];
  metrics: ImageMetricDescriptor[];
  visibleMetricKeys: string[];
  currentLayer: number;
  isPlaying: boolean;
  metricColors: Record<string, string>;
}

export function LayerMetricsChart({
  layers,
  metrics,
  visibleMetricKeys,
  currentLayer,
  isPlaying,
  metricColors,
}: LayerMetricsChartProps) {
  const visibleMetrics = useMemo(
    () => metrics.filter((metric) => visibleMetricKeys.includes(metric.key)),
    [metrics, visibleMetricKeys],
  );

  const chartData = useMemo(
    () => buildLayerChartData(layers, visibleMetrics, currentLayer, isPlaying),
    [layers, visibleMetrics, currentLayer, isPlaying],
  );

  const yAxisConfig = useMemo(
    () => computeYAxisConfig(layers, visibleMetricKeys),
    [layers, visibleMetricKeys],
  );
  const xAxisTicks = useMemo(() => getXAxisTicks(layers), [layers]);
  const activeLayerValues = useMemo(
    () => layers.find((point) => point.layer === currentLayer)?.values ?? {},
    [layers, currentLayer],
  );

  return (
    <div className="w-full" data-testid="layer-metrics-chart">
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 12, right: 12, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#dbe3ec" />
            <XAxis
              dataKey="layer"
              axisLine={false}
              tickLine={false}
              ticks={xAxisTicks}
              interval={0}
              tickFormatter={(value) => String(value)}
              tick={{ fill: '#64748b', fontSize: 12 }}
              height={30}
              tickMargin={8}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              domain={yAxisConfig.domain}
              ticks={yAxisConfig.ticks}
              tickFormatter={(value: number) => formatAxisTick(value)}
              tickMargin={8}
              width={60}
            />
            <Tooltip content={<LayerMetricsTooltip metrics={visibleMetrics} />} />
            <ReferenceLine x={currentLayer} stroke="#0f172a" strokeDasharray="4 4" />

            {visibleMetrics.map((metric) => (
              <Line
                key={metric.key}
                type="monotone"
                dataKey={getRenderedSeriesKey(metric.key)}
                stroke={metricColors[metric.key] || '#2563eb'}
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
                name={metric.label}
              />
            ))}

            {visibleMetrics.map((metric) => {
              const value = activeLayerValues[metric.key];
              if (typeof value !== 'number') {
                return null;
              }

              return (
                <ReferenceDot
                  key={`active-dot-${metric.key}`}
                  x={currentLayer}
                  y={value}
                  r={5}
                  fill={metricColors[metric.key] || '#2563eb'}
                  stroke="#ffffff"
                  strokeWidth={2}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div
        className="mt-2 text-center text-sm font-medium text-slate-500"
        data-testid="chart-x-axis-caption"
      >
        Layers
      </div>
    </div>
  );
}

interface LayerMetricsTooltipProps {
  metrics: ImageMetricDescriptor[];
}

function LayerMetricsTooltip({ active, payload, label, metrics }: LayerMetricsTooltipProps & {
  active?: boolean;
  payload?: Array<{ dataKey?: string; value?: number | null }>;
  label?: number;
}) {
  if (!active || !payload?.length) {
    return null;
  }

  const valueBySeriesKey = new Map(payload.map((entry) => [entry.dataKey, entry.value]));

  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-lg">
      <div className="font-semibold text-slate-900">Layer {label}</div>
      <div className="mt-2 space-y-1">
        {metrics.map((metric) => {
          const value = valueBySeriesKey.get(getRenderedSeriesKey(metric.key));
          return (
            <div key={metric.key} className="flex items-center justify-between gap-4">
              <span className="text-slate-600">{metric.label}</span>
              <span className="font-medium text-slate-900">
                {typeof value === 'number' ? value.toFixed(3) : '—'}
              </span>
            </div>
          );
        })}
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
