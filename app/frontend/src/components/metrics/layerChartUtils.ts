import type {
  ImageLayerMetricPoint,
  ImageMetricDescriptor,
  MetricDirection,
} from '../../types';

export interface LayerChartDatum {
  layer: number;
  label: string;
  isActive: boolean;
  [key: string]: boolean | number | string | null;
}

export interface NumericAxisConfig {
  domain: [number, number];
  ticks: number[];
}

const DEFAULT_AXIS_CONFIG: NumericAxisConfig = {
  domain: [0, 1],
  ticks: [0, 0.2, 0.4, 0.6, 0.8, 1],
};

export function getRenderedSeriesKey(metricKey: string): string {
  return `series_${metricKey}`;
}

export function buildLayerChartData(
  layers: ImageLayerMetricPoint[],
  metrics: ImageMetricDescriptor[],
  currentLayer: number,
  isPlaying: boolean
): LayerChartDatum[] {
  return layers.map((point) => {
    const datum: LayerChartDatum = {
      layer: point.layer,
      label: `L${point.layer}`,
      isActive: point.layer === currentLayer,
    };

    for (const metric of metrics) {
      const value = point.values[metric.key];
      datum[getRenderedSeriesKey(metric.key)] = isPlaying && point.layer > currentLayer ? null : value;
    }

    return datum;
  });
}

export function computeYAxisConfig(
  layers: ImageLayerMetricPoint[],
  visibleMetricKeys: string[]
): NumericAxisConfig {
  const values = layers.flatMap((point) =>
    visibleMetricKeys
      .map((metricKey) => point.values[metricKey])
      .filter((value): value is number => value !== null && value !== undefined)
  );

  return computeFlexibleNumericAxisConfig(values);
}

export function computeFlexibleNumericAxisConfig(values: number[]): NumericAxisConfig {
  if (values.length === 0) {
    return DEFAULT_AXIS_CONFIG;
  }

  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return DEFAULT_AXIS_CONFIG;
  }

  const maxValue = Math.max(...finiteValues);
  if (maxValue <= 0) {
    return DEFAULT_AXIS_CONFIG;
  }

  const roughStep = maxValue / 5;
  const step = getNiceStep(roughStep);
  const max = roundAxisValue(Math.ceil(maxValue / step) * step);
  const tickCount = Math.max(1, Math.round(max / step));
  const ticks = Array.from({ length: tickCount + 1 }, (_, index) =>
    roundAxisValue(index * step)
  );

  return { domain: [0, max], ticks };
}

export function getXAxisTicks(layers: ImageLayerMetricPoint[]): number[] {
  const evenTicks = layers
    .map((point) => point.layer)
    .filter((layer) => layer % 2 === 0);

  return evenTicks.length >= 2 ? evenTicks : layers.map((point) => point.layer);
}

export function getMetricDirectionLabel(direction: MetricDirection): string {
  return direction === 'higher' ? 'Higher better' : 'Lower better';
}

function getNiceStep(value: number): number {
  const exponent = Math.floor(Math.log10(value));
  const magnitude = 10 ** exponent;
  const normalized = value / magnitude;

  if (normalized <= 1) return 1 * magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function roundAxisValue(value: number): number {
  return Number(value.toFixed(6));
}
