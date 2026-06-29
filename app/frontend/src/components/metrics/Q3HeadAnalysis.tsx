import { useMemo, useState } from 'react';

import { useModels } from '../../hooks/useAttention';
import { ANALYSIS_METRIC_OPTIONS } from '../../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../../constants/percentiles';
import {
  type Q3ReportState,
  type Q3ReportView,
} from '../../constants/q3Routing';
import {
  Q3ReportDeltaView,
  Q3ReportHeadRankingView,
  Q3ReportMatrixView,
} from '../../pages/Q3Report';
import {
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3SelectionHelperText,
  getQ3VariantScopeStatus,
} from '../../constants/q3Scope';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Select } from '../ui/Select';
import { Q3ScopeChip } from './Q3ScopeFraming';
import type { AnalysisMetric, CompareVariantId } from '../../types';

const Q3_DASHBOARD_REPORT_VIEW_OPTIONS: Array<{ value: Q3ReportView; label: string }> = [
  { value: 'head-ranking', label: 'Head ranking' },
  { value: 'head-feature-matrix', label: 'Head feature matrix' },
  { value: 'frozen-delta', label: 'Frozen to adapted delta' },
];

export function Q3HeadAnalysis() {
  const { data: modelsData } = useModels();
  const [view, setView] = useState<Q3ReportView>('head-ranking');
  const [model, setModel] = useState('dinov3');
  const [variant, setVariant] = useState<CompareVariantId>('frozen');
  const [layer, setLayer] = useState(10);
  const [metric, setMetric] = useState<AnalysisMetric>('iou');
  const [percentile, setPercentile] = useState(90);
  const [head, setHead] = useState<number | null>(null);
  const [featureLabel, setFeatureLabel] = useState<number | null>(null);

  const availableQ3Models = useMemo(() => {
    const visibleModels = (modelsData?.models ?? []).filter((value) =>
      Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number]),
    );
    return visibleModels.length > 0 ? visibleModels : [...Q3_PRIMARY_MODELS];
  }, [modelsData?.models]);

  const resolvedModel = availableQ3Models.includes(model) ? model : availableQ3Models[0];
  const maxLayer = modelsData?.num_layers_per_model?.[resolvedModel]
    ? modelsData.num_layers_per_model[resolvedModel] - 1
    : 11;
  const resolvedLayer = Math.min(layer, maxLayer);
  const variantScopeStatus = getQ3VariantScopeStatus(variant);
  const selectionHelperText = getQ3SelectionHelperText('primary', variantScopeStatus);
  const reportState: Q3ReportState = {
    view,
    model: resolvedModel,
    variant,
    layer: resolvedLayer,
    metric,
    percentile,
    head,
    featureLabel,
  };

  const modelOptions = availableQ3Models.map((value) => ({
    value,
    label: value,
  }));
  const variantOptions = Q3_VARIANT_OPTIONS.map((option) => ({
    value: option.value,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));

  const handleReportStateChange = (patch: Partial<Q3ReportState>) => {
    if (patch.head !== undefined) {
      setHead(patch.head);
    }
    if (patch.featureLabel !== undefined) {
      setFeatureLabel(patch.featureLabel);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="text-xl font-bold text-slate-950">Q3 Per-Head Specialization</h3>
            <p className="mt-2 max-w-5xl text-base font-semibold leading-7 text-slate-900">
              Do individual attention heads exhibit descriptive specialization for different architectural features, and do the dominant heads change across variants?
            </p>
            <p className="mt-1 text-sm text-gray-600">
              Select one Q3 report view at a time for screenshots or narrated demos.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-6">
          <Select
            value={view}
            onChange={(value) => {
              setView(value as Q3ReportView);
              setHead(null);
              setFeatureLabel(null);
            }}
            options={Q3_DASHBOARD_REPORT_VIEW_OPTIONS}
            label="View"
          />
          <Select
            value={resolvedModel}
            onChange={(value) => {
              setModel(value);
              setHead(null);
              setFeatureLabel(null);
            }}
            options={modelOptions}
            label="Model"
          />
          <Select
            value={variant}
            onChange={(value) => {
              setVariant(value as CompareVariantId);
              setHead(null);
              setFeatureLabel(null);
            }}
            options={variantOptions}
            label="Variant"
            disabled={view === 'frozen-delta'}
          />
          <Select
            value={resolvedLayer}
            onChange={(value) => {
              setLayer(Number(value));
              setHead(null);
              setFeatureLabel(null);
            }}
            options={Array.from({ length: maxLayer + 1 }, (_, index) => ({
              value: index,
              label: `Layer ${index}`,
            }))}
            label="Layer"
          />
          <Select
            value={metric}
            onChange={(value) => {
              setMetric(value as AnalysisMetric);
              setHead(null);
              setFeatureLabel(null);
            }}
            options={ANALYSIS_METRIC_OPTIONS}
            label="Metric"
          />
          <Select
            value={percentile}
            onChange={(value) => {
              setPercentile(Number(value));
              setHead(null);
              setFeatureLabel(null);
            }}
            options={PERCENTILE_OPTIONS}
            label="Percentile"
          />
        </div>

        <div
          className="rounded-lg border border-slate-200 bg-white px-4 py-3"
          data-testid="q3-selection-scope"
        >
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            Current Q3 workflow context
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">View</span>
              <span>{Q3_DASHBOARD_REPORT_VIEW_OPTIONS.find((option) => option.value === view)?.label ?? view}</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Model</span>
              <span>{resolvedModel}</span>
              <Q3ScopeChip status="primary" dataTestId="q3-model-scope-chip" />
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
              <span className="font-medium text-slate-900">Variant</span>
              <span>{Q3_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant}</span>
              <Q3ScopeChip status={variantScopeStatus} dataTestId="q3-variant-scope-chip" />
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-600" data-testid="q3-selection-helper">
            {view === 'frozen-delta'
              ? 'This view ignores the variant selector and compares Frozen against LoRA and Full.'
              : selectionHelperText}
          </p>
        </div>

        {view === 'head-ranking' && (
          <Q3ReportHeadRankingView
            model={resolvedModel}
            variant={variant}
            layer={resolvedLayer}
            metric={metric}
            percentile={percentile}
          />
        )}

        {view === 'head-feature-matrix' && (
          <Q3ReportMatrixView
            state={reportState}
            percentile={percentile}
            onStateChange={handleReportStateChange}
          />
        )}

        {view === 'frozen-delta' && (
          <Q3ReportDeltaView
            model={resolvedModel}
            layer={resolvedLayer}
            metric={metric}
            percentile={percentile}
          />
        )}
      </CardContent>
    </Card>
  );
}
