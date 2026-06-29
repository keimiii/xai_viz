import { useState } from 'react';

import { ANALYSIS_METRIC_METADATA, ANALYSIS_METRIC_OPTIONS, COMPARE_VARIANT_OPTIONS, formatMetricValue } from '../../constants/metricMetadata';
import {
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3ScopeChipClassName,
  getQ3ScopeLabel,
  getQ3VariantScopeStatus,
} from '../../constants/q3Scope';
import { Card, CardContent, CardHeader } from '../ui/Card';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';
import type { AnalysisMetric, CompareVariantId, ImageHeadRankingEntry, ImageHeadRankingResponse } from '../../types';

const TOP_HEAD_COUNT = 4;

interface Q3ImageDetailControlsProps {
  model: string;
  variant: CompareVariantId;
  layer: number;
  head: number | null;
  rankingMetric: AnalysisMetric;
  maxLayer: number;
  numHeads: number;
  showBboxes: boolean;
  featureName: string | null;
  rankingData?: ImageHeadRankingResponse;
  rankingLoading: boolean;
  rankingError: string | null;
  variantSupportsPerHead?: boolean;
  onModelChange: (model: string) => void;
  onVariantChange: (variant: CompareVariantId) => void;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number | null) => void;
  onMetricChange: (metric: AnalysisMetric) => void;
  onShowBboxesChange: (show: boolean) => void;
}

export function Q3ImageDetailControls({
  model,
  variant,
  layer,
  head,
  rankingMetric,
  maxLayer,
  numHeads,
  showBboxes,
  featureName,
  rankingData,
  rankingLoading,
  rankingError,
  variantSupportsPerHead,
  onModelChange,
  onVariantChange,
  onLayerChange,
  onHeadChange,
  onMetricChange,
  onShowBboxesChange,
}: Q3ImageDetailControlsProps) {
  const [showAllHeads, setShowAllHeads] = useState(false);
  const variantStatus = getQ3VariantScopeStatus(variant);
  const controlVariantLabel = COMPARE_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant;
  const rankedHeads = rankingData?.heads ?? [];
  const topHeads = rankedHeads.slice(0, TOP_HEAD_COUNT);
  const rankingScopeLabel = rankingData?.selection.mode === 'bbox'
    ? `Selected bbox: ${rankingData.selection.bbox_label ?? `bbox ${rankingData.selection.bbox_index}`}`
    : 'Whole-image union of annotations';
  const availabilityReason = variantSupportsPerHead === false
    ? `Per-head Q3 cache is not available for ${model} / ${controlVariantLabel} yet. You can keep using All (Fused) while this variant is backfilled.`
    : rankingData?.reason ?? null;

  return (
    <Card>
      <CardHeader>
        <div className="space-y-1">
          <h3 className="font-semibold text-gray-900">Q3 Controls</h3>
          <p className="text-sm text-gray-600">
            Stay inside the selected Q3 workflow context while you inspect one exemplar image.
          </p>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3">
          <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            Current drill-down
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-sm text-slate-700">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1">
              <span className="font-medium text-slate-900">Model</span>
              <span>{model}</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1">
              <span className="font-medium text-slate-900">Variant</span>
              <span>{controlVariantLabel}</span>
              <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold ${getQ3ScopeChipClassName(variantStatus)}`}>
                {getQ3ScopeLabel(variantStatus)}
              </span>
            </span>
          </div>
          {featureName && (
            <p className="mt-2 text-xs text-slate-600">
              Feature context from Dashboard Q3: {featureName}
            </p>
          )}
          <p className="mt-2 text-xs text-slate-600" data-testid="q3-ranking-scope-copy">
            Ranking scope: {rankingScopeLabel}
          </p>
        </div>

        <Select
          value={model}
          onChange={onModelChange}
          options={Q3_PRIMARY_MODELS.map((value) => ({
            value,
            label: value,
          }))}
          label="Model"
        />

        <Select
          value={variant}
          onChange={(value) => onVariantChange(value as CompareVariantId)}
          options={Q3_VARIANT_OPTIONS.map((option) => ({
            value: option.value,
            label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
          }))}
          label="Variant"
        />

        <Select
          value={rankingMetric}
          onChange={(value) => onMetricChange(value as AnalysisMetric)}
          options={ANALYSIS_METRIC_OPTIONS}
          label="Rank by"
        />

        <div className="space-y-3 rounded-lg border border-slate-200 bg-white px-3 py-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-slate-900">Top heads</div>
              <div className="text-xs text-slate-500">
                Ranked by {ANALYSIS_METRIC_METADATA[rankingMetric].optionLabel} for this image context.
              </div>
            </div>
            <button
              type="button"
              onClick={() => onHeadChange(null)}
              data-testid="q3-head-choice-all"
              className={buildHeadButtonClassName(head === null, true)}
            >
              <span>All (Fused)</span>
            </button>
          </div>

          {rankingLoading && (
            <div className="grid grid-cols-1 gap-2" data-testid="q3-head-ranking-loading">
              {Array.from({ length: TOP_HEAD_COUNT }).map((_, index) => (
                <div key={index} className="h-12 animate-pulse rounded-lg bg-slate-100" />
              ))}
            </div>
          )}

          {!rankingLoading && rankingError && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
              Failed to load ranked heads. {rankingError}
            </div>
          )}

          {!rankingLoading && !rankingError && rankingData?.supported === false && availabilityReason && (
            <div
              className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
              data-testid="q3-head-ranking-unavailable"
            >
              {availabilityReason}
            </div>
          )}

          {!rankingLoading && !rankingError && rankingData?.supported !== false && rankedHeads.length === 0 && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
              No ranked heads are available for this image selection yet.
            </div>
          )}

          {!rankingLoading && !rankingError && rankingData?.supported !== false && topHeads.length > 0 && (
            <>
              <div className="grid grid-cols-1 gap-2" data-testid="q3-top-head-strip">
                {topHeads.map((entry, index) => (
                  <HeadChoiceButton
                    key={entry.head}
                    entry={entry}
                    index={index}
                    metric={rankingMetric}
                    selected={head === entry.head}
                    testId={`q3-top-head-${entry.head}`}
                    onClick={() => onHeadChange(entry.head)}
                  />
                ))}
              </div>

              {rankedHeads.length > TOP_HEAD_COUNT && (
                <button
                  type="button"
                  onClick={() => setShowAllHeads((current) => !current)}
                  className="text-sm font-medium text-primary-700 hover:text-primary-800"
                  data-testid="q3-head-gallery-toggle"
                >
                  {showAllHeads ? 'Hide all ranked heads' : `Show all ${Math.min(numHeads, rankedHeads.length)} ranked heads`}
                </button>
              )}

              {showAllHeads && (
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2" data-testid="q3-head-gallery">
                  {rankedHeads.map((entry, index) => (
                    <HeadChoiceButton
                      key={entry.head}
                      entry={entry}
                      index={index}
                      metric={rankingMetric}
                      selected={head === entry.head}
                      testId={`q3-gallery-head-${entry.head}`}
                      onClick={() => onHeadChange(entry.head)}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        <Slider
          value={layer}
          onChange={onLayerChange}
          min={0}
          max={maxLayer}
          label={`Layer ${layer}`}
          showValue={false}
        />

        <Toggle
          checked={showBboxes}
          onChange={onShowBboxesChange}
          label="Show Bounding Boxes"
        />

        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          Head context stays loaded even in Feature Similarity mode so you can switch back to attention inspection without losing the selected Q3 finding.
        </div>
      </CardContent>
    </Card>
  );
}

function HeadChoiceButton({
  entry,
  index,
  metric,
  selected,
  testId,
  onClick,
}: {
  entry: ImageHeadRankingEntry;
  index: number;
  metric: AnalysisMetric;
  selected: boolean;
  testId: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={testId}
      className={buildHeadButtonClassName(selected)}
    >
      <span className="flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
          #{index + 1}
        </span>
        <span className="font-medium text-slate-900">Head {entry.head}</span>
      </span>
      <span className="text-sm font-medium text-slate-700">
        {formatMetricValue(metric, entry.score)}
      </span>
    </button>
  );
}

function buildHeadButtonClassName(selected: boolean, compact = false): string {
  const base = compact
    ? 'inline-flex items-center justify-center rounded-full border px-3 py-1.5 text-sm font-medium transition-colors'
    : 'flex items-center justify-between rounded-lg border px-3 py-2 text-left transition-colors';

  if (selected) {
    return `${base} border-primary-300 bg-primary-50 text-primary-900 ring-2 ring-primary-200`;
  }

  return `${base} border-slate-200 bg-white text-slate-700 hover:border-primary-200 hover:bg-primary-50`;
}
