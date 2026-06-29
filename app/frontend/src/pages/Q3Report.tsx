import { useEffect, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { useModels } from '../hooks/useAttention';
import { useHeadFeatureMatrix, useHeadRanking } from '../hooks/useMetrics';
import { Card, CardContent, CardHeader } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import {
  ANALYSIS_METRIC_METADATA,
  ANALYSIS_METRIC_OPTIONS,
  formatMetricValue,
  metricImprovementTone,
} from '../constants/metricMetadata';
import { PERCENTILE_OPTIONS } from '../constants/percentiles';
import {
  createQ3ReportSearchParams,
  parseQ3ReportState,
  type Q3ReportState,
  type Q3ReportView,
} from '../constants/q3Routing';
import {
  Q3_DEFAULTS,
  Q3_PRIMARY_MODELS,
  Q3_VARIANT_OPTIONS,
  formatQ3ScopeOptionLabel,
  getQ3SelectionHelperText,
  getQ3VariantScopeStatus,
} from '../constants/q3Scope';
import type {
  AnalysisMetric,
  CompareVariantId,
  HeadFeatureMatrixRow,
  HeadRankingEntry,
  HeadRankingResponse,
  MetricDirection,
} from '../types';

const REPORT_VIEW_OPTIONS: Array<{ value: Q3ReportView; label: string }> = [
  { value: 'head-ranking', label: 'Head ranking' },
  { value: 'head-feature-matrix', label: 'Head feature matrix' },
  { value: 'frozen-delta', label: 'Frozen to adapted delta' },
];

const MATRIX_ROW_LIMIT = 24;
const MATRIX_DEFAULT_MIN_ANNOTATIONS = 5;

const Q3_VIEW_QUESTIONS: Record<Q3ReportView, string> = {
  'head-ranking':
    'Do some heads consistently score better than others for the selected model, variant, layer, metric, and percentile?',
  'head-feature-matrix':
    'Do the stronger heads align with particular architectural feature labels?',
  'frozen-delta':
    'Does fine-tuning preserve, sharpen, or reorganize the dominant head set?',
};

type DeltaState = 'promoted' | 'demoted' | 'stable';
type DeltaTargetVariant = 'lora' | 'full';

interface RankedHeadEntry extends HeadRankingEntry {
  scoreRank: number;
}

interface DeltaRow {
  head: number;
  frozenRank: number | null;
  adaptedRank: number | null;
  rankDelta: number | null;
  frozenScore: number | null;
  adaptedScore: number | null;
  scoreDelta: number | null;
  state: DeltaState | null;
}

interface DeltaComparison {
  targetVariant: DeltaTargetVariant;
  label: string;
  available: boolean;
  reason: string | null;
  rows: DeltaRow[];
  summary: Record<DeltaState, number>;
  topFrozen: RankedHeadEntry | null;
  topAdapted: RankedHeadEntry | null;
}

function getVariantLabel(variant: CompareVariantId): string {
  return Q3_VARIANT_OPTIONS.find((option) => option.value === variant)?.label ?? variant;
}

function getMetricLabel(metric: AnalysisMetric, percentile: number): string {
  const metadata = ANALYSIS_METRIC_METADATA[metric];
  return metadata.thresholdFree ? metadata.shortLabel : `${metadata.shortLabel}@${percentile}`;
}

function sortHeadEntries(
  heads: HeadRankingEntry[] | undefined,
  direction: MetricDirection,
): RankedHeadEntry[] {
  return [...(heads ?? [])]
    .sort((left, right) => {
      const scoreDiff = direction === 'higher'
        ? right.mean_score - left.mean_score
        : left.mean_score - right.mean_score;
      if (scoreDiff !== 0) {
        return scoreDiff;
      }

      const rankDiff = left.mean_rank - right.mean_rank;
      if (rankDiff !== 0) {
        return rankDiff;
      }

      return left.head - right.head;
    })
    .map((entry, index) => ({
      ...entry,
      scoreRank: index + 1,
    }));
}

function getHeadRankMap(heads: RankedHeadEntry[]): Map<number, RankedHeadEntry> {
  return new Map(heads.map((entry) => [entry.head, entry]));
}

function getScoreQualityGap(
  topHead: RankedHeadEntry | null,
  nextHead: RankedHeadEntry | null,
  direction: MetricDirection,
): number | null {
  if (!topHead || !nextHead) {
    return null;
  }

  return direction === 'higher'
    ? topHead.mean_score - nextHead.mean_score
    : nextHead.mean_score - topHead.mean_score;
}

function compareNullableScores(
  left: number | null,
  right: number | null,
  direction: MetricDirection,
): number {
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return direction === 'higher' ? right - left : left - right;
}

function getFeatureScore(feature: HeadFeatureMatrixRow, headIndex: number): number | null {
  if (headIndex < 0 || headIndex >= feature.scores.length) {
    return null;
  }
  return feature.scores[headIndex];
}

function chooseDefaultFeature(
  features: HeadFeatureMatrixRow[],
  headIndex: number,
  direction: MetricDirection,
): HeadFeatureMatrixRow | null {
  const withEnoughSupport = features.filter((feature) =>
    feature.bbox_count >= MATRIX_DEFAULT_MIN_ANNOTATIONS && getFeatureScore(feature, headIndex) !== null,
  );
  const candidates = withEnoughSupport.length > 0
    ? withEnoughSupport
    : features.filter((feature) => getFeatureScore(feature, headIndex) !== null);

  return [...candidates].sort((left, right) =>
    compareNullableScores(getFeatureScore(left, headIndex), getFeatureScore(right, headIndex), direction)
      || right.bbox_count - left.bbox_count
      || left.feature_name.localeCompare(right.feature_name),
  )[0] ?? null;
}

function getMatrixRange(features: HeadFeatureMatrixRow[]): { min: number; max: number } | null {
  const scores = features.flatMap((feature) =>
    feature.scores.filter((score): score is number => score !== null),
  );
  if (scores.length === 0) {
    return null;
  }
  return {
    min: Math.min(...scores),
    max: Math.max(...scores),
  };
}

function getHeatmapIntensity(
  score: number | null,
  range: { min: number; max: number } | null,
  direction: MetricDirection,
): number {
  if (score === null || !range || range.min === range.max) {
    return score === null ? 0 : 1;
  }
  const normalized = Math.max(0, Math.min(1, (score - range.min) / (range.max - range.min)));
  return direction === 'higher' ? normalized : 1 - normalized;
}

function interpolate(start: number, end: number, ratio: number): number {
  return Math.round(start + (end - start) * ratio);
}

function getHeatmapCellStyle(
  score: number | null,
  range: { min: number; max: number } | null,
  direction: MetricDirection,
) {
  if (score === null) {
    return {
      backgroundColor: '#f8fafc',
      color: '#94a3b8',
    };
  }

  const intensity = getHeatmapIntensity(score, range, direction);
  return {
    backgroundColor: `rgb(${interpolate(248, 8, intensity)}, ${interpolate(250, 145, intensity)}, ${interpolate(252, 178, intensity)})`,
    color: intensity > 0.6 ? '#ffffff' : '#0f172a',
  };
}

function getDeltaState(rankDelta: number | null): DeltaState | null {
  if (rankDelta === null) return null;
  if (rankDelta > 0) return 'promoted';
  if (rankDelta < 0) return 'demoted';
  return 'stable';
}

function formatRank(rank: number | null): string {
  return rank === null ? 'n/a' : `#${rank}`;
}

function formatRankDelta(rankDelta: number | null): string {
  if (rankDelta === null) return 'n/a';
  return `${rankDelta > 0 ? '+' : ''}${rankDelta}`;
}

function getDeltaStateClassName(state: DeltaState | null): string {
  if (state === 'promoted') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  if (state === 'demoted') return 'border-rose-200 bg-rose-50 text-rose-800';
  if (state === 'stable') return 'border-slate-200 bg-slate-100 text-slate-700';
  return 'border-slate-200 bg-slate-50 text-slate-500';
}

function getScoreDeltaClassName(metric: AnalysisMetric, delta: number | null): string {
  const tone = metricImprovementTone(ANALYSIS_METRIC_METADATA[metric].direction, delta);
  if (tone === 'positive') return 'bg-emerald-50 text-emerald-800';
  if (tone === 'negative') return 'bg-rose-50 text-rose-800';
  return 'bg-slate-100 text-slate-700';
}

function buildDeltaComparison(
  frozenResponse: HeadRankingResponse | undefined,
  adaptedResponse: HeadRankingResponse | undefined,
  targetVariant: DeltaTargetVariant,
  direction: MetricDirection,
): DeltaComparison | null {
  if (!frozenResponse || !adaptedResponse) {
    return null;
  }

  const frozenHeads = sortHeadEntries(frozenResponse.heads, direction);
  const adaptedHeads = sortHeadEntries(adaptedResponse.heads, direction);
  const frozenMap = getHeadRankMap(frozenHeads);
  const adaptedMap = getHeadRankMap(adaptedHeads);
  const allHeads = Array.from(new Set([...frozenMap.keys(), ...adaptedMap.keys()])).sort((a, b) => a - b);
  const label = targetVariant === 'lora' ? 'Frozen -> LoRA' : 'Frozen -> Full';

  if (!adaptedResponse.supported) {
    return {
      targetVariant,
      label,
      available: false,
      reason: adaptedResponse.reason || `${label} data is not available for this setting.`,
      rows: [],
      summary: { promoted: 0, demoted: 0, stable: 0 },
      topFrozen: frozenHeads[0] ?? null,
      topAdapted: null,
    };
  }

  const rows = allHeads.map((head) => {
    const frozen = frozenMap.get(head);
    const adapted = adaptedMap.get(head);
    const rankDelta = frozen && adapted ? frozen.scoreRank - adapted.scoreRank : null;
    const scoreDelta = frozen && adapted ? adapted.mean_score - frozen.mean_score : null;

    return {
      head,
      frozenRank: frozen?.scoreRank ?? null,
      adaptedRank: adapted?.scoreRank ?? null,
      rankDelta,
      frozenScore: frozen?.mean_score ?? null,
      adaptedScore: adapted?.mean_score ?? null,
      scoreDelta,
      state: getDeltaState(rankDelta),
    };
  });

  rows.sort((left, right) => {
    const movementDiff = Math.abs(right.rankDelta ?? 0) - Math.abs(left.rankDelta ?? 0);
    if (movementDiff !== 0) return movementDiff;
    const adaptedRankDiff = (left.adaptedRank ?? Number.MAX_SAFE_INTEGER) - (right.adaptedRank ?? Number.MAX_SAFE_INTEGER);
    if (adaptedRankDiff !== 0) return adaptedRankDiff;
    return left.head - right.head;
  });

  return {
    targetVariant,
    label,
    available: true,
    reason: null,
    rows,
    summary: {
      promoted: rows.filter((row) => row.state === 'promoted').length,
      demoted: rows.filter((row) => row.state === 'demoted').length,
      stable: rows.filter((row) => row.state === 'stable').length,
    },
    topFrozen: frozenHeads[0] ?? null,
    topAdapted: adaptedHeads[0] ?? null,
  };
}

function normalizeMatrixRows(
  features: HeadFeatureMatrixRow[],
  selectedFeatureLabel: number | null,
  selectedHeadIndex: number,
  direction: MetricDirection,
): HeadFeatureMatrixRow[] {
  const sorted = [...features].sort((left, right) =>
    compareNullableScores(getFeatureScore(left, selectedHeadIndex), getFeatureScore(right, selectedHeadIndex), direction)
      || right.bbox_count - left.bbox_count
      || left.feature_name.localeCompare(right.feature_name),
  );
  const visible = sorted.slice(0, MATRIX_ROW_LIMIT);

  if (selectedFeatureLabel === null || visible.some((feature) => feature.feature_label === selectedFeatureLabel)) {
    return visible;
  }

  const selectedFeature = features.find((feature) => feature.feature_label === selectedFeatureLabel);
  if (!selectedFeature) {
    return visible;
  }

  return [selectedFeature, ...visible.slice(0, MATRIX_ROW_LIMIT - 1)];
}

function SupportMessage({
  children,
  tone = 'slate',
}: {
  children: string;
  tone?: 'slate' | 'amber';
}) {
  const toneClassName = tone === 'amber'
    ? 'border-amber-200 bg-amber-50 text-amber-900'
    : 'border-slate-200 bg-slate-50 text-slate-700';

  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${toneClassName}`}>
      {children}
    </div>
  );
}

function Q3ViewQuestion({ view }: { view: Q3ReportView }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="text-sm font-bold uppercase tracking-[0.14em] text-slate-600">
        Question it answers for Q3
      </div>
      <p className="mt-2 text-base font-semibold leading-7 text-slate-900">
        {Q3_VIEW_QUESTIONS[view]}
      </p>
    </div>
  );
}

export function Q3ReportPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: modelsData } = useModels();
  const searchParamsString = searchParams.toString();

  const availableQ3Models = useMemo(() => {
    const visibleModels = (modelsData?.models ?? []).filter((value) =>
      Q3_PRIMARY_MODELS.includes(value as (typeof Q3_PRIMARY_MODELS)[number]),
    );
    return visibleModels.length > 0 ? visibleModels : [...Q3_PRIMARY_MODELS];
  }, [modelsData?.models]);

  const provisionalState = useMemo(
    () => parseQ3ReportState(new URLSearchParams(searchParamsString), {
      availableModels: availableQ3Models,
    }),
    [availableQ3Models, searchParamsString],
  );

  const maxLayer = modelsData?.num_layers_per_model?.[provisionalState.model]
    ? modelsData.num_layers_per_model[provisionalState.model] - 1
    : Q3_DEFAULTS.layer;
  const numHeads = modelsData?.num_heads_per_model?.[provisionalState.model] ?? 12;

  const reportState = useMemo(
    () => parseQ3ReportState(new URLSearchParams(searchParamsString), {
      availableModels: availableQ3Models,
      maxLayer,
      numHeads,
    }),
    [availableQ3Models, maxLayer, numHeads, searchParamsString],
  );

  const metricMetadata = ANALYSIS_METRIC_METADATA[reportState.metric];
  const effectivePercentile = metricMetadata.thresholdFree ? 90 : reportState.percentile;
  const variantScopeStatus = getQ3VariantScopeStatus(reportState.variant);
  const variantOptions = Q3_VARIANT_OPTIONS.map((option) => ({
    value: option.value,
    label: formatQ3ScopeOptionLabel(option.label, getQ3VariantScopeStatus(option.value)),
  }));

  useEffect(() => {
    const normalized = createQ3ReportSearchParams(reportState);
    if (normalized.toString() !== searchParamsString) {
      setSearchParams(normalized, { replace: true });
    }
  }, [reportState, searchParamsString, setSearchParams]);

  const updateReportState = (patch: Partial<Q3ReportState>, replace = false) => {
    const nextState = {
      ...reportState,
      ...patch,
    };
    const normalized = createQ3ReportSearchParams(nextState);
    if (normalized.toString() === searchParamsString) {
      return;
    }
    setSearchParams(normalized, { replace });
  };

  return (
    <div className="space-y-6" data-testid="q3-report-page">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Q3 screenshot surface
          </div>
          <h1 className="mt-2 text-3xl font-bold text-slate-950">Q3 Report View</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            One Q3 question at a time: ranking, feature specialization, or adaptation delta.
          </p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/dashboard?tab=q3')}
          className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
        >
          Dashboard Q3
        </button>
      </div>

      <Card>
        <CardHeader>
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Report controls</h2>
            <p className="mt-1 text-sm text-slate-600">
              URL state is shareable, so each view can be reopened exactly for screenshots or narration.
            </p>
          </div>
        </CardHeader>
        <CardContent className="space-y-4" data-testid="q3-report-controls">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-6">
            <Select
              value={reportState.view}
              onChange={(value) => updateReportState({ view: value as Q3ReportView })}
              options={REPORT_VIEW_OPTIONS}
              label="View"
            />
            <Select
              value={reportState.model}
              onChange={(value) => updateReportState({ model: value, head: null, featureLabel: null })}
              options={availableQ3Models.map((value) => ({ value, label: value }))}
              label="Model"
            />
            <Select
              value={reportState.variant}
              onChange={(value) => updateReportState({
                variant: value as CompareVariantId,
                head: null,
                featureLabel: null,
              })}
              options={variantOptions}
              label="Variant"
              disabled={reportState.view === 'frozen-delta'}
            />
            <Select
              value={reportState.layer}
              onChange={(value) => updateReportState({ layer: Number(value), head: null, featureLabel: null })}
              options={Array.from({ length: maxLayer + 1 }, (_, index) => ({
                value: index,
                label: `Layer ${index}`,
              }))}
              label="Layer"
            />
            <Select
              value={reportState.metric}
              onChange={(value) => updateReportState({
                metric: value as AnalysisMetric,
                head: null,
                featureLabel: null,
              })}
              options={ANALYSIS_METRIC_OPTIONS}
              label="Metric"
            />
            <Select
              value={reportState.percentile}
              onChange={(value) => updateReportState({ percentile: Number(value), head: null, featureLabel: null })}
              options={PERCENTILE_OPTIONS}
              label="Percentile"
            />
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
            <div className="flex flex-wrap gap-2 text-sm">
              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 font-medium text-emerald-800">
                Primary Q3 models
              </span>
              <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-slate-700">
                {reportState.model} · layer{reportState.layer} · {getMetricLabel(reportState.metric, effectivePercentile)}
              </span>
              <span className={`rounded-full border px-3 py-1 font-medium ${
                variantScopeStatus === 'control'
                  ? 'border-amber-200 bg-amber-50 text-amber-800'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-800'
              }`}>
                {reportState.view === 'frozen-delta'
                  ? 'Delta compares Frozen, LoRA, and Full'
                  : getVariantLabel(reportState.variant)}
              </span>
            </div>
            <p className="mt-2 text-xs text-slate-600">
              {reportState.view === 'frozen-delta'
                ? 'The variant selector is parked because this view always compares Frozen against LoRA and Full.'
                : getQ3SelectionHelperText('primary', variantScopeStatus)}
            </p>
            {metricMetadata.thresholdFree && (
              <p className="mt-2 text-xs text-slate-500">
                {metricMetadata.infoBanner}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {reportState.view === 'head-ranking' && (
        <Q3ReportHeadRankingView
          model={reportState.model}
          variant={reportState.variant}
          layer={reportState.layer}
          metric={reportState.metric}
          percentile={effectivePercentile}
        />
      )}

      {reportState.view === 'head-feature-matrix' && (
        <Q3ReportMatrixView
          state={reportState}
          percentile={effectivePercentile}
          onStateChange={updateReportState}
        />
      )}

      {reportState.view === 'frozen-delta' && (
        <Q3ReportDeltaView
          model={reportState.model}
          layer={reportState.layer}
          metric={reportState.metric}
          percentile={effectivePercentile}
        />
      )}
    </div>
  );
}

export function Q3ReportHeadRankingView({
  model,
  variant,
  layer,
  metric,
  percentile,
}: {
  model: string;
  variant: CompareVariantId;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
}) {
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const rankingQuery = useHeadRanking(model, layer, percentile, metric, variant);
  const sortedHeads = useMemo(
    () => sortHeadEntries(rankingQuery.data?.heads, metricMetadata.direction),
    [metricMetadata.direction, rankingQuery.data?.heads],
  );
  const topHead = sortedHeads[0] ?? null;
  const secondHead = sortedHeads[1] ?? null;
  const scoreGap = getScoreQualityGap(topHead, secondHead, metricMetadata.direction);
  const metricLabel = getMetricLabel(metric, percentile);

  if (rankingQuery.isLoading && !rankingQuery.data) {
    return <SupportMessage>Loading head ranking...</SupportMessage>;
  }

  if (rankingQuery.data?.supported === false) {
    return <SupportMessage tone="amber">{rankingQuery.data.reason || 'Q3 head ranking is not available for this setting.'}</SupportMessage>;
  }

  if (sortedHeads.length === 0) {
    return <SupportMessage>No Q3 head ranking rows are available for this setting.</SupportMessage>;
  }

  return (
    <section className="space-y-4" data-testid="q3-report-head-ranking-view">
      <Q3ViewQuestion view="head-ranking" />
      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="grid gap-4 border-b border-slate-100 px-5 py-4 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-center">
          <div>
            <div className="text-sm font-bold uppercase tracking-[0.16em] text-slate-600">
              Head Ranking
            </div>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950">
              {model} {getVariantLabel(variant)} · layer{layer}
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-600">
              Head {topHead?.head} leads this setting by mean {metricLabel}
              {scoreGap !== null && secondHead
                ? `, ahead of Head ${secondHead.head} by ${formatMetricValue(metric, scoreGap)}.`
                : '.'}
            </p>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-800">Top head</div>
            <div className="mt-2 text-3xl font-bold text-cyan-950">H{topHead?.head}</div>
            <div className="mt-2 text-sm text-cyan-900">
              {metricLabel} {formatMetricValue(metric, topHead?.mean_score)} · mean rank {topHead?.mean_rank.toFixed(2)}
            </div>
            <div className="mt-1 text-xs text-cyan-800">
              Top-3 on {topHead?.top3_count}/{topHead?.image_count} images
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
              <tr>
                <th className="px-5 py-3">Score rank</th>
                <th className="px-5 py-3">Head</th>
                <th className="px-5 py-3 text-right">Mean score</th>
                <th className="px-5 py-3 text-right">Mean rank</th>
                <th className="px-5 py-3 text-right">Top-1 count</th>
                <th className="px-5 py-3 text-right">Top-3 count</th>
                <th className="px-5 py-3 text-right">Images</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sortedHeads.map((entry) => {
                const isTopHead = entry.scoreRank === 1;
                return (
                  <tr
                    key={entry.head}
                    className={isTopHead ? 'bg-cyan-50/70' : 'bg-white'}
                    data-testid={`q3-report-ranking-row-${entry.head}`}
                  >
                    <td className="px-5 py-3 font-semibold text-slate-900">#{entry.scoreRank}</td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex min-w-16 justify-center rounded-full border px-3 py-1 font-semibold ${
                        isTopHead
                          ? 'border-cyan-300 bg-white text-cyan-900'
                          : 'border-slate-200 bg-slate-50 text-slate-700'
                      }`}>
                        Head {entry.head}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right font-medium text-slate-900">
                      {formatMetricValue(metric, entry.mean_score)}
                    </td>
                    <td className="px-5 py-3 text-right text-slate-600">{entry.mean_rank.toFixed(2)}</td>
                    <td className="px-5 py-3 text-right text-slate-600">{entry.top1_count}</td>
                    <td className="px-5 py-3 text-right text-slate-600">{entry.top3_count}</td>
                    <td className="px-5 py-3 text-right text-slate-600">{entry.image_count}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function Q3ReportMatrixView({
  state,
  percentile,
  onStateChange,
}: {
  state: Q3ReportState;
  percentile: number;
  onStateChange: (patch: Partial<Q3ReportState>, replace?: boolean) => void;
}) {
  const metricMetadata = ANALYSIS_METRIC_METADATA[state.metric];
  const rankingQuery = useHeadRanking(state.model, state.layer, percentile, state.metric, state.variant);
  const matrixQuery = useHeadFeatureMatrix(state.model, state.layer, percentile, state.metric, state.variant);
  const rankedHeads = useMemo(
    () => sortHeadEntries(rankingQuery.data?.heads, metricMetadata.direction),
    [metricMetadata.direction, rankingQuery.data?.heads],
  );
  const matrixHeads = matrixQuery.data?.heads ?? [];
  const selectedHead = state.head ?? rankedHeads[0]?.head ?? matrixHeads[0] ?? 0;
  const selectedHeadIndex = Math.max(0, matrixHeads.indexOf(selectedHead));
  const features = matrixQuery.data?.features ?? [];
  const fallbackFeature = chooseDefaultFeature(features, selectedHeadIndex, metricMetadata.direction);
  const selectedFeature = (
    state.featureLabel !== null
      ? features.find((feature) => feature.feature_label === state.featureLabel)
      : null
  ) ?? fallbackFeature;
  const selectedFeatureLabel = selectedFeature?.feature_label ?? null;
  const selectedScore = selectedFeature ? getFeatureScore(selectedFeature, selectedHeadIndex) : null;
  const visibleFeatures = normalizeMatrixRows(features, selectedFeatureLabel, selectedHeadIndex, metricMetadata.direction);
  const matrixRange = getMatrixRange(features);
  const metricLabel = getMetricLabel(state.metric, percentile);

  if ((rankingQuery.isLoading && !rankingQuery.data) || (matrixQuery.isLoading && !matrixQuery.data)) {
    return <SupportMessage>Loading head-feature matrix...</SupportMessage>;
  }

  if (matrixQuery.data?.supported === false || rankingQuery.data?.supported === false) {
    return (
      <SupportMessage tone="amber">
        {matrixQuery.data?.reason || rankingQuery.data?.reason || 'Q3 head-feature data is not available for this setting.'}
      </SupportMessage>
    );
  }

  if (features.length === 0 || matrixHeads.length === 0) {
    return <SupportMessage>No Q3 head-feature matrix rows are available for this setting.</SupportMessage>;
  }

  return (
    <section className="space-y-4" data-testid="q3-report-matrix-view">
      <Q3ViewQuestion view="head-feature-matrix" />
      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="grid gap-4 border-b border-slate-100 px-5 py-4 xl:grid-cols-[minmax(0,1fr)_22rem] xl:items-stretch">
          <div>
            <div className="text-sm font-bold uppercase tracking-[0.16em] text-slate-600">
              Head-Feature Matrix
            </div>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950">
              {state.model} {getVariantLabel(state.variant)} · layer{state.layer}
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-600">
              Feature rows are sorted by the selected head, with one-off labels de-emphasized for the default cell.
            </p>

            <div className="mt-4 flex flex-wrap gap-2" data-testid="q3-report-head-strip">
              {rankedHeads.slice(0, 6).map((entry) => (
                <button
                  key={entry.head}
                  type="button"
                  onClick={() => onStateChange({ head: entry.head, featureLabel: null })}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors ${
                    entry.head === selectedHead
                      ? 'border-cyan-300 bg-cyan-50 text-cyan-900'
                      : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                  }`}
                >
                  #{entry.scoreRank} H{entry.head}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3" data-testid="q3-report-selected-cell">
            <div className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-800">Selected cell</div>
            <div className="mt-2 text-xl font-semibold text-cyan-950">
              {selectedFeature?.feature_name ?? 'No feature selected'}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-cyan-900">
              <div>
                <div className="text-xs uppercase tracking-[0.12em] text-cyan-700">Head</div>
                <div className="font-semibold">layer{state.layer}/head{selectedHead}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-[0.12em] text-cyan-700">{metricLabel}</div>
                <div className="font-semibold">{formatMetricValue(state.metric, selectedScore)}</div>
              </div>
              <div className="col-span-2">
                <div className="text-xs uppercase tracking-[0.12em] text-cyan-700">Annotations</div>
                <div className="font-semibold">{selectedFeature?.bbox_count ?? 0}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto" data-testid="q3-report-matrix-scroll">
          <table className="min-w-[76rem] border-separate border-spacing-0 text-sm">
            <thead className="bg-white">
              <tr>
                <th className="sticky left-0 z-20 border-b border-r border-slate-200 bg-white px-5 py-3 text-left text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
                  Feature
                </th>
                {matrixHeads.map((head) => (
                  <th
                    key={head}
                    className={`border-b border-slate-200 px-2 py-3 text-center text-xs font-semibold uppercase tracking-[0.08em] ${
                      head === selectedHead ? 'bg-cyan-50 text-cyan-900' : 'bg-white text-slate-500'
                    }`}
                  >
                    H{head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleFeatures.map((feature) => {
                const isSelectedRow = feature.feature_label === selectedFeatureLabel;
                return (
                  <tr key={feature.feature_label}>
                    <th
                      className={`sticky left-0 z-10 border-b border-r border-slate-200 px-5 py-3 text-left ${
                        isSelectedRow ? 'bg-cyan-50' : 'bg-white'
                      }`}
                    >
                      <div className="font-semibold text-slate-900">{feature.feature_name}</div>
                      <div className="mt-1 text-xs text-slate-500">{feature.bbox_count} annotations</div>
                    </th>
                    {matrixHeads.map((head, headIndex) => {
                      const score = getFeatureScore(feature, headIndex);
                      const isSelectedCell = head === selectedHead && isSelectedRow;
                      return (
                        <td
                          key={`${feature.feature_label}-${head}`}
                          className={`border-b border-slate-100 px-1.5 py-2 text-center ${
                            head === selectedHead ? 'bg-cyan-50/40' : 'bg-white'
                          }`}
                        >
                          <button
                            type="button"
                            onClick={() => {
                              if (score === null) return;
                              onStateChange({ head, featureLabel: feature.feature_label });
                            }}
                            disabled={score === null}
                            className={`h-11 w-20 rounded-md border text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
                              isSelectedCell
                                ? 'border-cyan-500 ring-2 ring-cyan-400 ring-offset-1'
                                : 'border-white/80 hover:scale-[1.02]'
                            } ${score === null ? 'cursor-not-allowed opacity-60' : ''}`}
                            style={getHeatmapCellStyle(score, matrixRange, metricMetadata.direction)}
                            data-testid={`q3-report-matrix-cell-${feature.feature_label}-${head}`}
                          >
                            {formatMetricValue(state.metric, score)}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function Q3ReportDeltaView({
  model,
  layer,
  metric,
  percentile,
}: {
  model: string;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
}) {
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const frozenRankingQuery = useHeadRanking(model, layer, percentile, metric, 'frozen');
  const loraRankingQuery = useHeadRanking(model, layer, percentile, metric, 'lora');
  const fullRankingQuery = useHeadRanking(model, layer, percentile, metric, 'full');
  const comparisons = useMemo(
    () => [
      buildDeltaComparison(frozenRankingQuery.data, loraRankingQuery.data, 'lora', metricMetadata.direction),
      buildDeltaComparison(frozenRankingQuery.data, fullRankingQuery.data, 'full', metricMetadata.direction),
    ].filter((comparison): comparison is DeltaComparison => comparison !== null),
    [frozenRankingQuery.data, fullRankingQuery.data, loraRankingQuery.data, metricMetadata.direction],
  );
  const metricLabel = getMetricLabel(metric, percentile);
  const isLoading = (
    (frozenRankingQuery.isLoading && !frozenRankingQuery.data)
    || (loraRankingQuery.isLoading && !loraRankingQuery.data)
    || (fullRankingQuery.isLoading && !fullRankingQuery.data)
  );

  if (isLoading) {
    return <SupportMessage>Loading frozen-to-adapted deltas...</SupportMessage>;
  }

  if (frozenRankingQuery.data?.supported === false) {
    return (
      <SupportMessage tone="amber">
        {frozenRankingQuery.data.reason || 'Frozen Q3 ranking data is not available for this setting.'}
      </SupportMessage>
    );
  }

  if (comparisons.length === 0) {
    return <SupportMessage>No frozen-to-adapted comparison rows are available for this setting.</SupportMessage>;
  }

  return (
    <section className="space-y-4" data-testid="q3-report-delta-view">
      <Q3ViewQuestion view="frozen-delta" />
      <div className="rounded-lg border border-slate-200 bg-white">
        <div className="border-b border-slate-100 px-5 py-4">
          <div className="text-sm font-bold uppercase tracking-[0.16em] text-slate-600">
            Frozen-to-Adapted Delta
          </div>
          <h2 className="mt-2 text-2xl font-semibold text-slate-950">
            {model} · layer{layer} · {metricLabel}
          </h2>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            Rank movement is computed from mean-score ordering inside this selected layer. Positive rank delta means the adapted variant promoted the head.
          </p>
        </div>

        <div className="grid gap-5 p-5 2xl:grid-cols-2">
          {comparisons.map((comparison) => (
            <DeltaComparisonPanel
              key={comparison.targetVariant}
              comparison={comparison}
              metric={metric}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function DeltaComparisonPanel({
  comparison,
  metric,
}: {
  comparison: DeltaComparison;
  metric: AnalysisMetric;
}) {
  const topFrozenLabel = comparison.topFrozen ? `H${comparison.topFrozen.head}` : 'n/a';
  const topAdaptedLabel = comparison.topAdapted ? `H${comparison.topAdapted.head}` : 'n/a';
  const headline = topFrozenLabel === topAdaptedLabel
    ? `Preserves ${topFrozenLabel} as the top head.`
    : `Moves the top head from ${topFrozenLabel} to ${topAdaptedLabel}.`;

  return (
    <div
      className="rounded-lg border border-slate-200 bg-white"
      data-testid={`q3-report-delta-${comparison.targetVariant}`}
    >
      <div className="border-b border-slate-100 px-4 py-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-950">{comparison.label}</h3>
            <p className="mt-1 text-sm text-slate-600">{comparison.available ? headline : comparison.reason}</p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2 xl:grid-cols-5">
          <DeltaSummaryPill label="Promoted" value={comparison.summary.promoted} state="promoted" />
          <DeltaSummaryPill label="Demoted" value={comparison.summary.demoted} state="demoted" />
          <DeltaSummaryPill label="Stable" value={comparison.summary.stable} state="stable" />
          <DeltaSummaryPill label="Top frozen" value={topFrozenLabel} />
          <DeltaSummaryPill label="Top adapted" value={topAdaptedLabel} />
        </div>
      </div>

      {!comparison.available ? (
        <div className="px-4 py-4 text-sm text-amber-900">
          {comparison.reason}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
              <tr>
                <th className="px-4 py-3">Head</th>
                <th className="px-4 py-3 text-right">Frozen rank</th>
                <th className="px-4 py-3 text-right">Adapted rank</th>
                <th className="px-4 py-3 text-right">Rank delta</th>
                <th className="px-4 py-3 text-right">Frozen score</th>
                <th className="px-4 py-3 text-right">Adapted score</th>
                <th className="px-4 py-3 text-right">Score delta</th>
                <th className="px-4 py-3 text-right">State</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparison.rows.map((row) => (
                <tr key={row.head}>
                  <td className="px-4 py-3 font-semibold text-slate-900">H{row.head}</td>
                  <td className="px-4 py-3 text-right text-slate-600">{formatRank(row.frozenRank)}</td>
                  <td className="px-4 py-3 text-right text-slate-600">{formatRank(row.adaptedRank)}</td>
                  <td className="px-4 py-3 text-right">
                    <span className={`inline-flex min-w-12 justify-center rounded-full border px-2 py-0.5 text-xs font-semibold ${getDeltaStateClassName(row.state)}`}>
                      {formatRankDelta(row.rankDelta)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-slate-600">
                    {formatMetricValue(metric, row.frozenScore)}
                  </td>
                  <td className="px-4 py-3 text-right text-slate-600">
                    {formatMetricValue(metric, row.adaptedScore)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`inline-flex min-w-16 justify-center rounded px-2 py-0.5 text-xs font-semibold ${getScoreDeltaClassName(metric, row.scoreDelta)}`}>
                      {formatMetricValue(metric, row.scoreDelta, { signed: true })}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${getDeltaStateClassName(row.state)}`}>
                      {row.state === null ? 'n/a' : row.state}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function DeltaSummaryPill({
  label,
  value,
  state,
}: {
  label: string;
  value: number | string;
  state?: DeltaState;
}) {
  const className = state ? getDeltaStateClassName(state) : 'border-slate-200 bg-slate-50 text-slate-800';

  return (
    <div className={`rounded-lg border px-3 py-2 ${className}`}>
      <div className="text-[0.65rem] font-semibold uppercase tracking-[0.12em] opacity-80">{label}</div>
      <div className="mt-1 text-lg font-bold">{value}</div>
    </div>
  );
}
