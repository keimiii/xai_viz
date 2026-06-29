import { useMemo } from 'react';

import { useHeadRanking } from '../../hooks/useMetrics';
import { ANALYSIS_METRIC_METADATA, formatMetricValue, metricImprovementTone } from '../../constants/metricMetadata';
import type {
  AnalysisMetric,
  HeadRankingEntry,
  HeadRankingResponse,
} from '../../types';

type Q3DeltaState = 'promoted' | 'demoted' | 'stable';
type Q3DeltaTargetVariant = 'lora' | 'full';

interface Q3VariantDeltaRow {
  head: number;
  frozenRank: number | null;
  adaptedRank: number | null;
  rankDelta: number | null;
  frozenScore: number | null;
  adaptedScore: number | null;
  scoreDelta: number | null;
  state: Q3DeltaState | null;
  complete: boolean;
}

interface Q3VariantDeltaComparison {
  targetVariant: Q3DeltaTargetVariant;
  label: string;
  available: boolean;
  reason: string | null;
  rows: Q3VariantDeltaRow[];
  summary: Record<Q3DeltaState, number>;
}

interface Q3DeltaPanelProps {
  model: string;
  layer: number;
  metric: AnalysisMetric;
  percentile: number;
  title?: string;
  description?: string;
  helperText?: string;
  testIdPrefix?: string;
}

function getDeltaState(rankDelta: number | null): Q3DeltaState | null {
  if (rankDelta === null) {
    return null;
  }
  if (rankDelta > 0) {
    return 'promoted';
  }
  if (rankDelta < 0) {
    return 'demoted';
  }
  return 'stable';
}

function getDeltaStateClassName(state: Q3DeltaState | null): string {
  if (state === 'promoted') {
    return 'border-green-200 bg-green-50 text-green-700';
  }
  if (state === 'demoted') {
    return 'border-rose-200 bg-rose-50 text-rose-700';
  }
  if (state === 'stable') {
    return 'border-slate-200 bg-slate-100 text-slate-700';
  }
  return 'border-slate-200 bg-slate-50 text-slate-500';
}

function formatRankValue(rank: number | null): string {
  return rank === null ? 'n/a' : String(rank);
}

function formatRankDelta(rankDelta: number | null): string {
  if (rankDelta === null) {
    return 'n/a';
  }
  return `${rankDelta > 0 ? '+' : ''}${rankDelta}`;
}

function toneClassFromDelta(metric: AnalysisMetric, delta: number | null | undefined): string {
  const tone = metricImprovementTone(ANALYSIS_METRIC_METADATA[metric].direction, delta);
  if (tone === 'positive') {
    return 'text-green-700 bg-green-50';
  }
  if (tone === 'negative') {
    return 'text-rose-700 bg-rose-50';
  }
  return 'text-slate-600 bg-slate-100';
}

function buildHeadRankingMap(response: HeadRankingResponse | undefined): Map<number, { rank: number; entry: HeadRankingEntry }> {
  const rankingMap = new Map<number, { rank: number; entry: HeadRankingEntry }>();
  for (const [index, entry] of (response?.heads ?? []).entries()) {
    rankingMap.set(entry.head, {
      rank: index + 1,
      entry,
    });
  }
  return rankingMap;
}

function getDeltaSortMagnitude(value: number | null): number {
  return value === null ? -1 : Math.abs(value);
}

function buildQ3VariantDeltaComparison(
  frozenResponse: HeadRankingResponse | undefined,
  adaptedResponse: HeadRankingResponse | undefined,
  targetVariant: Q3DeltaTargetVariant,
): Q3VariantDeltaComparison | null {
  if (!frozenResponse || !adaptedResponse) {
    return null;
  }

  const label = targetVariant === 'lora' ? 'Frozen -> LoRA' : 'Frozen -> Full';
  if (!adaptedResponse.supported) {
    return {
      targetVariant,
      label,
      available: false,
      reason: adaptedResponse.reason || `${label} data is not available for this selection.`,
      rows: [],
      summary: {
        promoted: 0,
        demoted: 0,
        stable: 0,
      },
    };
  }

  const frozenRankingMap = buildHeadRankingMap(frozenResponse);
  const adaptedRankingMap = buildHeadRankingMap(adaptedResponse);
  const headIds = Array.from(new Set([...frozenRankingMap.keys(), ...adaptedRankingMap.keys()]));

  const rows = headIds.map((head) => {
    const frozenEntry = frozenRankingMap.get(head);
    const adaptedEntry = adaptedRankingMap.get(head);
    const complete = Boolean(frozenEntry && adaptedEntry);
    const rankDelta = complete ? frozenEntry!.rank - adaptedEntry!.rank : null;
    const scoreDelta = complete ? adaptedEntry!.entry.mean_score - frozenEntry!.entry.mean_score : null;

    return {
      head,
      frozenRank: frozenEntry?.rank ?? null,
      adaptedRank: adaptedEntry?.rank ?? null,
      rankDelta,
      frozenScore: frozenEntry?.entry.mean_score ?? null,
      adaptedScore: adaptedEntry?.entry.mean_score ?? null,
      scoreDelta,
      state: getDeltaState(rankDelta),
      complete,
    };
  });

  rows.sort((left, right) => {
    if (left.complete !== right.complete) {
      return left.complete ? -1 : 1;
    }

    const rankMovementDifference = getDeltaSortMagnitude(right.rankDelta) - getDeltaSortMagnitude(left.rankDelta);
    if (rankMovementDifference !== 0) {
      return rankMovementDifference;
    }

    const scoreMovementDifference = getDeltaSortMagnitude(right.scoreDelta) - getDeltaSortMagnitude(left.scoreDelta);
    if (scoreMovementDifference !== 0) {
      return scoreMovementDifference;
    }

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
  };
}

function makeTestId(prefix: string, suffix: string): string {
  return `${prefix}-${suffix}`;
}

export function Q3DeltaPanel({
  model,
  layer,
  metric,
  percentile,
  title = 'Frozen-to-adapted head delta',
  description = 'Compare how dominant heads move from the frozen baseline into the primary adapted variants for the current model, layer, metric, and percentile.',
  helperText = 'This delta view always compares Frozen to LoRA and Full. Linear Probe remains a control in the single-variant analysis below.',
  testIdPrefix = 'q3',
}: Q3DeltaPanelProps) {
  const metricMetadata = ANALYSIS_METRIC_METADATA[metric];
  const effectivePercentile = metricMetadata.thresholdFree ? 90 : percentile;

  const frozenRankingQuery = useHeadRanking(model, layer, effectivePercentile, metric, 'frozen');
  const loraRankingQuery = useHeadRanking(model, layer, effectivePercentile, metric, 'lora');
  const fullRankingQuery = useHeadRanking(model, layer, effectivePercentile, metric, 'full');

  const deltaPanelLoading = (
    (frozenRankingQuery.isLoading && !frozenRankingQuery.data)
    || (loraRankingQuery.isLoading && !loraRankingQuery.data)
    || (fullRankingQuery.isLoading && !fullRankingQuery.data)
  );
  const frozenDeltaUnavailable = frozenRankingQuery.data?.supported === false;
  const frozenDeltaUnavailableReason = frozenRankingQuery.data?.reason
    || 'Frozen Q3 ranking data is not available for this selection.';

  const loraDeltaComparison = useMemo(
    () => buildQ3VariantDeltaComparison(frozenRankingQuery.data, loraRankingQuery.data, 'lora'),
    [frozenRankingQuery.data, loraRankingQuery.data],
  );
  const fullDeltaComparison = useMemo(
    () => buildQ3VariantDeltaComparison(frozenRankingQuery.data, fullRankingQuery.data, 'full'),
    [frozenRankingQuery.data, fullRankingQuery.data],
  );
  const deltaComparisons = [loraDeltaComparison, fullDeltaComparison].filter(
    (comparison): comparison is Q3VariantDeltaComparison => comparison !== null,
  );

  return (
    <section
      className="space-y-4"
      data-testid={makeTestId(testIdPrefix, 'delta-panel')}
    >
      <div className="rounded-lg border border-indigo-100 bg-indigo-50 px-4 py-3">
        <div className="text-sm font-medium text-indigo-950">{title}</div>
        <p className="mt-1 text-sm text-indigo-900">{description}</p>
        <p className="mt-2 text-xs text-indigo-800" data-testid={makeTestId(testIdPrefix, 'delta-helper')}>
          {helperText}
        </p>
      </div>

      {deltaPanelLoading ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
          Loading frozen-to-adapted comparisons...
        </div>
      ) : frozenDeltaUnavailable ? (
        <div
          className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
          data-testid={makeTestId(testIdPrefix, 'delta-unavailable')}
        >
          {frozenDeltaUnavailableReason}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
          {deltaComparisons.map((comparison) => (
            <div
              key={comparison.targetVariant}
              className="rounded-lg border border-slate-200 bg-white"
              data-testid={makeTestId(testIdPrefix, `delta-card-${comparison.targetVariant}`)}
            >
              <div className="border-b border-slate-100 px-4 py-3">
                <div className="font-medium text-slate-900">{comparison.label}</div>
                <p className="mt-1 text-xs text-slate-500">
                  Positive rank Δ means the head moved up in the adapted ordering.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <span
                    className="inline-flex items-center rounded-full border border-green-200 bg-green-50 px-2.5 py-1 text-xs font-medium text-green-700"
                    data-testid={makeTestId(testIdPrefix, `delta-summary-${comparison.targetVariant}-promoted`)}
                  >
                    Promoted {comparison.summary.promoted}
                  </span>
                  <span
                    className="inline-flex items-center rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-xs font-medium text-rose-700"
                    data-testid={makeTestId(testIdPrefix, `delta-summary-${comparison.targetVariant}-demoted`)}
                  >
                    Demoted {comparison.summary.demoted}
                  </span>
                  <span
                    className="inline-flex items-center rounded-full border border-slate-200 bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
                    data-testid={makeTestId(testIdPrefix, `delta-summary-${comparison.targetVariant}-stable`)}
                  >
                    Stable {comparison.summary.stable}
                  </span>
                </div>
              </div>

              {!comparison.available ? (
                <div
                  className="px-4 py-4 text-sm text-amber-900"
                  data-testid={makeTestId(testIdPrefix, `delta-card-${comparison.targetVariant}-unavailable`)}
                >
                  {comparison.reason}
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="border-b border-slate-200 text-left text-xs text-slate-500">
                      <tr>
                        <th className="px-4 py-2 font-medium">Head</th>
                        <th className="px-4 py-2 font-medium text-right">Frozen rank</th>
                        <th className="px-4 py-2 font-medium text-right">Adapted rank</th>
                        <th className="px-4 py-2 font-medium text-right">Rank Δ</th>
                        <th className="px-4 py-2 font-medium text-right">Frozen score</th>
                        <th className="px-4 py-2 font-medium text-right">Adapted score</th>
                        <th className="px-4 py-2 font-medium text-right">Score Δ</th>
                        <th className="px-4 py-2 font-medium text-right">State</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {comparison.rows.map((row) => (
                        <tr
                          key={row.head}
                          data-testid={makeTestId(testIdPrefix, `delta-row-${comparison.targetVariant}-${row.head}`)}
                        >
                          <td className="px-4 py-2 font-medium text-slate-900">Head {row.head}</td>
                          <td className="px-4 py-2 text-right text-slate-600">{formatRankValue(row.frozenRank)}</td>
                          <td className="px-4 py-2 text-right text-slate-600">{formatRankValue(row.adaptedRank)}</td>
                          <td className="px-4 py-2 text-right">
                            <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${getDeltaStateClassName(row.state)}`}>
                              {formatRankDelta(row.rankDelta)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right text-slate-600">
                            {formatMetricValue(metric, row.frozenScore)}
                          </td>
                          <td className="px-4 py-2 text-right text-slate-600">
                            {formatMetricValue(metric, row.adaptedScore)}
                          </td>
                          <td className="px-4 py-2 text-right">
                            <span
                              className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${toneClassFromDelta(metric, row.scoreDelta)}`}
                            >
                              {formatMetricValue(metric, row.scoreDelta, { signed: true })}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right">
                            <span
                              className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${getDeltaStateClassName(row.state)}`}
                            >
                              {row.state === null
                                ? 'n/a'
                                : row.state.charAt(0).toUpperCase() + row.state.slice(1)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
