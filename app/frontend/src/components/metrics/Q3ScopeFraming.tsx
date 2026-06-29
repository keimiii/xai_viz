import type { ReactNode } from 'react';

import { Q3_SCOPE_COPY, getQ3ScopeChipClassName, getQ3ScopeLabel, type Q3ScopeStatus } from '../../constants/q3Scope';

interface Q3ScopeChipProps {
  status: Q3ScopeStatus;
  className?: string;
  children?: ReactNode;
  dataTestId?: string;
}

export function Q3ScopeChip({
  status,
  className = '',
  children,
  dataTestId,
}: Q3ScopeChipProps) {
  return (
    <span
      data-testid={dataTestId}
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${getQ3ScopeChipClassName(status)} ${className}`}
    >
      {children ?? getQ3ScopeLabel(status)}
    </span>
  );
}

interface Q3StudyScopeCalloutProps {
  context: 'dashboard' | 'imageDetail';
  className?: string;
  dataTestId?: string;
  currentModelLabel?: string;
  currentModelStatus?: Q3ScopeStatus;
  action?: {
    label: string;
    onClick: () => void;
    dataTestId?: string;
  };
}

export function Q3StudyScopeCallout({
  context,
  className = '',
  dataTestId,
  currentModelLabel,
  currentModelStatus,
  action,
}: Q3StudyScopeCalloutProps) {
  const summary = context === 'dashboard'
    ? Q3_SCOPE_COPY.dashboardSummary
    : Q3_SCOPE_COPY.imageDetailSummary;
  const detail = context === 'dashboard'
    ? Q3_SCOPE_COPY.dashboardDetail
    : Q3_SCOPE_COPY.imageDetailDetail;

  return (
    <div
      data-testid={dataTestId}
      className={`rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 ${className}`}
    >
      <div className="space-y-3">
        <div className="space-y-1">
          <div className="font-semibold text-slate-900">{Q3_SCOPE_COPY.title}</div>
          <p>{summary}</p>
          <p>{detail}</p>
          <p>{Q3_SCOPE_COPY.scopeNote}</p>
        </div>

        <div className="flex flex-wrap gap-2" data-testid={dataTestId ? `${dataTestId}-legend` : undefined}>
          <Q3ScopeChip status="primary" />
          <Q3ScopeChip status="control" />
        </div>

        {context === 'imageDetail' && currentModelLabel && currentModelStatus && (
          <div
            className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white px-3 py-3"
            data-testid="image-detail-q3-current-model"
          >
            <div className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
              Current model
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-slate-900">{currentModelLabel}</span>
              <Q3ScopeChip
                status={currentModelStatus}
                dataTestId="image-detail-q3-current-model-status"
              />
            </div>
            <p className="text-xs text-slate-600">{Q3_SCOPE_COPY.imageDetailCurrentContext}</p>
          </div>
        )}

        {action && (
          <div className="flex justify-start">
            <button
              type="button"
              onClick={action.onClick}
              data-testid={action.dataTestId}
              className="rounded-md border border-primary-200 bg-white px-3 py-2 text-sm font-medium text-primary-700 transition-colors hover:border-primary-300 hover:bg-primary-50"
            >
              {action.label}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
