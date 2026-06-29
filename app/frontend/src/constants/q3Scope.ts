import type { AnalysisMetric, CompareVariantId, ImageDetailMode } from '../types';

export type Q3ScopeStatus = 'primary' | 'control' | 'outside';

export const Q3_PRIMARY_MODELS = ['dinov2', 'dinov3', 'mae', 'clip'] as const;
export const Q3_OUTSIDE_MODELS = ['siglip', 'siglip2', 'resnet50'] as const;
export const Q3_PRIMARY_VARIANTS: CompareVariantId[] = ['frozen', 'lora', 'full'];
export const Q3_CONTROL_VARIANT: CompareVariantId = 'linear_probe';
export const Q3_VARIANT_OPTIONS: Array<{ value: CompareVariantId; label: string }> = [
  { value: 'frozen', label: 'Frozen' },
  { value: 'lora', label: 'LoRA' },
  { value: 'full', label: 'Full Fine-tune' },
  { value: 'linear_probe', label: 'Linear Probe' },
];

export const Q3_SCOPE_COPY = {
  title: 'Primary Q3 workflow',
  dashboardSummary:
    'Start on Dashboard Q3 to compare candidate heads across dinov2, dinov3, mae, and clip.',
  dashboardDetail:
    'Use the heatmap and inline exemplar panel to open one concrete image in Image Detail Q3 with the same variant, layer, head, and feature context already loaded.',
  dashboardSelectionNote:
    'Frozen, LoRA, and Full are the headline comparison set. Linear Probe remains available as a control condition.',
  imageDetailSummary:
    'Image Detail Q3 is the qualitative drill-down step for a finding you selected on Dashboard Q3.',
  imageDetailDetail:
    'Stay inside the selected model, variant, layer, and head context while you inspect a representative image and its annotations.',
  imageDetailCurrentContext:
    'This drill-down keeps the dashboard context loaded so you can inspect the chosen exemplar without re-entering the Q3 state by hand.',
  scopeNote:
    'The broader model set still stays available on the non-Q3 exploration surfaces.',
} as const;

export const Q3_DEFAULTS: {
  model: string;
  method: string;
  variant: CompareVariantId;
  layer: number;
  head: number | null;
  metric: AnalysisMetric;
  percentile: number;
  mode: ImageDetailMode;
  showBboxes: boolean;
} = {
  model: 'dinov2',
  method: 'cls',
  variant: 'frozen',
  layer: 11,
  head: null,
  metric: 'iou',
  percentile: 90,
  mode: 'head_attention',
  showBboxes: true,
};

const Q3_SCOPE_LABELS: Record<Q3ScopeStatus, string> = {
  primary: 'Primary study',
  control: 'Control',
  outside: 'Outside primary scope',
};

export function getQ3ModelScopeStatus(model: string): Q3ScopeStatus {
  return Q3_PRIMARY_MODELS.includes(model as (typeof Q3_PRIMARY_MODELS)[number])
    ? 'primary'
    : 'outside';
}

export function getQ3VariantScopeStatus(variant: CompareVariantId): Q3ScopeStatus {
  return variant === Q3_CONTROL_VARIANT ? 'control' : 'primary';
}

export function getQ3ScopeLabel(status: Q3ScopeStatus): string {
  return Q3_SCOPE_LABELS[status];
}

export function getQ3ScopeChipClassName(status: Q3ScopeStatus): string {
  switch (status) {
    case 'primary':
      return 'border-emerald-200 bg-emerald-50 text-emerald-800';
    case 'control':
      return 'border-amber-200 bg-amber-50 text-amber-800';
    case 'outside':
      return 'border-slate-200 bg-slate-100 text-slate-700';
    default:
      return 'border-slate-200 bg-slate-100 text-slate-700';
  }
}

export function formatQ3ScopeOptionLabel(label: string, status: Q3ScopeStatus): string {
  return `${label} (${getQ3ScopeLabel(status)})`;
}

export function getQ3SelectionHelperText(
  modelStatus: Q3ScopeStatus,
  variantStatus?: Q3ScopeStatus,
): string {
  if (modelStatus === 'outside') {
    return 'This selection falls outside the primary Q3 workflow and is only available on the broader exploration surfaces.';
  }

  if (variantStatus === 'control') {
    return 'Linear Probe remains visible as a control rather than a peer headline comparison condition.';
  }

  return 'This selection stays inside the primary Q3 workflow.';
}
