export const ATTENTION_METHOD_LABELS: Record<string, string> = {
  cls: 'CLS Attention',
  rollout: 'Rollout Attention',
  gradcam: 'Grad-CAM',
  mean: 'Mean Attention',
};

export function getAttentionMethodLabel(method: string): string {
  return ATTENTION_METHOD_LABELS[method] || method;
}
