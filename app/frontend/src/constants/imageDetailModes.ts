import type { ImageDetailMode } from '../types';

export const DEFAULT_IMAGE_DETAIL_MODE: ImageDetailMode = 'head_attention';

export const IMAGE_DETAIL_MODE_OPTIONS: Array<{ value: ImageDetailMode; label: string }> = [
  { value: 'head_attention', label: 'Head Attention' },
  { value: 'feature_similarity', label: 'Feature Similarity' },
];

export function isImageDetailMode(value: string | null | undefined): value is ImageDetailMode {
  return value === 'head_attention' || value === 'feature_similarity';
}

export function parseImageDetailMode(value: string | null | undefined): ImageDetailMode {
  return isImageDetailMode(value) ? value : DEFAULT_IMAGE_DETAIL_MODE;
}

export function getImageDetailModeLabel(mode: ImageDetailMode): string {
  return mode === 'feature_similarity' ? 'Feature Similarity' : 'Head Attention';
}
