import { Card, CardContent } from '../ui/Card';
import { IMAGE_DETAIL_MODE_OPTIONS } from '../../constants/imageDetailModes';
import type { ImageDetailMode } from '../../types';

interface ImageDetailModeSwitchProps {
  mode: ImageDetailMode;
  onChange: (mode: ImageDetailMode) => void;
}

const MODE_SUMMARY: Record<ImageDetailMode, string> = {
  head_attention:
    'Inspect fused or per-head attention overlays. Bounding boxes stay available as visual context, but similarity is not the active interpretation in this mode.',
  feature_similarity:
    'Select a bounding box to use it as the similarity query. This mode focuses on bbox-conditioned feature similarity while keeping the selected head context available in the Q3 controls.',
};

export function ImageDetailModeSwitch({
  mode,
  onChange,
}: ImageDetailModeSwitchProps) {
  return (
    <div data-testid="image-detail-mode-switch">
      <Card>
        <CardContent className="space-y-3">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold text-gray-900">Interpretation mode</p>
              <p className="text-xs text-gray-500">Choose the visual explanation you want to inspect.</p>
            </div>
            <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
              {IMAGE_DETAIL_MODE_OPTIONS.map((option) => {
                const isSelected = option.value === mode;
                return (
                  <button
                    key={option.value}
                    type="button"
                    data-testid={`image-detail-mode-${option.value}`}
                    aria-pressed={isSelected}
                    onClick={() => onChange(option.value)}
                    className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                      isSelected
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div
            className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
            data-testid="image-detail-mode-helper"
          >
            {MODE_SUMMARY[mode]}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
