import { Card, CardContent, CardHeader } from '../ui/Card';
import type { ImageAnnotation, ImageDetailMode } from '../../types';

interface AnnotationsCardProps {
  annotation: ImageAnnotation;
  mode: ImageDetailMode;
  showBboxes: boolean;
  bboxSelectionDrivesOverlay?: boolean;
  selectedBboxIndex: number | null;
  onBboxSelect: (index: number | null) => void;
}

export function AnnotationsCard({
  annotation,
  mode,
  showBboxes,
  bboxSelectionDrivesOverlay = false,
  selectedBboxIndex,
  onBboxSelect,
}: AnnotationsCardProps) {
  const selectedBbox = selectedBboxIndex !== null ? annotation.bboxes[selectedBboxIndex] : null;
  const helperCopy = getHelperCopy(mode, showBboxes, bboxSelectionDrivesOverlay, selectedBbox?.label_name ?? null);

  return (
    <div data-testid="annotations-card">
      <Card>
        <CardHeader>
          <h3 className="font-semibold">Annotations</h3>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <span className="text-sm text-gray-500">Styles:</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {annotation.style_names.map((style) => (
                <span
                  key={style}
                  className="rounded bg-primary-100 px-2 py-0.5 text-sm text-primary-700"
                >
                  {style}
                </span>
              ))}
            </div>
          </div>

          <div>
            <span className="text-sm text-gray-500">Bounding Boxes:</span>
            <span className="ml-2 font-medium">{annotation.num_bboxes}</span>
          </div>

          <div
            className={`rounded px-2 py-1 text-xs ${helperCopy.className}`}
            data-testid="annotations-helper-copy"
          >
            {helperCopy.text}
          </div>

          <div className="max-h-48 space-y-1 overflow-y-auto text-xs text-gray-500">
            {annotation.bboxes.map((bbox, index) => {
              const isSelected = selectedBboxIndex === index;
              return (
                <button
                  key={`${bbox.label}-${index}`}
                  type="button"
                  data-testid={`bbox-list-item-${index}`}
                  className={`flex w-full items-center justify-between rounded px-2 py-1 text-left transition-colors ${
                    showBboxes ? 'hover:bg-gray-100' : 'cursor-default'
                  } ${isSelected ? 'bg-green-100 text-green-700' : ''}`}
                  onClick={() => {
                    if (!showBboxes) return;
                    onBboxSelect(isSelected ? null : index);
                  }}
                >
                  <span>{bbox.label_name || `Label ${bbox.label}`}</span>
                  <span className="text-gray-400">
                    {(bbox.width * 100).toFixed(0)}% x {(bbox.height * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function getHelperCopy(
  mode: ImageDetailMode,
  showBboxes: boolean,
  bboxSelectionDrivesOverlay: boolean,
  selectedLabel: string | null,
) {
  if (!showBboxes) {
    return {
      className: 'bg-amber-50 text-amber-800',
      text: 'Turn on Show Bounding Boxes to interact with annotated features from this card or the image viewer.',
    };
  }

  if (mode === 'feature_similarity') {
    return selectedLabel
      ? {
          className: 'bg-green-50 text-green-700',
          text: `${selectedLabel} is driving the feature-similarity overlay. Click it again to clear or choose a different feature.`,
        }
      : {
          className: 'bg-green-50 text-green-700',
          text: 'Click a bounding box to use it as the feature-similarity query.',
        };
  }

  if (bboxSelectionDrivesOverlay) {
    return selectedLabel
      ? {
          className: 'bg-green-50 text-green-700',
          text: `${selectedLabel} is driving the focused overlay. Click it again to return to the global attention view.`,
        }
      : {
          className: 'bg-green-50 text-green-700',
          text: 'Click a bounding box to swap the global attention view for a focused overlay around that feature.',
        };
  }

  return selectedLabel
    ? {
        className: 'bg-sky-50 text-sky-700',
        text: `${selectedLabel} stays highlighted as context while you inspect attention. Switch to Feature Similarity for bbox-conditioned similarity overlays.`,
      }
    : {
        className: 'bg-sky-50 text-sky-700',
        text: 'Click a bounding box to keep that feature highlighted as context while you inspect attention.',
      };
}
