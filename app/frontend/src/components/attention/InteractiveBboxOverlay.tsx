/**
 * Interactive SVG overlay for clicking on bounding boxes.
 */

import type { BoundingBox } from '../../types';

interface InteractiveBboxOverlayProps {
  bboxes: BoundingBox[];
  selectedIndex: number | null;
  onBboxClick: (bbox: BoundingBox, index: number) => void;
  className?: string;
}

export function InteractiveBboxOverlay({
  bboxes,
  selectedIndex,
  onBboxClick,
  className = '',
}: InteractiveBboxOverlayProps) {
  return (
    <svg
      className={`absolute inset-0 w-full h-full pointer-events-none ${className}`}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
    >
      {bboxes.map((bbox, index) => {
        const isSelected = selectedIndex === index;

        return (
          <g key={index}>
            {/* Clickable area - slightly larger for easier clicking */}
            <rect
              x={bbox.left}
              y={bbox.top}
              width={bbox.width}
              height={bbox.height}
              fill="transparent"
              data-testid={`bbox-hitbox-${index}`}
              className="pointer-events-auto cursor-pointer"
              onClick={(e) => {
                e.stopPropagation();
                onBboxClick(bbox, index);
              }}
            />

            {/* Visual border */}
            <rect
              x={bbox.left}
              y={bbox.top}
              width={bbox.width}
              height={bbox.height}
              fill="none"
              stroke={isSelected ? '#22c55e' : '#3b82f6'}
              strokeWidth={isSelected ? 0.008 : 0.004}
              strokeDasharray={isSelected ? 'none' : '0.02 0.01'}
              className="pointer-events-none transition-all duration-200"
              style={{
                filter: isSelected ? 'drop-shadow(0 0 2px rgba(34, 197, 94, 0.8))' : 'none',
              }}
            />

            {/* Label badge for selected bbox */}
            {isSelected && bbox.label_name && (
              <g>
                <rect
                  x={bbox.left}
                  y={Math.max(0, bbox.top - 0.04)}
                  width={Math.min(0.25, bbox.label_name.length * 0.012 + 0.02)}
                  height={0.035}
                  fill="#22c55e"
                  rx={0.005}
                />
                <text
                  x={bbox.left + 0.01}
                  y={Math.max(0.025, bbox.top - 0.015)}
                  fill="white"
                  fontSize={0.022}
                  fontWeight="bold"
                  className="pointer-events-none"
                >
                  {bbox.label_name}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
