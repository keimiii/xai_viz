/**
 * Layer progression slider with animation support.
 */

import { useEffect, useRef } from 'react';
import { Slider } from '../ui/Slider';

interface LayerSliderProps {
  currentLayer: number;
  maxLayers?: number;
  onChange: (layer: number) => void;
  isPlaying: boolean;
  onPlayingChange: (isPlaying: boolean) => void;
  playSpeed?: number; // ms between frames
}

export function LayerSlider({
  currentLayer,
  maxLayers = 12,
  onChange,
  isPlaying,
  onPlayingChange,
  playSpeed = 500,
}: LayerSliderProps) {
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        // Stop at last layer instead of looping
        if (currentLayer >= maxLayers - 1) {
          onPlayingChange(false);
          return;
        }
        onChange(currentLayer + 1);
      }, playSpeed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, currentLayer, maxLayers, playSpeed, onChange, onPlayingChange]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4">
        <button
          type="button"
          data-testid="layer-play-toggle"
          onClick={() => {
            // Reset to layer 0 when at end and not playing
            if (currentLayer >= maxLayers - 1 && !isPlaying) {
              onChange(0);
            }
            onPlayingChange(!isPlaying);
          }}
          className="px-3 py-1 text-sm font-medium bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button
          type="button"
          data-testid="layer-first"
          onClick={() => onChange(0)}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          |&lt;
        </button>

        <button
          type="button"
          data-testid="layer-prev"
          onClick={() => onChange(Math.max(0, currentLayer - 1))}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &lt;
        </button>

        <button
          type="button"
          data-testid="layer-next"
          onClick={() => onChange(Math.min(maxLayers - 1, currentLayer + 1))}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &gt;
        </button>

        <button
          type="button"
          data-testid="layer-last"
          onClick={() => onChange(maxLayers - 1)}
          className="px-2 py-1 text-sm text-gray-600 hover:text-gray-900"
          disabled={isPlaying}
        >
          &gt;|
        </button>
      </div>

      <Slider
        value={currentLayer}
        onChange={onChange}
        min={0}
        max={maxLayers - 1}
        label={`Layer ${currentLayer}`}
        showValue={false}
      />

      <div className="flex justify-between text-xs text-gray-500">
        <span>Early (L0)</span>
        <span>Late (L{maxLayers - 1})</span>
      </div>
    </div>
  );
}
