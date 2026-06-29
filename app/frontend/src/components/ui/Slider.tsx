/**
 * Range slider component.
 */

import { Tooltip } from './Tooltip';

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  label?: string;
  tooltip?: string;
  showValue?: boolean;
  className?: string;
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step = 1,
  label,
  tooltip,
  showValue = true,
  className = '',
}: SliderProps) {
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {(label || showValue) && (
        <div className="flex justify-between items-center">
          {label && (
            <div className="flex items-center">
              <label className="text-sm font-medium text-gray-700">{label}</label>
              {tooltip && <Tooltip content={tooltip} />}
            </div>
          )}
          {showValue && (
            <span className="text-sm text-gray-500">{value}</span>
          )}
        </div>
      )}
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
      />
    </div>
  );
}
