/**
 * Select dropdown component.
 */

import { useId } from 'react';
import { Tooltip } from './Tooltip';

interface SelectProps {
  value: string | number;
  onChange: (value: string) => void;
  options: Array<{ value: string | number; label: string }>;
  label?: string;
  tooltip?: string;
  className?: string;
  disabled?: boolean;
}

export function Select({
  value,
  onChange,
  options,
  label,
  tooltip,
  className = '',
  disabled = false,
}: SelectProps) {
  const generatedId = useId();
  const selectId = label ? `select-${generatedId}` : undefined;

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {label && (
        <div className="flex items-center">
          <label htmlFor={selectId} className="text-sm font-medium text-gray-700">{label}</label>
          {tooltip && <Tooltip content={tooltip} />}
        </div>
      )}
      <select
        id={selectId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-500"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
