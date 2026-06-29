/**
 * Tooltip component for displaying help information on hover.
 *
 * Uses a React portal so the popup escapes ancestor overflow:hidden
 * containers (e.g. Card).
 */

import { useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  content: string;
  /** Horizontal alignment of the popup relative to the "?" button. */
  align?: 'center' | 'left' | 'right';
  /** Optional custom trigger. If omitted, renders the default "?" button. */
  children?: ReactNode;
  /** Popup width in pixels. */
  width?: number;
}

export function Tooltip({ content, align = 'center', children, width = 256 }: TooltipProps) {
  const [position, setPosition] = useState<DOMRect | null>(null);
  const triggerRef = useRef<HTMLSpanElement>(null);

  const show = () => {
    if (triggerRef.current) {
      setPosition(triggerRef.current.getBoundingClientRect());
    }
  };

  const hide = () => {
    setPosition(null);
  };

  // Compute popup position from state (not from ref during render)
  let popup = null;
  if (position) {
    let left: number;
    if (align === 'left') {
      left = position.left;
    } else if (align === 'right') {
      left = position.right - width;
    } else {
      left = position.left + position.width / 2 - width / 2;
    }

    let arrowLeft: number;
    if (align === 'left') {
      arrowLeft = clampArrowOffset(position.width / 2, width);
    } else if (align === 'right') {
      arrowLeft = clampArrowOffset(width - position.width / 2, width);
    } else {
      arrowLeft = width / 2;
    }

    popup = createPortal(
      <div
        className="fixed z-50 whitespace-pre-line p-2 text-xs text-white bg-gray-900 rounded-lg shadow-lg pointer-events-none"
        style={{
          left,
          top: position.top - 8, // 8px gap (mb-2)
          transform: 'translateY(-100%)',
          width,
        }}
      >
        {content}
        <div
          className="absolute border-4 border-transparent border-t-gray-900"
          style={{ top: '100%', left: arrowLeft, transform: 'translateX(-50%)' }}
        />
      </div>,
      document.body,
    );
  }

  return (
    <span
      ref={triggerRef}
      className="relative inline-flex items-center"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocusCapture={show}
      onBlurCapture={hide}
    >
      {children ?? (
        <button
          type="button"
          className="ml-1 inline-flex items-center justify-center w-4 h-4 text-xs text-gray-500 bg-gray-200 rounded-full hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-primary-500"
          aria-label="Help"
        >
          ?
        </button>
      )}
      {popup}
    </span>
  );
}

function clampArrowOffset(offset: number, width: number): number {
  return Math.max(16, Math.min(offset, width - 16));
}
