/**
 * Reusable error boundary component with two display modes.
 *
 * - "page" level: centered card with retry + gallery link (for route-level boundaries)
 * - "widget" level: compact inline banner (for individual components)
 *
 * Supports resetKeys to auto-clear errors when props change (e.g., navigation, model switch).
 */

import { Component, type ReactNode } from 'react';
import { useLocation } from 'react-router-dom';

interface ErrorBoundaryProps {
  children: ReactNode;
  level?: 'page' | 'widget';
  resetKeys?: ReadonlyArray<unknown>;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (!this.state.hasError) return;

    const prev = prevProps.resetKeys ?? [];
    const curr = this.props.resetKeys ?? [];

    const changed =
      prev.length !== curr.length ||
      prev.some((v, i) => !Object.is(v, curr[i]));

    if (changed) {
      this.setState({ hasError: false, error: null });
    }
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    const level = this.props.level ?? 'widget';

    if (level === 'page') {
      return (
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md text-center space-y-4">
            <h2 className="text-lg font-semibold text-red-800">
              Something went wrong
            </h2>
            <p className="text-sm text-red-700">
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <div className="flex gap-3 justify-center">
              <button
                onClick={this.handleReset}
                className="px-4 py-2 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition-colors"
              >
                Try Again
              </button>
              <a
                href="/"
                className="px-4 py-2 bg-gray-100 text-gray-700 text-sm rounded-lg hover:bg-gray-200 transition-colors"
              >
                Back to Gallery
              </a>
            </div>
          </div>
        </div>
      );
    }

    // Widget-level fallback
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        <p className="text-sm font-medium">Failed to render this section.</p>
        <button
          onClick={this.handleReset}
          className="text-sm text-red-600 underline hover:text-red-800 mt-1"
        >
          Try again
        </button>
      </div>
    );
  }
}

/**
 * Route-level wrapper that auto-resets on navigation changes.
 */
export function RouteErrorBoundary({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <ErrorBoundary level="page" resetKeys={[location.pathname, location.search]}>
      {children}
    </ErrorBoundary>
  );
}
