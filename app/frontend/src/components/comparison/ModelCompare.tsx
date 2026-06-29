/**
 * Side-by-side model comparison component with interactive similarity heatmaps.
 *
 * Displays ground truth bounding boxes on both panels. When a bbox is clicked,
 * similarity heatmaps are shown for both models simultaneously, allowing users
 * to compare how different models "see" the same architectural feature.
 */

import { useMemo, useState } from 'react';
import { useModelComparison, useModels } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Card, CardContent } from '../ui/Card';
import { SimilarityViewer } from './SimilarityViewer';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { renderHeatmapLegend } from '../../utils/renderHeatmap';
import type { BoundingBox, ImageMetricSelection, IoUResult } from '../../types';

interface ModelCompareProps {
  imageId: string;
  layer: number;
  percentile: number;
  method?: string;
  availableModels: string[];
  bboxes: BoundingBox[];
}

export function ModelCompare({
  imageId,
  layer,
  percentile,
  method,
  availableModels,
  bboxes,
}: ModelCompareProps) {
  const [leftModel, setLeftModel] = useState(availableModels[0] || 'dinov2');
  const [rightModel, setRightModel] = useState(availableModels[1] || 'clip');
  const [selectedBboxIndex, setSelectedBboxIndex] = useState<number | null>(null);
  const { data: modelsData, isLoading: modelsLoading } = useModels();

  const effectiveLayer = useMemo(() => {
    const numLayersByModel = modelsData?.num_layers_per_model;
    if (!numLayersByModel) {
      return layer;
    }

    const leftMaxLayer = Math.max((numLayersByModel[leftModel] ?? layer + 1) - 1, 0);
    const rightMaxLayer = Math.max((numLayersByModel[rightModel] ?? layer + 1) - 1, 0);
    return Math.min(layer, leftMaxLayer, rightMaxLayer);
  }, [layer, leftModel, modelsData?.num_layers_per_model, rightModel]);

  const comparisonMethod = useMemo(() => {
    if (!method) {
      return undefined;
    }

    const methodsByModel = modelsData?.methods;
    if (!methodsByModel) {
      return undefined;
    }

    const leftMethods = methodsByModel[leftModel] ?? [];
    const rightMethods = methodsByModel[rightModel] ?? [];
    return leftMethods.includes(method) && rightMethods.includes(method) ? method : undefined;
  }, [leftModel, method, modelsData?.methods, rightModel]);

  const compareMethodCopy = useMemo(() => {
    if (comparisonMethod) {
      return `Comparing with shared method: ${comparisonMethod}`;
    }

    if (method) {
      return `Using each model's default attention method because ${method} is not shared by both selected models.`;
    }

    return "Using each model's default attention method.";
  }, [comparisonMethod, method]);

  const { data: comparison, isLoading, error } = useModelComparison(
    imageId,
    [leftModel, rightModel],
    effectiveLayer,
    percentile,
    comparisonMethod,
    selectedBboxIndex,
    !modelsLoading
  );

  const modelOptions = availableModels.map((m) => ({
    value: m,
    label: m.charAt(0).toUpperCase() + m.slice(1),
  }));

  // Generate legend URL (static, computed once)
  const legendUrl = useMemo(() => renderHeatmapLegend(200, 16), []);

  // Get selected bbox for display
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;
  const metricSelection: ImageMetricSelection = comparison?.selection ?? {
    mode: selectedBbox ? 'bbox' : 'union',
    bbox_index: selectedBboxIndex,
    bbox_label: selectedBbox?.label_name || (selectedBbox ? `Feature ${selectedBbox.label}` : null),
  };
  const leftResult = comparison?.results.find((result) => result.model === leftModel);
  const rightResult = comparison?.results.find((result) => result.model === rightModel);
  const leftUnavailableReason = comparison?.unavailable_models[leftModel];
  const rightUnavailableReason = comparison?.unavailable_models[rightModel];

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <Select
          value={leftModel}
          onChange={setLeftModel}
          options={modelOptions}
          label="Left Model"
          className="flex-1"
        />
        <Select
          value={rightModel}
          onChange={setRightModel}
          options={modelOptions}
          label="Right Model"
          className="flex-1"
        />
      </div>

      <div
        className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-700"
        data-testid="comparison-method-context"
      >
        {compareMethodCopy}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Left model */}
        <Card>
          <ErrorBoundary resetKeys={[imageId, leftModel, effectiveLayer]}>
            <SimilarityViewer
              imageId={imageId}
              model={leftModel}
              layer={effectiveLayer}
              bboxes={bboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={setSelectedBboxIndex}
            />
          </ErrorBoundary>
          <ComparisonMetricsPanel
            panelId="left"
            selection={metricSelection}
            result={leftResult}
            unavailableReason={leftUnavailableReason}
            isLoading={isLoading}
          />
        </Card>

        {/* Right model */}
        <Card>
          <ErrorBoundary resetKeys={[imageId, rightModel, effectiveLayer]}>
            <SimilarityViewer
              imageId={imageId}
              model={rightModel}
              layer={effectiveLayer}
              bboxes={bboxes}
              selectedBboxIndex={selectedBboxIndex}
              onBboxSelect={setSelectedBboxIndex}
            />
          </ErrorBoundary>
          <ComparisonMetricsPanel
            panelId="right"
            selection={metricSelection}
            result={rightResult}
            unavailableReason={rightUnavailableReason}
            isLoading={isLoading}
          />
        </Card>
      </div>

      {/* Selection info and controls */}
      {selectedBbox && (
        <div className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-2">
          <span className="text-sm text-gray-700">
            <span className="font-medium">Selected:</span>{' '}
            {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
          </span>
          <button
            onClick={() => setSelectedBboxIndex(null)}
            className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded transition-colors"
          >
            Clear Selection
          </button>
        </div>
      )}

      {/* Colormap legend (shown when bbox is selected) */}
      {selectedBbox && (
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <span>Low similarity</span>
          <img src={legendUrl} alt="Similarity scale" className="h-4 rounded" />
          <span>High similarity</span>
        </div>
      )}

      {/* Hint for users when no bbox is selected */}
      {bboxes.length > 0 && !selectedBbox && (
        <p className="text-center text-sm text-gray-500">
          Click on a bounding box to compare similarity heatmaps between models
        </p>
      )}

      {(modelsLoading || isLoading) && (
        <div className="text-center text-gray-500">Loading comparison...</div>
      )}
      {error && (
        <div className="text-center text-red-500">Failed to load comparison</div>
      )}
    </div>
  );
}

interface ComparisonMetricsPanelProps {
  panelId: 'left' | 'right';
  selection: ImageMetricSelection;
  result?: IoUResult;
  unavailableReason?: string;
  isLoading: boolean;
}

function ComparisonMetricsPanel({
  panelId,
  selection,
  result,
  unavailableReason,
  isLoading,
}: ComparisonMetricsPanelProps) {
  const scopeCopy = selection.mode === 'bbox' ? 'Feature-level metrics' : 'Whole-image metrics';

  return (
    <CardContent data-testid={`comparison-metrics-${panelId}`}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="text-xs font-medium uppercase tracking-[0.12em] text-slate-500"
            data-testid={`comparison-metric-scope-${panelId}`}
          >
            {scopeCopy}
          </span>
          {selection.bbox_label && (
            <span className="rounded-full bg-green-50 px-2 py-1 text-xs font-medium text-green-700">
              {selection.bbox_label}
            </span>
          )}
        </div>

        {isLoading && (
          <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
            Loading metrics for this scope...
          </div>
        )}

        {!isLoading && unavailableReason && (
          <div
            className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
            data-testid={`comparison-metrics-${panelId}-unavailable`}
          >
            {unavailableReason}
          </div>
        )}

        {!isLoading && !unavailableReason && result && (
          <div className="space-y-1 text-sm">
            <div>
              <span className="font-medium">IoU:</span>{' '}
              {result.iou.toFixed(3)}
            </div>
            {result.method && (
              <div>
                <span className="font-medium">Method:</span>{' '}
                {result.method}
              </div>
            )}
            <div>
              <span className="font-medium">MSE:</span>{' '}
              {result.mse.toFixed(4)}
            </div>
            {Number.isFinite(result.kl) && (
              <div>
                <span className="font-medium">KL:</span>{' '}
                {result.kl.toFixed(4)}
              </div>
            )}
            {Number.isFinite(result.emd) && (
              <div>
                <span className="font-medium">EMD:</span>{' '}
                {result.emd.toFixed(4)}
              </div>
            )}
          </div>
        )}

        {!isLoading && !unavailableReason && !result && (
          <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
            Metrics are not available for this selection.
          </div>
        )}
      </div>
    </CardContent>
  );
}
