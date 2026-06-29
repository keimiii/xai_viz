/**
 * Control panel for selecting model, layer, attention method, and display options.
 */

import { useEffect } from 'react';
import { useViewStore } from '../../store/viewStore';
import { useModels } from '../../hooks/useAttention';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { Toggle } from '../ui/Toggle';
import { GLOSSARY } from '../../constants/glossary';
import { getAttentionMethodLabel } from '../../constants/attentionMethods';
import { PERCENTILE_OPTIONS } from '../../constants/percentiles';
import type { HeatmapStyle, ImageDetailMode } from '../../types';

interface ControlPanelProps {
  className?: string;
  mode: ImageDetailMode;
  title?: string;
  showLayerControl?: boolean;
  showPercentileControl?: boolean;
  showBoundingBoxesToggle?: boolean;
  showOverlayAppearanceControls?: boolean;
}

function formatModelOptionLabel(model: string): string {
  return model.charAt(0).toUpperCase() + model.slice(1);
}

export function ControlPanel({
  className = '',
  mode,
  title = 'View Settings',
  showLayerControl = true,
  showPercentileControl = true,
  showBoundingBoxesToggle = true,
  showOverlayAppearanceControls = true,
}: ControlPanelProps) {
  const {
    model,
    layer,
    method,
    head,
    percentile,
    showBboxes,
    heatmapOpacity,
    heatmapStyle,
    availableMethods,
    numHeadsPerModel,
    perHeadMethods,
    perHeadAvailableModels,
    setModel,
    setLayer,
    setMethod,
    setHead,
    setPercentile,
    setShowBboxes,
    setHeatmapOpacity,
    setHeatmapStyle,
    setMethodsConfig,
    setNumLayersPerModel,
    setPerHeadConfig,
  } = useViewStore();

  const { data: modelsData, isLoading } = useModels();

  // Update store with methods config when API data loads
  useEffect(() => {
    if (modelsData?.methods && modelsData?.default_methods) {
      setMethodsConfig(modelsData.methods, modelsData.default_methods);
    }
  }, [modelsData, setMethodsConfig]);

  // Update store with num_layers_per_model when API data loads
  useEffect(() => {
    if (modelsData?.num_layers_per_model) {
      setNumLayersPerModel(modelsData.num_layers_per_model);
    }
  }, [modelsData, setNumLayersPerModel]);

  useEffect(() => {
    if (modelsData?.num_heads_per_model && modelsData?.per_head_methods) {
      setPerHeadConfig(
        modelsData.num_heads_per_model,
        modelsData.per_head_methods,
        modelsData.per_head_available_models ?? [],
      );
    }
  }, [modelsData, setPerHeadConfig]);

  // Get max layer for current model (0-indexed, so subtract 1)
  const maxLayer = modelsData?.num_layers_per_model?.[model]
    ? modelsData.num_layers_per_model[model] - 1
    : (modelsData?.num_layers || 12) - 1;

  // Note: Layer clamping now happens synchronously in viewStore.setModel()
  // This prevents race conditions where API calls fire before the useEffect runs

  if (isLoading) {
    return (
      <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded" />
          <div className="h-8 bg-gray-200 rounded" />
          <div className="h-8 bg-gray-200 rounded" />
        </div>
      </div>
    );
  }

  const modelOptions =
    modelsData?.models.map((m) => ({
      value: m,
      label: formatModelOptionLabel(m),
    })) || [];

  // Get available methods for current model
  const currentModelMethods = availableMethods[model] || modelsData?.methods?.[model] || ['cls'];
  const methodOptions = currentModelMethods.map((m) => ({
    value: m,
    label: getAttentionMethodLabel(m),
  }));
  const supportsHeadSelection =
    (numHeadsPerModel[model] || 0) > 0
    && perHeadMethods.includes(method)
    && perHeadAvailableModels.includes(model);
  const headOptions = [
    { value: '-1', label: 'All (Fused)' },
    ...Array.from({ length: numHeadsPerModel[model] || 0 }, (_, idx) => ({
      value: String(idx),
      label: `Head ${idx}`,
    })),
  ];

  const heatmapStyleOptions = [
    { value: 'smooth', label: 'Smooth Gradient' },
    { value: 'squares', label: 'Squares' },
    { value: 'circles', label: 'Circles' },
  ];

  return (
    <div className={`p-4 bg-white rounded-lg shadow space-y-4 ${className}`}>
      <h3 className="font-semibold text-gray-900">{title}</h3>

      <Select
        value={model}
        onChange={setModel}
        options={modelOptions}
        label="Model"
        tooltip={GLOSSARY['Model']}
      />

      {/* Only show method selector if model has multiple methods */}
      {currentModelMethods.length > 1 && (
        <Select
          value={method}
          onChange={setMethod}
          options={methodOptions}
          label="Attention Method"
          tooltip={GLOSSARY['Attention Method']}
        />
      )}

      {supportsHeadSelection && mode === 'head_attention' && (
        <Select
          value={head === null ? '-1' : String(head)}
          onChange={(value) => setHead(value === '-1' ? null : Number(value))}
          options={headOptions}
          label="Attention Head"
          tooltip={GLOSSARY['Attention Head']}
        />
      )}

      {showLayerControl && (
        <Slider
          value={layer}
          onChange={setLayer}
          min={0}
          max={maxLayer}
          label={`Layer ${layer}`}
          tooltip={GLOSSARY['Layer']}
        />
      )}

      {showPercentileControl && (
        <Select
          value={percentile}
          onChange={(v) => setPercentile(Number(v))}
          options={PERCENTILE_OPTIONS}
          label="Attention Threshold"
          tooltip={GLOSSARY['Attention Threshold']}
        />
      )}

      {showBoundingBoxesToggle && (
        <Toggle
          checked={showBboxes}
          onChange={setShowBboxes}
          label="Show Bounding Boxes"
        />
      )}

      {showOverlayAppearanceControls && (
        <div className="border-t pt-4 mt-2 space-y-3">
          <h4 className="text-sm font-medium text-gray-700">Overlay Appearance</h4>
          <p className="text-xs text-gray-500">
            These controls style the active overlay for the current interpretation mode.
          </p>

          <Select
            value={heatmapStyle}
            onChange={(v) => setHeatmapStyle(v as HeatmapStyle)}
            options={heatmapStyleOptions}
            label="Heatmap Style"
            tooltip={GLOSSARY['Heatmap Style']}
          />

          <Slider
            value={heatmapOpacity}
            onChange={setHeatmapOpacity}
            min={0.2}
            max={0.9}
            step={0.1}
            label={`Opacity ${Math.round(heatmapOpacity * 100)}%`}
            tooltip={GLOSSARY['Heatmap Opacity']}
            showValue={false}
          />
        </div>
      )}
    </div>
  );
}
