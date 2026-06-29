/**
 * Global view settings store using Zustand.
 */

import { create } from 'zustand';
import { DEFAULT_IMAGE_DETAIL_MODE } from '../constants/imageDetailModes';
import type { ViewSettings, HeatmapStyle, ImageDetailMode } from '../types';

interface ViewStore extends ViewSettings {
  // Selection state for bbox similarity
  selectedBboxIndex: number | null;

  // Available methods per model (populated from API)
  availableMethods: Record<string, string[]>;
  defaultMethods: Record<string, string>;

  // Number of layers per model (populated from API)
  numLayersPerModel: Record<string, number>;
  numHeadsPerModel: Record<string, number>;
  perHeadMethods: string[];
  perHeadAvailableModels: string[];

  // Actions
  setModel: (model: string) => void;
  setLayer: (layer: number) => void;
  setMethod: (method: string) => void;
  setHead: (head: number | null) => void;
  setImageDetailMode: (mode: ImageDetailMode) => void;
  setModelWithPreferredMethod: (model: string, preferredMethod: string) => void;
  setPercentile: (percentile: number) => void;
  setShowBboxes: (show: boolean) => void;
  setHeatmapOpacity: (opacity: number) => void;
  setHeatmapStyle: (style: HeatmapStyle) => void;
  setSelectedBboxIndex: (index: number | null) => void;
  setMethodsConfig: (methods: Record<string, string[]>, defaults: Record<string, string>) => void;
  setNumLayersPerModel: (numLayers: Record<string, number>) => void;
  setPerHeadConfig: (numHeads: Record<string, number>, methods: string[], availableModels?: string[]) => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: ViewSettings = {
  model: 'dinov2',
  layer: 0,  // Safe default for all models (including ResNet-50 which has only 4 layers)
  method: 'cls',
  head: null,
  imageDetailMode: DEFAULT_IMAGE_DETAIL_MODE,
  percentile: 90,
  showBboxes: true,
  heatmapOpacity: 0.5,
  heatmapStyle: 'smooth',
};

export const useViewStore = create<ViewStore>((set, get) => ({
  ...DEFAULT_SETTINGS,
  selectedBboxIndex: null,
  availableMethods: {},
  defaultMethods: {},
  numLayersPerModel: {},
  numHeadsPerModel: {},
  perHeadMethods: [],
  perHeadAvailableModels: [],

  setModel: (model) => {
    const {
      defaultMethods,
      numLayersPerModel,
      numHeadsPerModel,
      layer,
      perHeadMethods,
      perHeadAvailableModels,
      head,
    } = get();
    const newMethod = defaultMethods[model] || 'cls';
    const maxLayer = (numLayersPerModel[model] || 12) - 1;
    const clampedLayer = Math.min(layer, maxLayer);
    const supportsHead =
      (numHeadsPerModel[model] || 0) > 0
      && perHeadMethods.includes(newMethod)
      && perHeadAvailableModels.includes(model);
    set({ model, method: newMethod, head: supportsHead ? head : null, layer: clampedLayer, selectedBboxIndex: null });
  },
  setLayer: (layer) => set({ layer }), // Keep selection on layer change to compare
  setMethod: (method) => {
    const { model, numHeadsPerModel, perHeadMethods, perHeadAvailableModels } = get();
    const supportsHead =
      (numHeadsPerModel[model] || 0) > 0
      && perHeadMethods.includes(method)
      && perHeadAvailableModels.includes(model);
    set({ method, head: supportsHead ? get().head : null });
  },
  setHead: (head) => set({ head }),
  setImageDetailMode: (imageDetailMode) => set({ imageDetailMode }),
  setModelWithPreferredMethod: (model, preferredMethod) => {
    const {
      availableMethods,
      defaultMethods,
      numLayersPerModel,
      numHeadsPerModel,
      perHeadMethods,
      perHeadAvailableModels,
      layer,
    } = get();
    const modelMethods = availableMethods[model] || [];
    const nextMethod = modelMethods.includes(preferredMethod)
      ? preferredMethod
      : (defaultMethods[model] || preferredMethod || 'cls');
    const maxLayer = (numLayersPerModel[model] || 12) - 1;
    const clampedLayer = Math.min(layer, maxLayer);
    const supportsHead =
      (numHeadsPerModel[model] || 0) > 0
      && perHeadMethods.includes(nextMethod)
      && perHeadAvailableModels.includes(model);
    set({ model, method: nextMethod, head: supportsHead ? get().head : null, layer: clampedLayer, selectedBboxIndex: null });
  },
  setPercentile: (percentile) => set({ percentile }),
  setShowBboxes: (showBboxes) => set({ showBboxes }),
  setHeatmapOpacity: (heatmapOpacity) => set({ heatmapOpacity }),
  setHeatmapStyle: (heatmapStyle) => set({ heatmapStyle }),
  setSelectedBboxIndex: (selectedBboxIndex) => set({ selectedBboxIndex }),
  setMethodsConfig: (availableMethods, defaultMethods) => {
    const { model, method } = get();
    // Update method if current one isn't available for current model
    const available = availableMethods[model] || [];
    const newMethod = available.includes(method) ? method : (defaultMethods[model] || 'cls');
    set({ availableMethods, defaultMethods, method: newMethod });
  },
  setNumLayersPerModel: (numLayersPerModel) => {
    const { model, layer } = get();
    // Clamp current layer if it exceeds new model's layer count
    const maxLayer = (numLayersPerModel[model] || 12) - 1;
    const clampedLayer = Math.min(layer, maxLayer);
    set({ numLayersPerModel, layer: clampedLayer });
  },
  setPerHeadConfig: (numHeadsPerModel, perHeadMethods, perHeadAvailableModels = []) => {
    const { model, method, head } = get();
    const supportsHead =
      (numHeadsPerModel[model] || 0) > 0
      && perHeadMethods.includes(method)
      && perHeadAvailableModels.includes(model);
    set({
      numHeadsPerModel,
      perHeadMethods,
      perHeadAvailableModels,
      head: supportsHead ? head : null,
    });
  },
  reset: () => set({ ...DEFAULT_SETTINGS, selectedBboxIndex: null }),
}));

// Selector hooks for specific values
export const useModel = () => useViewStore((state) => state.model);
export const useLayer = () => useViewStore((state) => state.layer);
export const useMethod = () => useViewStore((state) => state.method);
export const useHead = () => useViewStore((state) => state.head);
export const useImageDetailMode = () => useViewStore((state) => state.imageDetailMode);
export const usePercentile = () => useViewStore((state) => state.percentile);
export const useShowBboxes = () => useViewStore((state) => state.showBboxes);
export const useHeatmapOpacity = () => useViewStore((state) => state.heatmapOpacity);
export const useHeatmapStyle = () => useViewStore((state) => state.heatmapStyle);
export const useSelectedBboxIndex = () => useViewStore((state) => state.selectedBboxIndex);
export const useAvailableMethods = () => useViewStore((state) => state.availableMethods);
export const useNumLayersPerModel = () => useViewStore((state) => state.numLayersPerModel);
