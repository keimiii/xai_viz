/**
 * Main Image Detail viewer that switches between attention and bbox-similarity interpretations.
 * Supports dynamic percentile thresholding via client-side Canvas rendering.
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { attentionAPI, imagesAPI } from '../../api/client';
import { InteractiveBboxOverlay } from './InteractiveBboxOverlay';
import { renderHeatmap, renderHeatmapLegend, computeSimilarityStats, renderAttentionHeatmap } from '../../utils/renderHeatmap';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import type { BoundingBox, ImageDetailMode } from '../../types';

interface AttentionViewerProps {
  imageId: string;
  model: string;
  layer: number;
  method: string;
  head: number | null;
  mode: ImageDetailMode;
  percentile: number;
  showBboxes: boolean;
  bboxSelectionDrivesOverlay?: boolean;
  bboxes?: BoundingBox[];
  selectedBboxIndex: number | null;
  onBboxSelect: (index: number | null) => void;
  className?: string;
}

export function AttentionViewer({
  imageId,
  model,
  layer,
  method,
  head,
  mode,
  percentile,
  showBboxes,
  bboxSelectionDrivesOverlay = false,
  bboxes = [],
  selectedBboxIndex,
  onBboxSelect,
  className = '',
}: AttentionViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [imageError, setImageError] = useState(false);
  const previousImageIdRef = useRef(imageId);
  const isHeadAttentionMode = mode === 'head_attention';
  const isFeatureSimilarityMode = mode === 'feature_similarity';
  const canShowFocusedOverlay = isFeatureSimilarityMode || (isHeadAttentionMode && bboxSelectionDrivesOverlay);

  // Reset error state when model/layer/method changes
  // Uses the "adjust state during render" pattern recommended by React
  // instead of setState-in-effect which causes cascading renders.
  const errorResetKey = `${imageId}|${model}|${layer}|${method}|${head ?? 'all'}|${mode}`;
  const [prevResetKey, setPrevResetKey] = useState(errorResetKey);
  if (errorResetKey !== prevResetKey) {
    setPrevResetKey(errorResetKey);
    setImageError(false);
  }
  const [previousMode, setPreviousMode] = useState(mode);
  if (mode !== previousMode) {
    setPreviousMode(mode);
    setShowOverlay(true);
  }

  // Get heatmap settings from store
  const heatmapOpacity = useHeatmapOpacity();
  const heatmapStyle = useHeatmapStyle();

  const originalUrl = imagesAPI.getImageUrl(imageId, 224);

  // Get selected bbox
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;

  // Fetch raw attention data for dynamic rendering
  const { data: rawAttentionData, isLoading: attentionLoading, error: attentionError } = useQuery({
    queryKey: ['rawAttention', imageId, model, layer, method, head],
    queryFn: () => attentionAPI.getRawAttention(imageId, model, layer, method, head),
    enabled: isHeadAttentionMode,
    staleTime: 60000, // Cache for 1 minute
    placeholderData: keepPreviousData,
  });

  // Render thresholded attention heatmap
  const attentionHeatmapUrl = useMemo(() => {
    if (!rawAttentionData) return null;
    try {
      return renderAttentionHeatmap({
        attention: rawAttentionData.attention,
        shape: rawAttentionData.shape,
        percentile,
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [rawAttentionData, percentile, heatmapOpacity, heatmapStyle]);

  // Fetch similarity when a bbox is selected
  const { data: similarityData, isLoading: similarityLoading } = useQuery({
    queryKey: ['similarity', imageId, model, layer, selectedBbox],
    queryFn: () => {
      if (!selectedBbox) return null;
      return attentionAPI.getSimilarity(
        imageId,
        {
          left: selectedBbox.left,
          top: selectedBbox.top,
          width: selectedBbox.width,
          height: selectedBbox.height,
          label: selectedBbox.label_name || undefined,
        },
        model,
        layer
      );
    },
    enabled: canShowFocusedOverlay && !!selectedBbox,
    staleTime: 60000, // Cache for 1 minute
    placeholderData: keepPreviousData,
  });

  // Render heatmap when similarity data is available
  const similarityHeatmapUrl = useMemo(() => {
    if (!similarityData) return null;
    try {
      return renderHeatmap({
        similarity: similarityData.similarity,
        patchGrid: similarityData.patch_grid as [number, number],
        opacity: heatmapOpacity,
        style: heatmapStyle,
      });
    } catch {
      return null;
    }
  }, [similarityData, heatmapOpacity, heatmapStyle]);

  // Compute stats for display
  const stats = useMemo(() => {
    if (!similarityData) return null;
    return computeSimilarityStats(similarityData.similarity);
  }, [similarityData]);

  // Generate legend URL (static, computed once)
  const legendUrl = useMemo(() => renderHeatmapLegend(120, 12), []);

  // Reset bbox selection only after the user actually navigates to a different image.
  // This preserves URL-restored bbox state on the initial mount.
  useEffect(() => {
    if (previousImageIdRef.current === imageId) {
      return;
    }
    previousImageIdRef.current = imageId;
    onBboxSelect(null);
  }, [imageId, onBboxSelect]);

  const handleBboxClick = (_bbox: BoundingBox, index: number) => {
    // Toggle selection if clicking the same bbox
    if (selectedBboxIndex === index) {
      onBboxSelect(null);
    } else {
      onBboxSelect(index);
    }
  };

  const handleImageClick = () => {
    // Deselect when clicking outside bboxes
    if (selectedBboxIndex !== null) {
      onBboxSelect(null);
    }
  };

  if (imageError || (isHeadAttentionMode && attentionError)) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`}>
        <div className="text-center text-gray-500">
          <p className="text-sm">Heatmap not available</p>
          <p className="text-xs mt-1">Run pre-computation first</p>
        </div>
      </div>
    );
  }

  const showSimilarityHeatmap = canShowFocusedOverlay && showOverlay && selectedBbox && similarityHeatmapUrl && !similarityLoading;
  const showAttentionHeatmap = isHeadAttentionMode && showOverlay && attentionHeatmapUrl && !showSimilarityHeatmap;
  const showAttentionLoading = isHeadAttentionMode && attentionLoading;
  const showSimilarityLoading = canShowFocusedOverlay && similarityLoading && selectedBbox;
  const showingFocusedOverlay = canShowFocusedOverlay && !!selectedBbox;
  const overlayButtonLabel = showingFocusedOverlay
    ? (showOverlay ? 'Hide Focused Overlay' : 'Show Focused Overlay')
    : (showOverlay ? 'Hide Attention Overlay' : 'Show Attention Overlay');
  const canToggleOverlay = showingFocusedOverlay
    ? Boolean(selectedBbox && similarityHeatmapUrl)
    : Boolean(attentionHeatmapUrl);

  return (
    <div className={`relative group ${className}`}>
      {/* Base image (always the original) */}
      <img
        src={originalUrl}
        alt={`${imageId}`}
        className="w-full h-auto rounded-lg"
        onError={() => setImageError(true)}
        onClick={handleImageClick}
      />

      {/* Dynamic attention heatmap overlay */}
      {showAttentionHeatmap && (
        <img
          src={attentionHeatmapUrl}
          alt="Attention heatmap"
          data-testid="attention-overlay-image"
          className="absolute inset-0 w-full h-full rounded-lg pointer-events-none"
          style={{ mixBlendMode: 'normal' }}
        />
      )}

      {/* Similarity heatmap overlay (when bbox selected) */}
      {showSimilarityHeatmap && (
        <img
          src={similarityHeatmapUrl}
          alt="Similarity heatmap"
          data-testid="similarity-overlay-image"
          className="absolute inset-0 w-full h-full rounded-lg pointer-events-none"
          style={{ mixBlendMode: 'normal' }}
        />
      )}

      {/* Loading spinner for attention data */}
      {showAttentionLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent" />
        </div>
      )}

      {/* Loading spinner for similarity computation */}
      {showSimilarityLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent" />
        </div>
      )}

      {/* Interactive bbox overlay */}
      {showBboxes && bboxes.length > 0 && (
        <InteractiveBboxOverlay
          bboxes={bboxes}
          selectedIndex={selectedBboxIndex}
          onBboxClick={handleBboxClick}
        />
      )}

      {isFeatureSimilarityMode && !selectedBbox && (
        <div
          className="absolute left-2 top-2 max-w-[16rem] rounded bg-black/60 px-3 py-2 text-xs text-white"
          data-testid="similarity-selection-hint"
        >
          Select a bounding box to activate feature similarity.
        </div>
      )}

      {/* Toggle overlay button */}
      {canToggleOverlay && (
        <button
          onClick={() => setShowOverlay(!showOverlay)}
          className="absolute top-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
        >
          {overlayButtonLabel}
        </button>
      )}

      {/* Clear selection button (when bbox is selected) */}
      {selectedBbox && (
        <button
          onClick={() => onBboxSelect(null)}
          className="absolute top-2 right-24 px-2 py-1 text-xs bg-green-600/80 text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
        >
          Clear Selection
        </button>
      )}

      {/* Info badge */}
      <div className="absolute bottom-2 left-2 px-2 py-1 text-xs bg-black/50 text-white rounded" data-testid="viewer-info-badge">
        {isHeadAttentionMode ? (
          <>
            Head Attention · {model} / {method} / Layer {layer} / Top {100 - percentile}%
            {head !== null && (
              <span className="ml-2 text-sky-300"> · Head {head}</span>
            )}
          </>
        ) : (
          <>Feature Similarity · {model} / Layer {layer}</>
        )}
        {selectedBbox && (
          <span className="ml-2 text-green-300">
            {' '}
            · {selectedBbox.label_name || `Feature ${selectedBbox.label}`}
          </span>
        )}
      </div>

      {/* Similarity stats badge */}
      {canShowFocusedOverlay && stats && selectedBbox && (
        <div
          className="absolute bottom-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded"
          data-testid="similarity-stats"
        >
          Sim: {stats.min.toFixed(2)} - {stats.max.toFixed(2)}
        </div>
      )}

      {/* Colormap legend */}
      {canShowFocusedOverlay && stats && selectedBbox && (
        <div
          className="absolute bottom-8 right-2 flex items-center gap-1 px-2 py-1 bg-black/50 rounded text-xs text-white"
          data-testid="similarity-legend"
        >
          <span>0</span>
          <img src={legendUrl} alt="Similarity scale" className="h-3 rounded-sm" />
          <span>1</span>
        </div>
      )}
    </div>
  );
}
