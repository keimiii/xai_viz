/**
 * SimilarityViewer - displays an image with interactive bounding boxes
 * that show similarity heatmaps when clicked.
 *
 * Used in ModelCompare for side-by-side model comparison with synchronized
 * bbox selection across both panels.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { attentionAPI, imagesAPI } from '../../api/client';
import { InteractiveBboxOverlay } from '../attention/InteractiveBboxOverlay';
import { renderHeatmap, computeSimilarityStats } from '../../utils/renderHeatmap';
import { useHeatmapOpacity, useHeatmapStyle } from '../../store/viewStore';
import type { BoundingBox } from '../../types';

interface SimilarityViewerProps {
  imageId: string;
  model: string;
  layer: number;
  bboxes: BoundingBox[];
  selectedBboxIndex: number | null;
  onBboxSelect: (index: number | null) => void;
}

export function SimilarityViewer({
  imageId,
  model,
  layer,
  bboxes,
  selectedBboxIndex,
  onBboxSelect,
}: SimilarityViewerProps) {
  // Get heatmap settings from store
  const heatmapOpacity = useHeatmapOpacity();
  const heatmapStyle = useHeatmapStyle();

  const originalUrl = imagesAPI.getImageUrl(imageId, 224);

  // Get selected bbox
  const selectedBbox = selectedBboxIndex !== null ? bboxes[selectedBboxIndex] : null;

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
    enabled: !!selectedBbox,
    staleTime: 60000, // Cache for 1 minute
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

  const showSimilarityHeatmap = selectedBbox && similarityHeatmapUrl && !similarityLoading;

  return (
    <div className="relative">
      {/* Base image */}
      <img
        src={originalUrl}
        alt={`${imageId}`}
        className="w-full h-auto"
        onClick={handleImageClick}
      />

      {/* Similarity heatmap overlay (when bbox selected) */}
      {showSimilarityHeatmap && (
        <img
          src={similarityHeatmapUrl}
          alt="Similarity heatmap"
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ mixBlendMode: 'normal' }}
        />
      )}

      {/* Loading spinner for similarity computation */}
      {similarityLoading && selectedBbox && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent" />
        </div>
      )}

      {/* Interactive bbox overlay */}
      {bboxes.length > 0 && (
        <InteractiveBboxOverlay
          bboxes={bboxes}
          selectedIndex={selectedBboxIndex}
          onBboxClick={handleBboxClick}
        />
      )}

      {/* Model label badge */}
      <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
        {model}
      </div>

      {/* Similarity stats badge */}
      {stats && selectedBbox && (
        <div className="absolute bottom-2 right-2 px-2 py-1 text-xs bg-black/50 text-white rounded">
          Sim: {stats.min.toFixed(2)} - {stats.max.toFixed(2)}
        </div>
      )}
    </div>
  );
}
