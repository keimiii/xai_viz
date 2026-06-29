/**
 * Home page with image grid browser.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { Card } from '../components/ui/Card';
import { Select } from '../components/ui/Select';
import type { ImageListItem } from '../types';

export function HomePage() {
  const navigate = useNavigate();
  const [styleFilter, setStyleFilter] = useState<string>('');
  const [filenameFilter, setFilenameFilter] = useState<string>('');

  // Fetch styles
  const { data: styles, error: stylesError } = useQuery({
    queryKey: ['styles'],
    queryFn: () => imagesAPI.getStyles(),
  });

  // Fetch images
  const { data: images, isLoading, error } = useQuery({
    queryKey: ['images', styleFilter],
    queryFn: () => imagesAPI.list({ style: styleFilter || undefined }),
  });
  const hasLoadError = !!stylesError || !!error;
  const normalizedFilenameFilter = filenameFilter.trim().toLowerCase();
  const filteredImages = images?.filter((image) => (
    !normalizedFilenameFilter || image.image_id.toLowerCase().includes(normalizedFilenameFilter)
  )) || [];
  const visibleBboxCount = filteredImages.reduce((sum, image) => sum + image.num_bboxes, 0);

  const styleOptions = [
    { value: '', label: 'All Styles' },
    ...(styles?.map((s) => ({ value: s, label: s })) || []),
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">WikiChurches Attention Analysis</h1>
          <p className="text-gray-600 mt-1">
            {hasLoadError
              ? 'Dataset summary unavailable while the backend is offline.'
              : `${filteredImages.length} annotated images with ${visibleBboxCount} bounding boxes`}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:w-auto">
          <div className="flex flex-col gap-1">
            <label htmlFor="gallery-filename-filter" className="text-sm font-medium text-gray-700">
              Filename
            </label>
            <input
              id="gallery-filename-filter"
              type="text"
              value={filenameFilter}
              onChange={(event) => setFilenameFilter(event.target.value)}
              placeholder="Search filename..."
              className="w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 sm:w-64"
            />
          </div>

          <Select
            value={styleFilter}
            onChange={setStyleFilter}
            options={styleOptions}
            label="Style"
            className="w-full sm:w-48"
          />
        </div>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="aspect-square bg-gray-200 rounded-lg" />
              <div className="h-4 bg-gray-200 rounded mt-2 w-3/4" />
            </div>
          ))}
        </div>
      )}

      {/* Error state */}
      {hasLoadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
          Failed to load gallery data. Make sure the backend is running, then refresh this page to retry.
        </div>
      )}

      {/* Image grid */}
      {images && !hasLoadError && filteredImages.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredImages.map((image) => (
            <ImageCard
              key={image.image_id}
              image={image}
              onClick={() => navigate(`/image/${encodeURIComponent(image.image_id)}`)}
            />
          ))}
        </div>
      )}

      {images && !hasLoadError && filteredImages.length === 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-600">
          No images match the current filename and style filters.
        </div>
      )}
    </div>
  );
}

interface ImageCardProps {
  image: ImageListItem;
  onClick: () => void;
}

function ImageCard({ image, onClick }: ImageCardProps) {
  const [imgError, setImgError] = useState(false);

  return (
    <Card hoverable onClick={onClick}>
      <div className="aspect-square relative bg-gray-100">
        {!imgError ? (
          <img
            src={imagesAPI.getThumbnailUrl(image.image_id)}
            alt={image.image_id}
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No image
          </div>
        )}

        {/* Bbox count badge */}
        <div className="absolute top-2 right-2 px-2 py-0.5 bg-black/50 text-white text-xs rounded">
          {image.num_bboxes} boxes
        </div>
      </div>

      <div className="p-2">
        <div className="text-xs text-gray-500 truncate" title={image.image_id}>
          {image.image_id}
        </div>
        <div className="flex gap-1 mt-1 flex-wrap">
          {image.style_names.map((style) => (
            <span
              key={style}
              className="px-1.5 py-0.5 bg-primary-100 text-primary-700 text-xs rounded"
            >
              {style}
            </span>
          ))}
        </div>
      </div>
    </Card>
  );
}
