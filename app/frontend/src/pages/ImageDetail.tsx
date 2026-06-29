/**
 * Image detail page with attention viewer and metrics.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { imagesAPI } from '../api/client';
import { useViewStore } from '../store/viewStore';
import { useModels } from '../hooks/useAttention';
import { useImageHeadRanking } from '../hooks/useMetrics';
import { AttentionViewer } from '../components/attention/AttentionViewer';
import { ControlPanel } from '../components/attention/ControlPanel';
import { LayerSlider } from '../components/attention/LayerSlider';
import { ImageDetailMetricsPanel } from '../components/metrics/ImageDetailMetricsPanel';
import { Q3StudyScopeCallout } from '../components/metrics/Q3ScopeFraming';
import { Card, CardContent } from '../components/ui/Card';
import { ErrorBoundary } from '../components/ui/ErrorBoundary';
import { PageTabs } from '../components/ui/PageTabs';
import { AnnotationsCard } from '../components/image-detail/AnnotationsCard';
import { ImageDetailModeSwitch } from '../components/image-detail/ImageDetailModeSwitch';
import { Q3ImageDetailControls } from '../components/image-detail/Q3ImageDetailControls';
import { parsePageTab } from '../constants/pageTabs';
import { createImageDetailQ3SearchParams, getQ3ViewerModel, parseImageDetailQ3State } from '../constants/q3Routing';
import { Q3_DEFAULTS, getQ3ModelScopeStatus } from '../constants/q3Scope';
import type { AnalysisMetric, CompareVariantId, PageTab } from '../types';

export function ImageDetailPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const decodedId = imageId ? decodeURIComponent(imageId) : '';
  const [isPlaying, setIsPlaying] = useState(false);
  const resolvedFeatureKeyRef = useRef<string | null>(null);

  const {
    model: mainModel,
    layer: mainLayer,
    method: mainMethod,
    head: mainHead,
    percentile: mainPercentile,
    showBboxes: mainShowBboxes,
    selectedBboxIndex: mainSelectedBboxIndex,
    setLayer: setMainLayer,
    setSelectedBboxIndex: setMainSelectedBboxIndex,
  } = useViewStore();

  const currentTab = parsePageTab(searchParams.get('tab'));
  const isQ3Tab = currentTab === 'q3';
  const searchParamsString = searchParams.toString();
  const { data: modelsData } = useModels();

  const rawQ3State = parseImageDetailQ3State(searchParams);
  const q3MaxLayer = modelsData?.num_layers_per_model?.[rawQ3State.model]
    ? modelsData.num_layers_per_model[rawQ3State.model] - 1
    : Q3_DEFAULTS.layer;
  const q3NumHeads = modelsData?.num_heads_per_model?.[rawQ3State.model] ?? 12;
  const q3State = parseImageDetailQ3State(searchParams, {
    maxLayer: q3MaxLayer,
    numHeads: q3NumHeads,
  });
  const q3Method = modelsData?.default_methods?.[q3State.model] ?? Q3_DEFAULTS.method;
  const q3ViewerModel = getQ3ViewerModel(q3State.model, q3State.variant);
  const q3ViewerPercentile = Q3_DEFAULTS.percentile;
  const q3VariantPerHeadAvailable = modelsData
    ? (
      modelsData.q3_per_head_variant_availability?.[q3State.model]?.[q3State.variant]
      ?? (q3State.variant === 'frozen'
        ? (modelsData.per_head_available_models ?? []).includes(q3State.model)
        : undefined)
    )
    : undefined;

  const persistSearchParams = useCallback((nextParams: URLSearchParams, replace = false) => {
    setSearchParams(nextParams, { replace });
  }, [setSearchParams]);

  const updateQ3State = useCallback((patch: Partial<ReturnType<typeof parseImageDetailQ3State>>, replace = false) => {
    const nextState = {
      ...q3State,
      ...patch,
    };
    persistSearchParams(createImageDetailQ3SearchParams(nextState, searchParams), replace);
  }, [persistSearchParams, q3State, searchParams]);

  const handleMainBboxSelect = useCallback((index: number | null) => {
    setMainSelectedBboxIndex(index);
  }, [setMainSelectedBboxIndex]);

  const handleQ3BboxSelect = useCallback((index: number | null) => {
    updateQ3State({ bboxIndex: index });
  }, [updateQ3State]);

  const handleTabChange = useCallback((nextTab: PageTab) => {
    if (nextTab === 'q3') {
      persistSearchParams(createImageDetailQ3SearchParams(q3State, searchParams));
      return;
    }

    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('tab', 'main');
    persistSearchParams(nextParams);
  }, [persistSearchParams, q3State, searchParams]);

  const handleApplyQ3Defaults = useCallback(() => {
    setIsPlaying(false);
    updateQ3State(
      {
        model: Q3_DEFAULTS.model,
        variant: Q3_DEFAULTS.variant,
        layer: Q3_DEFAULTS.layer,
        head: Q3_DEFAULTS.head,
        metric: Q3_DEFAULTS.metric,
        mode: Q3_DEFAULTS.mode,
        showBboxes: Q3_DEFAULTS.showBboxes,
        bboxIndex: null,
        featureLabel: null,
        featureName: null,
      },
    );
  }, [updateQ3State]);

  const handleQ3ModelChange = useCallback((nextModel: string) => {
    const nextMaxLayer = modelsData?.num_layers_per_model?.[nextModel]
      ? modelsData.num_layers_per_model[nextModel] - 1
      : Q3_DEFAULTS.layer;
    const nextNumHeads = modelsData?.num_heads_per_model?.[nextModel] ?? 12;
    updateQ3State({
      model: nextModel,
      layer: Math.min(q3State.layer, nextMaxLayer),
      head: q3State.head === null ? null : Math.min(q3State.head, Math.max(0, nextNumHeads - 1)),
      bboxIndex: null,
    });
  }, [modelsData, q3State.head, q3State.layer, updateQ3State]);

  const handleQ3VariantChange = useCallback((nextVariant: CompareVariantId) => {
    updateQ3State({
      variant: nextVariant,
      bboxIndex: null,
    });
  }, [updateQ3State]);

  const handleQ3ModeChange = useCallback((nextMode: 'head_attention' | 'feature_similarity') => {
    updateQ3State({ mode: nextMode });
  }, [updateQ3State]);

  const handleQ3HeadChange = useCallback((nextHead: number | null) => {
    updateQ3State({ head: nextHead });
  }, [updateQ3State]);

  const handleQ3MetricChange = useCallback((nextMetric: AnalysisMetric) => {
    updateQ3State({ metric: nextMetric });
  }, [updateQ3State]);

  const handleQ3LayerChange = useCallback((nextLayer: number) => {
    setIsPlaying(false);
    updateQ3State({ layer: nextLayer });
  }, [updateQ3State]);

  const handleQ3ShowBboxesChange = useCallback((show: boolean) => {
    updateQ3State({ showBboxes: show });
  }, [updateQ3State]);

  useEffect(() => {
    if (!isQ3Tab) {
      return;
    }
    const normalizedParams = createImageDetailQ3SearchParams(q3State, searchParams);
    const normalizedString = normalizedParams.toString();
    if (normalizedString !== searchParamsString) {
      persistSearchParams(normalizedParams, true);
    }
  }, [isQ3Tab, persistSearchParams, q3State, searchParams, searchParamsString]);

  // Fetch image details
  const { data: imageDetail, isLoading: detailLoading, error } = useQuery({
    queryKey: ['imageDetail', decodedId],
    queryFn: () => imagesAPI.getDetail(decodedId),
    enabled: !!decodedId,
  });
  const q3HeadRankingQuery = useImageHeadRanking(
    decodedId,
    q3State.model,
    q3State.layer,
    q3ViewerPercentile,
    q3State.metric,
    q3State.variant,
    {
      bboxIndex: q3State.bboxIndex,
      enabled: isQ3Tab && !!decodedId,
    },
  );
  const q3HeadRankingError = q3HeadRankingQuery.error instanceof Error ? q3HeadRankingQuery.error.message : null;
  const q3RankedHeads = q3HeadRankingQuery.data?.heads ?? [];
  const q3SelectedHeadAvailable = q3State.head === null || (
    q3HeadRankingQuery.data?.supported === true
    && q3RankedHeads.some((entry) => entry.head === q3State.head)
  );

  const q3BboxParam = searchParams.get('bbox_index');
  useEffect(() => {
    if (!isQ3Tab || !imageDetail || q3State.featureLabel === null || q3BboxParam !== null) {
      return;
    }

    const nextResolutionKey = `${decodedId}|${q3State.featureLabel}`;
    if (resolvedFeatureKeyRef.current === nextResolutionKey) {
      return;
    }

    resolvedFeatureKeyRef.current = nextResolutionKey;
    const matchingIndex = imageDetail.annotation.bboxes.findIndex((bbox) => bbox.label === q3State.featureLabel);
    if (matchingIndex >= 0) {
      updateQ3State({ bboxIndex: matchingIndex }, true);
    }
  }, [decodedId, imageDetail, isQ3Tab, q3BboxParam, q3State.featureLabel, updateQ3State]);

  useEffect(() => {
    if (!isQ3Tab || q3State.head === null) {
      return;
    }
    if (q3VariantPerHeadAvailable === false) {
      updateQ3State({ head: null }, true);
      return;
    }
    if (q3HeadRankingQuery.data?.supported === false) {
      updateQ3State({ head: null }, true);
      return;
    }
    if (q3HeadRankingQuery.data && !q3SelectedHeadAvailable) {
      updateQ3State({ head: null }, true);
    }
  }, [isQ3Tab, q3HeadRankingQuery.data, q3SelectedHeadAvailable, q3State.head, q3VariantPerHeadAvailable, updateQ3State]);

  const activeMode = isQ3Tab ? q3State.mode : 'head_attention';
  const activeModel = isQ3Tab ? q3ViewerModel : mainModel;
  const activeLayer = isQ3Tab ? q3State.layer : mainLayer;
  const activeMethod = isQ3Tab ? q3Method : mainMethod;
  const activeHead = isQ3Tab
    ? (q3State.head !== null && q3VariantPerHeadAvailable !== false && q3SelectedHeadAvailable ? q3State.head : null)
    : mainHead;
  const activePercentile = isQ3Tab ? q3ViewerPercentile : mainPercentile;
  const activeShowBboxes = isQ3Tab ? q3State.showBboxes : mainShowBboxes;
  const activeBboxIndex = isQ3Tab ? q3State.bboxIndex : mainSelectedBboxIndex;
  const handleActiveBboxSelect = isQ3Tab ? handleQ3BboxSelect : handleMainBboxSelect;
  const centerColumnClassName = isQ3Tab
    ? 'order-3 min-w-0 space-y-4 lg:order-2 lg:col-span-5 xl:col-span-1'
    : 'min-w-0 space-y-4 lg:col-span-5 xl:col-span-1';
  const rightColumnClassName = isQ3Tab
    ? 'order-2 min-w-0 space-y-4 lg:order-3 lg:col-span-4 xl:col-span-1'
    : 'min-w-0 space-y-4 lg:col-span-4 xl:col-span-1';

  if (!decodedId) {
    return <div>Invalid image ID</div>;
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          Failed to load image: {decodedId}
        </div>
      </div>
    );
  }

  if (!detailLoading && !imageDetail) {
    return (
      <div className="space-y-4">
        <Link to="/" className="text-primary-600 hover:underline">
          &larr; Back to gallery
        </Link>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-amber-900">
          Image not found: {decodedId}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-sm">
        <Link to="/" className="text-primary-600 hover:underline">
          Gallery
        </Link>
        <span className="text-gray-400">/</span>
        <span className="text-gray-600">{decodedId}</span>
      </div>

      <PageTabs
        label="Image Detail sections"
        activeTab={currentTab}
        onChange={handleTabChange}
        tabs={[
          {
            value: 'main',
            label: 'Image Detail',
            id: 'image-detail-page-tab-main',
            dataTestId: 'image-detail-page-tab-main',
          },
          {
            value: 'q3',
            label: 'Q3',
            id: 'image-detail-page-tab-q3',
            dataTestId: 'image-detail-page-tab-q3',
          },
        ]}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 xl:grid-cols-[22rem_minmax(0,1fr)_minmax(0,1fr)] xl:gap-8">
        <div
          className="space-y-4 lg:col-span-3 xl:col-span-1"
          data-testid="image-detail-left-column"
        >
          {!isQ3Tab && (
            <div data-testid="view-settings-panel">
              <ControlPanel mode="head_attention" />
            </div>
          )}
          {isQ3Tab && (
            <div data-testid="q3-controls-panel">
              <Q3ImageDetailControls
                model={q3State.model}
                variant={q3State.variant}
                layer={q3State.layer}
                head={q3State.head}
                rankingMetric={q3State.metric}
                maxLayer={q3MaxLayer}
                numHeads={q3NumHeads}
                showBboxes={q3State.showBboxes}
                featureName={q3State.featureName}
                rankingData={q3HeadRankingQuery.data}
                rankingLoading={q3HeadRankingQuery.isLoading}
                rankingError={q3HeadRankingError}
                variantSupportsPerHead={q3VariantPerHeadAvailable}
                onModelChange={handleQ3ModelChange}
                onVariantChange={handleQ3VariantChange}
                onLayerChange={handleQ3LayerChange}
                onHeadChange={handleQ3HeadChange}
                onMetricChange={handleQ3MetricChange}
                onShowBboxesChange={handleQ3ShowBboxesChange}
              />
            </div>
          )}
        </div>

        <div
          className={centerColumnClassName}
          data-testid="image-detail-center-column"
        >
          {isQ3Tab && (
            <ImageDetailModeSwitch
              mode={q3State.mode}
              onChange={handleQ3ModeChange}
            />
          )}

          <ErrorBoundary resetKeys={[activeModel, activeLayer, activeMethod, activeHead, activeMode, activePercentile]}>
            <AttentionViewer
              imageId={decodedId}
              model={activeModel}
              layer={activeLayer}
              method={activeMethod}
              head={activeHead}
              mode={activeMode}
              percentile={activePercentile}
              showBboxes={activeShowBboxes}
              bboxSelectionDrivesOverlay={!isQ3Tab}
              bboxes={imageDetail?.annotation.bboxes ?? []}
              selectedBboxIndex={activeBboxIndex}
              onBboxSelect={handleActiveBboxSelect}
              className="aspect-square"
            />
          </ErrorBoundary>

          {!isQ3Tab && (
            <Card>
              <CardContent>
                <LayerSlider
                  currentLayer={mainLayer}
                  maxLayers={modelsData?.num_layers_per_model?.[mainModel] ?? modelsData?.num_layers ?? 12}
                  onChange={setMainLayer}
                  isPlaying={isPlaying}
                  onPlayingChange={setIsPlaying}
                  playSpeed={400}
                />
              </CardContent>
            </Card>
          )}

          {imageDetail && (
            <AnnotationsCard
              annotation={imageDetail.annotation}
              mode={activeMode}
              showBboxes={activeShowBboxes}
              bboxSelectionDrivesOverlay={!isQ3Tab}
              selectedBboxIndex={activeBboxIndex}
              onBboxSelect={handleActiveBboxSelect}
            />
          )}
        </div>

        <div className={rightColumnClassName} data-testid="image-detail-right-column">
          {isQ3Tab && (
            <Q3StudyScopeCallout
              context="imageDetail"
              dataTestId="image-detail-q3-scope-card"
              currentModelLabel={q3State.model}
              currentModelStatus={getQ3ModelScopeStatus(q3State.model)}
              action={{
                label: 'Use Q3 defaults',
                onClick: handleApplyQ3Defaults,
                dataTestId: 'image-detail-use-q3-defaults',
              }}
            />
          )}

          {!isQ3Tab && (
            <ErrorBoundary resetKeys={[mainModel, mainLayer, mainPercentile, mainMethod, activeBboxIndex, isPlaying]}>
              <ImageDetailMetricsPanel
                imageId={decodedId}
                model={mainModel}
                percentile={mainPercentile}
                method={mainMethod}
                mode="head_attention"
                bboxSelectionDrivesOverlay
                selectedBboxIndex={mainSelectedBboxIndex}
                currentLayer={mainLayer}
                isPlaying={isPlaying}
                enabled={!!decodedId}
              />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </div>
  );
}
