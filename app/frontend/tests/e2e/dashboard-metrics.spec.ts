import { expect, test, type Locator, type Page } from '@playwright/test';

const EXEMPLAR_IMAGE_ID = 'Q2034923_wd0.jpg';

const METRIC_SUMMARY_FIXTURES = {
  iou: {
    leaderboard: [
      { rank: 1, model: 'dinov2', metric: 'iou', score: 0.58, best_layer: 'layer11', method_used: 'cls' },
      { rank: 2, model: 'clip', metric: 'iou', score: 0.31, best_layer: 'layer10', method_used: 'cls' },
    ],
    models: {
      dinov2: {
        rank: 1,
        best_layer: 'layer11',
        best_score: 0.58,
        method_used: 'cls',
        layer_progression: { layer0: 0.22, layer1: 0.31, layer6: 0.44, layer11: 0.58 },
      },
      clip: {
        rank: 2,
        best_layer: 'layer10',
        best_score: 0.31,
        method_used: 'cls',
        layer_progression: { layer0: 0.11, layer1: 0.16, layer7: 0.28, layer10: 0.31 },
      },
    },
  },
  coverage: {
    leaderboard: [
      { rank: 1, model: 'clip', metric: 'coverage', score: 0.33, best_layer: 'layer9', method_used: 'cls' },
      { rank: 2, model: 'dinov2', metric: 'coverage', score: 0.29, best_layer: 'layer11', method_used: 'cls' },
    ],
    models: {
      clip: {
        rank: 1,
        best_layer: 'layer9',
        best_score: 0.33,
        method_used: 'cls',
        layer_progression: { layer0: 0.18, layer1: 0.24, layer6: 0.29, layer9: 0.33 },
      },
      dinov2: {
        rank: 2,
        best_layer: 'layer11',
        best_score: 0.29,
        method_used: 'cls',
        layer_progression: { layer0: 0.14, layer1: 0.19, layer7: 0.25, layer11: 0.29 },
      },
    },
  },
  mse: {
    leaderboard: [
      { rank: 1, model: 'dinov2', metric: 'mse', score: 0.012, best_layer: 'layer8', method_used: 'cls' },
      { rank: 2, model: 'clip', metric: 'mse', score: 0.018, best_layer: 'layer10', method_used: 'cls' },
    ],
    models: {
      dinov2: {
        rank: 1,
        best_layer: 'layer8',
        best_score: 0.012,
        method_used: 'cls',
        layer_progression: { layer0: 0.03, layer1: 0.027, layer8: 0.012, layer11: 0.015 },
      },
      clip: {
        rank: 2,
        best_layer: 'layer10',
        best_score: 0.018,
        method_used: 'cls',
        layer_progression: { layer0: 0.036, layer1: 0.031, layer7: 0.022, layer10: 0.018 },
      },
    },
  },
  kl: {
    leaderboard: [
      { rank: 1, model: 'dinov2', metric: 'kl', score: 2.8, best_layer: 'layer10', method_used: 'cls' },
      { rank: 2, model: 'clip', metric: 'kl', score: 3.6, best_layer: 'layer11', method_used: 'cls' },
    ],
    models: {
      dinov2: {
        rank: 1,
        best_layer: 'layer10',
        best_score: 2.8,
        method_used: 'cls',
        layer_progression: { layer0: 6.2, layer1: 5.4, layer7: 3.4, layer10: 2.8 },
      },
      clip: {
        rank: 2,
        best_layer: 'layer11',
        best_score: 3.6,
        method_used: 'cls',
        layer_progression: { layer0: 5.8, layer1: 5.1, layer8: 4.1, layer11: 3.6 },
      },
    },
  },
  emd: {
    leaderboard: [
      { rank: 1, model: 'dinov2', metric: 'emd', score: 2.6, best_layer: 'layer9', method_used: 'cls' },
      { rank: 2, model: 'clip', metric: 'emd', score: 2.9, best_layer: 'layer10', method_used: 'cls' },
    ],
    models: {
      dinov2: {
        rank: 1,
        best_layer: 'layer9',
        best_score: 2.6,
        method_used: 'cls',
        layer_progression: { layer0: 4.4, layer1: 4.0, layer6: 3.1, layer9: 2.6 },
      },
      clip: {
        rank: 2,
        best_layer: 'layer10',
        best_score: 2.9,
        method_used: 'cls',
        layer_progression: { layer0: 4.2, layer1: 3.7, layer7: 3.2, layer10: 2.9 },
      },
    },
  },
} as const;

interface StubQ3HeadRankingEntry {
  head: number;
  mean_score: number;
  std_score: number;
  mean_rank: number;
  top1_count: number;
  top3_count: number;
  image_count: number;
}

interface StubQ3HeadFeatureMatrixFeature {
  feature_label: number;
  feature_name: string;
  bbox_count: number;
  scores: Array<number | null>;
}

interface DashboardStubOptions {
  headRankingOverride?: (params: {
    model: string;
    metric: string;
    variant: string;
    percentile: number;
  }) => {
    supported: boolean;
    reason: string | null;
    heads: StubQ3HeadRankingEntry[];
  } | null;
  headFeatureMatrixOverride?: (params: {
    model: string;
    metric: string;
    variant: string;
    percentile: number;
  }) => {
    supported: boolean;
    reason: string | null;
    heads: number[];
    total_feature_types: number;
    features: StubQ3HeadFeatureMatrixFeature[];
  } | null;
}

const Q3_HIGHER_DIRECTION_HEAD_RANKINGS: Record<string, StubQ3HeadRankingEntry[]> = {
  frozen: [
    { head: 0, mean_score: 0.44, std_score: 0.04, mean_rank: 1.2, top1_count: 8, top3_count: 11, image_count: 12 },
    { head: 1, mean_score: 0.38, std_score: 0.05, mean_rank: 1.95, top1_count: 3, top3_count: 9, image_count: 12 },
    { head: 2, mean_score: 0.32, std_score: 0.05, mean_rank: 2.85, top1_count: 1, top3_count: 6, image_count: 12 },
  ],
  lora: [
    { head: 1, mean_score: 0.47, std_score: 0.04, mean_rank: 1.1, top1_count: 7, top3_count: 11, image_count: 12 },
    { head: 0, mean_score: 0.39, std_score: 0.05, mean_rank: 2.05, top1_count: 4, top3_count: 10, image_count: 12 },
    { head: 2, mean_score: 0.31, std_score: 0.05, mean_rank: 2.9, top1_count: 1, top3_count: 5, image_count: 12 },
  ],
  full: [
    { head: 0, mean_score: 0.46, std_score: 0.04, mean_rank: 1.05, top1_count: 8, top3_count: 12, image_count: 12 },
    { head: 2, mean_score: 0.36, std_score: 0.04, mean_rank: 1.9, top1_count: 3, top3_count: 9, image_count: 12 },
    { head: 1, mean_score: 0.35, std_score: 0.06, mean_rank: 3.0, top1_count: 1, top3_count: 6, image_count: 12 },
  ],
  linear_probe: [
    { head: 0, mean_score: 0.29, std_score: 0.04, mean_rank: 1.35, top1_count: 5, top3_count: 9, image_count: 12 },
    { head: 1, mean_score: 0.24, std_score: 0.05, mean_rank: 2.1, top1_count: 2, top3_count: 7, image_count: 12 },
    { head: 2, mean_score: 0.21, std_score: 0.04, mean_rank: 2.8, top1_count: 1, top3_count: 5, image_count: 12 },
  ],
};

const Q3_LOWER_DIRECTION_HEAD_RANKINGS: Record<string, StubQ3HeadRankingEntry[]> = {
  frozen: [
    { head: 0, mean_score: 0.022, std_score: 0.003, mean_rank: 1.15, top1_count: 8, top3_count: 11, image_count: 12 },
    { head: 1, mean_score: 0.03, std_score: 0.004, mean_rank: 1.9, top1_count: 3, top3_count: 9, image_count: 12 },
    { head: 2, mean_score: 0.041, std_score: 0.005, mean_rank: 2.95, top1_count: 1, top3_count: 6, image_count: 12 },
  ],
  lora: [
    { head: 1, mean_score: 0.018, std_score: 0.003, mean_rank: 1.05, top1_count: 7, top3_count: 11, image_count: 12 },
    { head: 0, mean_score: 0.027, std_score: 0.004, mean_rank: 2.1, top1_count: 4, top3_count: 10, image_count: 12 },
    { head: 2, mean_score: 0.043, std_score: 0.005, mean_rank: 2.9, top1_count: 1, top3_count: 5, image_count: 12 },
  ],
  full: [
    { head: 0, mean_score: 0.02, std_score: 0.003, mean_rank: 1.0, top1_count: 8, top3_count: 12, image_count: 12 },
    { head: 2, mean_score: 0.034, std_score: 0.004, mean_rank: 1.9, top1_count: 3, top3_count: 9, image_count: 12 },
    { head: 1, mean_score: 0.033, std_score: 0.005, mean_rank: 3.0, top1_count: 1, top3_count: 6, image_count: 12 },
  ],
  linear_probe: [
    { head: 0, mean_score: 0.028, std_score: 0.003, mean_rank: 1.3, top1_count: 5, top3_count: 9, image_count: 12 },
    { head: 1, mean_score: 0.036, std_score: 0.004, mean_rank: 2.15, top1_count: 2, top3_count: 7, image_count: 12 },
    { head: 2, mean_score: 0.045, std_score: 0.005, mean_rank: 2.85, top1_count: 1, top3_count: 5, image_count: 12 },
  ],
};

function getStubQ3HeadRankingEntries(metric: string, variant: string): StubQ3HeadRankingEntry[] {
  const fixtures = metric === 'iou' || metric === 'coverage'
    ? Q3_HIGHER_DIRECTION_HEAD_RANKINGS
    : Q3_LOWER_DIRECTION_HEAD_RANKINGS;
  return (fixtures[variant] ?? fixtures.frozen).map((entry) => ({ ...entry }));
}

async function stubDashboardApis(page: Page, options: DashboardStubOptions = {}) {
  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2', 'resnet50'],
        num_layers: 12,
        num_layers_per_model: {
          dinov2: 12,
          dinov3: 12,
          mae: 12,
          clip: 12,
          siglip: 12,
          siglip2: 12,
          resnet50: 4,
        },
        methods: {
          dinov2: ['cls', 'rollout'],
          dinov3: ['cls', 'rollout'],
          mae: ['cls', 'rollout'],
          clip: ['cls', 'rollout'],
          siglip: ['mean'],
          siglip2: ['mean'],
          resnet50: ['gradcam'],
        },
        num_heads_per_model: {
          dinov2: 12,
          dinov3: 12,
          mae: 12,
          clip: 12,
          siglip: 12,
          siglip2: 12,
          resnet50: 0,
        },
        per_head_methods: ['cls', 'mean'],
        per_head_available_models: ['clip', 'dinov2', 'dinov3', 'mae'],
        default_methods: {
          dinov2: 'cls',
          dinov3: 'cls',
          mae: 'cls',
          clip: 'cls',
          siglip: 'mean',
          siglip2: 'mean',
          resnet50: 'gradcam',
        },
      }),
    });
  });

  await page.route('**/api/compare/all_models_summary?**', async (route) => {
    const url = new URL(route.request().url());
    const metric = (url.searchParams.get('metric') ?? 'iou') as keyof typeof METRIC_SUMMARY_FIXTURES;
    const fixture = METRIC_SUMMARY_FIXTURES[metric] ?? METRIC_SUMMARY_FIXTURES.iou;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        percentile: 90,
        metric,
        ranking_mode: 'default_method',
        method: null,
        excluded_models: [],
        leaderboard: fixture.leaderboard,
        models: fixture.models,
      }),
    });
  });

  await page.route('**/api/metrics/model/*/style_breakdown?**', async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/style_breakdown/);
    const model = match?.[1] ?? 'dinov2';
    const metric = url.searchParams.get('metric') ?? 'iou';
    const direction = metric === 'iou' || metric === 'coverage' ? 'higher' : 'lower';
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model,
        layer: 'layer0',
        metric,
        direction,
        percentile: 90,
        method: 'cls',
        styles: { Gothic: 0.55, Romanesque: 0.48 },
        style_counts: { Gothic: 20, Romanesque: 15 },
      }),
    });
  });

  await page.route('**/api/metrics/model/*/feature_breakdown?**', async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/feature_breakdown/);
    const model = match?.[1] ?? 'dinov2';
    const metric = url.searchParams.get('metric') ?? 'iou';
    const direction = metric === 'iou' || metric === 'coverage' ? 'higher' : 'lower';
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model,
        layer: 'layer0',
        metric,
        direction,
        percentile: 90,
        method: 'cls',
        total_feature_types: 1,
        features: [
          {
            feature_label: 1,
            feature_name: 'Window',
            mean_score: 0.41,
            std_score: 0.05,
            bbox_count: 12,
          },
        ],
      }),
    });
  });

  await page.route('**/api/metrics/model/*/head_ranking?**', async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/head_ranking/);
    const model = match?.[1] ?? 'dinov2';
    const metric = url.searchParams.get('metric') ?? 'iou';
    const variant = url.searchParams.get('variant') ?? 'frozen';
    const percentile = Number(url.searchParams.get('percentile') ?? '90');
    const direction = metric === 'iou' || metric === 'coverage' ? 'higher' : 'lower';
    const override = options.headRankingOverride?.({
      model,
      metric,
      variant,
      percentile,
    });
    const supported = override?.supported ?? model !== 'resnet50';
    const reason = override?.reason ?? (
      model === 'resnet50' ? 'Q3 per-head analysis is not supported for model \'resnet50\'.' : null
    );
    const heads = override?.heads ?? (
      model === 'resnet50' ? [] : getStubQ3HeadRankingEntries(metric, variant)
    );
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model,
        variant,
        layer: 'layer11',
        method: model === 'siglip2' ? 'mean' : model === 'resnet50' ? null : 'cls',
        metric,
        direction,
        percentile,
        supported,
        reason,
        heads,
      }),
    });
  });

  await page.route('**/api/metrics/model/*/head_feature_matrix?**', async (route) => {
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/head_feature_matrix/);
    const model = match?.[1] ?? 'dinov2';
    const metric = url.searchParams.get('metric') ?? 'iou';
    const variant = url.searchParams.get('variant') ?? 'frozen';
    const percentile = Number(url.searchParams.get('percentile') ?? '90');
    const direction = metric === 'iou' || metric === 'coverage' ? 'higher' : 'lower';
    const override = options.headFeatureMatrixOverride?.({
      model,
      metric,
      variant,
      percentile,
    });
    const supported = override?.supported ?? model !== 'resnet50';
    const reason = override?.reason ?? (
      model === 'resnet50' ? 'Q3 per-head analysis is not supported for model \'resnet50\'.' : null
    );
    const heads = override?.heads ?? (model === 'resnet50' ? [] : [0, 1, 2]);
    const features = override?.features ?? (
      model === 'resnet50'
        ? []
        : [
            { feature_label: 7, feature_name: 'Door', bbox_count: 5, scores: [0.21, 0.33, 0.27] },
            { feature_label: 42, feature_name: 'Window', bbox_count: 9, scores: [0.48, 0.51, 0.45] },
          ]
    );
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model,
        variant,
        layer: 'layer11',
        method: model === 'siglip2' ? 'mean' : model === 'resnet50' ? null : 'cls',
        metric,
        direction,
        percentile,
        supported,
        reason,
        heads,
        total_feature_types: override?.total_feature_types ?? (model === 'resnet50' ? 0 : 2),
        features,
      }),
    });
  });

  await page.route('**/api/metrics/model/*/head_exemplars?**', async (route) => {
    const url = new URL(route.request().url());
    const featureLabel = url.searchParams.get('feature_label');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model: 'dinov2',
        variant: url.searchParams.get('variant') ?? 'frozen',
        layer: `layer${url.searchParams.get('layer') ?? '11'}`,
        metric: url.searchParams.get('metric') ?? 'iou',
        direction: 'higher',
        percentile: Number(url.searchParams.get('percentile') ?? '90'),
        head: Number(url.searchParams.get('head') ?? '0'),
        feature_label: featureLabel === null ? null : Number(featureLabel),
        feature_name: featureLabel === null ? null : 'Door',
        supported: true,
        reason: null,
        candidates: [
          {
            image_id: EXEMPLAR_IMAGE_ID,
            score: 0.72,
            thumbnail_url: `/api/images/${EXEMPLAR_IMAGE_ID}/thumbnail`,
            style_names: ['Gothic'],
            matching_bbox_indices: featureLabel === null ? [] : [0],
            default_bbox_index: featureLabel === null ? null : 0,
          },
        ],
      }),
    });
  });
}

async function stubQ2SummaryApi(page: Page) {
  await page.route('**/api/metrics/q2_summary?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        metric: 'iou',
        label: 'IoU',
        direction: 'higher',
        percentile_dependent: true,
        selected_percentile: 90,
        experiment_id: 'exp_test',
        split_id: 'split_test',
        analysis_git_commit_sha: 'deadbeef',
        analyzed_layer: 11,
        evaluation_image_count: 139,
        checkpoint_selection_rule:
          'best classification validation accuracy on shared non-annotated validation split',
        result_set_scope: 'primary',
        timestamp: null,
        rows: [],
        strategy_comparisons: [],
      }),
    });
  });
}

async function getYAxisTickValues(chart: Locator): Promise<number[]> {
  const textValues = await chart.locator('svg text').allTextContents();
  return textValues
    .map((value) => value.trim())
    .filter((value) => /^\d+(?:\.\d+)?$/.test(value))
    .map((value) => Number(value));
}

test.describe('Dashboard metrics', () => {
  test('supports selecting EMD and shows the threshold-free guidance', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard');

    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await page.getByRole('combobox').first().selectOption('emd');

    await expect(page.getByText('EMD (lower better)')).toBeVisible();
    await expect(
      page.getByText(/Earth Mover's Distance \/ Wasserstein-1 on a shared 8x8 support/)
    ).toBeVisible();
  });

  test('shows documented baseline legend only for continuous metrics in the leaderboard card', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard');

    await expect(page.getByTestId('leaderboard-score-chart')).toBeVisible();
    await expect(page.getByTestId('leaderboard-baseline-legend')).toHaveCount(0);

    const metricSelect = page.getByRole('combobox').first();
    await metricSelect.selectOption('mse');

    await expect(page.getByTestId('leaderboard-baseline-legend')).toBeVisible();
    await expect(page.getByTestId('leaderboard-baseline-random')).toContainText('Random');
    await expect(page.getByTestId('leaderboard-baseline-random')).toContainText('0.3192');
    await expect(page.getByTestId('leaderboard-baseline-center_gaussian')).toContainText('Center Gaussian');
    await expect(page.getByTestId('leaderboard-baseline-sobel_edge')).toContainText('Sobel Edge');

    await metricSelect.selectOption('coverage');
    await expect(page.getByTestId('leaderboard-baseline-legend')).toHaveCount(0);
  });

  test('rescales the layer progression y-axis for every dashboard metric', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard');

    const metricSelect = page.getByRole('combobox').first();
    const chart = page.getByTestId('dashboard-layer-progression-chart');
    const expectedMaxTicks: Array<[string, number]> = [
      ['iou', 0.6],
      ['coverage', 0.4],
      ['mse', 0.04],
      ['kl', 8],
      ['emd', 5],
      ['iou', 0.6],
    ];

    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();

    for (const [metric, expectedMaxTick] of expectedMaxTicks) {
      await metricSelect.selectOption(metric);
      await expect.poll(async () => {
        const ticks = await getYAxisTickValues(chart);
        return Math.max(...ticks);
      }).toBe(expectedMaxTick);
    }

    await expect.poll(async () => {
      const ticks = await getYAxisTickValues(chart);
      return ticks.includes(1);
    }).toBe(false);
  });

  test('switches dashboard ranking modes and uses the selected row method', async ({ page }) => {
    await page.route('**/api/attention/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2', 'resnet50'],
          num_layers: 12,
          num_layers_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 4,
          },
          methods: {
            dinov2: ['cls', 'rollout'],
            dinov3: ['cls', 'rollout'],
            mae: ['cls', 'rollout'],
            clip: ['cls', 'rollout'],
            siglip: ['mean'],
            siglip2: ['mean'],
            resnet50: ['gradcam'],
          },
          num_heads_per_model: {
            dinov2: 12,
            dinov3: 12,
            mae: 12,
            clip: 12,
            siglip: 12,
            siglip2: 12,
            resnet50: 0,
          },
          per_head_methods: ['cls', 'mean'],
          default_methods: {
            dinov2: 'cls',
            dinov3: 'cls',
            mae: 'cls',
            clip: 'cls',
            siglip: 'mean',
            siglip2: 'mean',
            resnet50: 'gradcam',
          },
        }),
      });
    });

    await page.route('**/api/compare/all_models_summary?**', async (route) => {
      const url = new URL(route.request().url());
      const rankingMode = url.searchParams.get('ranking_mode');
      const isBestAvailable = rankingMode === 'best_available';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          percentile: 90,
          metric: 'iou',
          ranking_mode: isBestAvailable ? 'best_available' : 'default_method',
          method: null,
          excluded_models: [],
          leaderboard: isBestAvailable
            ? [
                { rank: 1, model: 'dinov2', metric: 'iou', score: 0.64, best_layer: 'layer10', method_used: 'rollout' },
                { rank: 2, model: 'clip', metric: 'iou', score: 0.57, best_layer: 'layer7', method_used: 'rollout' },
                { rank: 3, model: 'dinov3', metric: 'iou', score: 0.54, best_layer: 'layer10', method_used: 'cls' },
                { rank: 4, model: 'mae', metric: 'iou', score: 0.5, best_layer: 'layer9', method_used: 'cls' },
              ]
            : [
                { rank: 1, model: 'dinov2', metric: 'iou', score: 0.58, best_layer: 'layer11', method_used: 'cls' },
                { rank: 2, model: 'dinov3', metric: 'iou', score: 0.54, best_layer: 'layer10', method_used: 'cls' },
                { rank: 3, model: 'mae', metric: 'iou', score: 0.5, best_layer: 'layer9', method_used: 'cls' },
                { rank: 4, model: 'clip', metric: 'iou', score: 0.46, best_layer: 'layer8', method_used: 'cls' },
              ],
          models: {
            dinov2: {
              rank: 1,
              best_layer: isBestAvailable ? 'layer10' : 'layer11',
              best_score: isBestAvailable ? 0.64 : 0.58,
              method_used: isBestAvailable ? 'rollout' : 'cls',
              layer_progression: isBestAvailable
                ? { layer0: 0.28, layer1: 0.39, layer10: 0.64 }
                : { layer0: 0.22, layer1: 0.31, layer11: 0.58 },
            },
            dinov3: {
              rank: isBestAvailable ? 3 : 2,
              best_layer: 'layer10',
              best_score: 0.54,
              method_used: 'cls',
              layer_progression: { layer0: 0.2, layer1: 0.29, layer10: 0.54 },
            },
            mae: {
              rank: 4,
              best_layer: 'layer9',
              best_score: 0.5,
              method_used: 'cls',
              layer_progression: { layer0: 0.18, layer1: 0.27, layer9: 0.5 },
            },
            clip: {
              rank: isBestAvailable ? 2 : 4,
              best_layer: isBestAvailable ? 'layer7' : 'layer8',
              best_score: isBestAvailable ? 0.57 : 0.46,
              method_used: isBestAvailable ? 'rollout' : 'cls',
              layer_progression: isBestAvailable
                ? { layer0: 0.21, layer1: 0.33, layer7: 0.57 }
                : { layer0: 0.16, layer1: 0.24, layer8: 0.46 },
            },
          },
        }),
      });
    });

    await page.route('**/api/metrics/model/*/style_breakdown?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/style_breakdown/);
      const model = match?.[1] ?? 'dinov2';
      const scores = model === 'clip'
        ? { Gothic: 0.43, Romanesque: 0.35 }
        : { Gothic: 0.55, Romanesque: 0.48 };
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          layer: 'layer0',
          metric: 'iou',
          direction: 'higher',
          percentile: 90,
          method: 'cls',
          styles: scores,
          style_counts: { Gothic: 20, Romanesque: 15 },
        }),
      });
    });

    await page.route('**/api/metrics/model/*/feature_breakdown?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/feature_breakdown/);
      const model = match?.[1] ?? 'dinov2';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          layer: 'layer0',
          metric: 'iou',
          direction: 'higher',
          percentile: 90,
          method: 'cls',
          total_feature_types: 1,
          features: [
            {
              feature_label: 1,
              feature_name: 'Window',
              mean_score: model === 'clip' ? 0.32 : 0.41,
              std_score: 0.05,
              bbox_count: 12,
            },
          ],
        }),
      });
    });

    await page.route('**/api/metrics/model/*/head_ranking?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/head_ranking/);
      const model = match?.[1] ?? 'dinov2';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          variant: url.searchParams.get('variant') ?? 'frozen',
          layer: 'layer11',
          method: model === 'siglip' || model === 'siglip2' ? 'mean' : model === 'resnet50' ? null : 'cls',
          metric: url.searchParams.get('metric') ?? 'iou',
          direction: 'higher',
          percentile: Number(url.searchParams.get('percentile') ?? '90'),
          supported: model !== 'resnet50',
          reason: model === 'resnet50' ? 'Q3 per-head analysis is not supported for model \'resnet50\'.' : null,
          heads: model === 'resnet50'
            ? []
            : [
                { head: 0, mean_score: 0.44, std_score: 0.04, mean_rank: 1.2, top1_count: 8, top3_count: 11, image_count: 12 },
              ],
        }),
      });
    });

    await page.route('**/api/metrics/model/*/head_feature_matrix?**', async (route) => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/api\/metrics\/model\/([^/]+)\/head_feature_matrix/);
      const model = match?.[1] ?? 'dinov2';
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          model,
          variant: url.searchParams.get('variant') ?? 'frozen',
          layer: 'layer11',
          method: model === 'siglip' || model === 'siglip2' ? 'mean' : model === 'resnet50' ? null : 'cls',
          metric: url.searchParams.get('metric') ?? 'iou',
          direction: 'higher',
          percentile: Number(url.searchParams.get('percentile') ?? '90'),
          supported: model !== 'resnet50',
          reason: model === 'resnet50' ? 'Q3 per-head analysis is not supported for model \'resnet50\'.' : null,
          heads: model === 'resnet50' ? [] : [0, 1, 2],
          total_feature_types: model === 'resnet50' ? 0 : 1,
          features: model === 'resnet50'
            ? []
            : [{ feature_label: 1, feature_name: 'Window', bbox_count: 12, scores: [0.41, 0.35, 0.28] }],
        }),
      });
    });

    const summaryResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/all_models_summary')
        && response.url().includes('ranking_mode=default_method')
        && response.status() === 200
    );

    await page.goto('/dashboard');
    await summaryResponse;
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      'Leaderboard and layer progression rank each model by its default attention method.'
    );
    await expect(page.getByTestId('leaderboard-row-meta-clip')).toContainText('CLS Attention');

    await expect(page.getByTestId('leaderboard-row-dinov2')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-dinov3')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-mae')).toBeVisible();
    await expect(page.getByTestId('leaderboard-row-clip')).toBeVisible();

    const bestAvailableResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/all_models_summary')
        && response.url().includes('ranking_mode=best_available')
        && response.status() === 200
    );

    await page.getByTestId('dashboard-ranking-mode-best_available').click();
    await bestAvailableResponse;

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      "Leaderboard and layer progression rank each model by its strongest available attention method."
    );
    await expect(page.getByTestId('leaderboard-row-meta-dinov2')).toContainText('Rollout Attention');
    await expect(page.getByTestId('leaderboard-row-meta-clip')).toContainText('Rollout Attention');

    const clipStyleResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/metrics/model/clip/style_breakdown')
        && response.url().includes('method=rollout')
        && response.status() === 200
    );

    await page.getByTestId('leaderboard-row-clip').click();
    await clipStyleResponse;

    await expect(page.getByTestId('dashboard-method-context')).toContainText(
      "Leaderboard and layer progression rank each model by its strongest available attention method."
    );
  });

  test('keeps Q3 analysis behind the Q3 tab and supports metric switching there', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard');

    const overviewTab = page.getByRole('tab', { name: 'Overview' });
    const q3Tab = page.getByRole('tab', { name: 'Q3' });

    await expect(overviewTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('dashboard-main-panel')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toHaveCount(0);

    await q3Tab.click();

    await expect(q3Tab).toHaveAttribute('aria-selected', 'true');
    const q3Panel = page.getByTestId('dashboard-q3-panel');

    await expect(q3Panel).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toBeVisible();
    await expect(q3Panel.getByRole('combobox', { name: 'View' })).toHaveValue('head-ranking');
    await expect(q3Panel.getByRole('combobox', { name: 'Model' })).toHaveValue('dinov3');
    await expect(q3Panel.getByRole('combobox', { name: 'Variant' })).toHaveValue('frozen');
    await expect(q3Panel.getByRole('combobox', { name: 'Layer' })).toHaveValue('10');
    await expect(q3Panel.getByRole('combobox', { name: 'Metric' })).toHaveValue('iou');
    await expect(q3Panel.getByRole('combobox', { name: 'Percentile' })).toHaveValue('90');
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toBeVisible();
    await expect(q3Panel.getByTestId('q3-report-matrix-view')).toHaveCount(0);
    await expect(q3Panel.getByTestId('q3-report-delta-view')).toHaveCount(0);
    await expect(q3Panel.getByText('Head 0').first()).toBeVisible();

    await q3Panel.getByRole('combobox', { name: 'Metric' }).selectOption('coverage');
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toContainText('mean Coverage');
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toContainText('Coverage 0.440');
    await expect(q3Panel.getByText(/This selection stays inside the primary Q3 workflow/)).toBeVisible();

    await overviewTab.click();
    await expect(overviewTab).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toHaveCount(0);
  });

  test('switches Dashboard Q3 between head ranking, matrix, and frozen-to-adapted delta views', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard?tab=q3');

    const q3Panel = page.getByTestId('dashboard-q3-panel');
    const viewSelect = q3Panel.getByRole('combobox', { name: 'View' });
    const variantSelect = q3Panel.getByRole('combobox', { name: 'Variant' });

    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toBeVisible();
    await expect(q3Panel.getByText('Head Ranking').last()).toBeVisible();
    await expect(q3Panel.getByText('Head-Feature Matrix')).toHaveCount(0);

    await viewSelect.selectOption('head-feature-matrix');
    await expect(q3Panel.getByTestId('q3-report-matrix-view')).toBeVisible();
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toHaveCount(0);
    await expect(q3Panel.getByTestId('q3-report-delta-view')).toHaveCount(0);
    await expect(q3Panel.getByText('Head-Feature Matrix')).toBeVisible();
    await expect(q3Panel.getByTestId('q3-report-matrix-cell-7-1')).toBeVisible();

    await q3Panel.getByTestId('q3-report-matrix-cell-7-1').click();
    await expect(q3Panel.getByTestId('q3-report-selected-cell')).toContainText('Door');
    await expect(q3Panel.getByTestId('q3-report-selected-cell')).toContainText('layer10/head1');

    await viewSelect.selectOption('frozen-delta');
    await expect(q3Panel.getByTestId('q3-report-delta-view')).toBeVisible();
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toHaveCount(0);
    await expect(q3Panel.getByTestId('q3-report-matrix-view')).toHaveCount(0);
    await expect(variantSelect).toBeDisabled();
    await expect(q3Panel.getByTestId('q3-selection-helper')).toContainText(
      'compares Frozen against LoRA and Full'
    );

    const loraDelta = q3Panel.getByTestId('q3-report-delta-lora');
    const fullDelta = q3Panel.getByTestId('q3-report-delta-full');
    await expect(loraDelta).toContainText('Frozen -> LoRA');
    await expect(loraDelta).toContainText('Promoted');
    await expect(loraDelta).toContainText('Demoted');
    await expect(loraDelta).toContainText('Stable');
    await expect(fullDelta).toContainText('Frozen -> Full');

    await viewSelect.selectOption('head-ranking');
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toBeVisible();
    await expect(variantSelect).toBeEnabled();
  });

  test('formats frozen-to-adapted score deltas for lower-is-better metrics', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard?tab=q3');

    const q3Panel = page.getByTestId('dashboard-q3-panel');

    await q3Panel.getByRole('combobox', { name: 'View' }).selectOption('frozen-delta');
    await q3Panel.getByRole('combobox', { name: 'Metric' }).selectOption('mse');

    const loraDelta = q3Panel.getByTestId('q3-report-delta-lora');
    const fullDelta = q3Panel.getByTestId('q3-report-delta-full');

    await expect(loraDelta.getByRole('row', { name: /H1/ })).toContainText('-0.0120');
    await expect(loraDelta.getByRole('row', { name: /H0/ })).toContainText('+0.0050');
    await expect(fullDelta.getByRole('row', { name: /H0/ })).toContainText('-0.0020');
  });

  test('keeps the control variant scoped to non-delta views', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/dashboard?tab=q3');

    const q3Panel = page.getByTestId('dashboard-q3-panel');
    const viewSelect = q3Panel.getByRole('combobox', { name: 'View' });
    const variantSelect = q3Panel.getByRole('combobox', { name: 'Variant' });

    await variantSelect.selectOption('linear_probe');

    await expect(q3Panel.getByTestId('q3-variant-scope-chip')).toHaveText('Control');
    await expect(q3Panel.getByTestId('q3-selection-helper')).toContainText(
      'Linear Probe remains visible as a control'
    );

    await viewSelect.selectOption('head-feature-matrix');
    await expect(variantSelect).toHaveValue('linear_probe');
    await expect(q3Panel.getByTestId('q3-report-matrix-view')).toBeVisible();

    await viewSelect.selectOption('frozen-delta');
    await expect(variantSelect).toBeDisabled();
    await expect(q3Panel.getByTestId('q3-selection-helper')).toContainText(
      'compares Frozen against LoRA and Full'
    );
  });

  test('shows adapted delta unavailability without blocking the other Q3 views', async ({ page }) => {
    await stubDashboardApis(page, {
      headRankingOverride: ({ variant }) => {
        if (variant !== 'lora') {
          return null;
        }
        return {
          supported: false,
          reason: 'LoRA delta data is not available for this selection.',
          heads: [],
        };
      },
    });
    await page.goto('/dashboard?tab=q3');

    const q3Panel = page.getByTestId('dashboard-q3-panel');
    const viewSelect = q3Panel.getByRole('combobox', { name: 'View' });

    await viewSelect.selectOption('frozen-delta');
    await expect(q3Panel.getByTestId('q3-report-delta-lora')).toContainText(
      'LoRA delta data is not available for this selection.'
    );
    await expect(q3Panel.getByTestId('q3-report-delta-full')).toBeVisible();

    await viewSelect.selectOption('head-ranking');
    await expect(q3Panel.getByTestId('q3-report-head-ranking-view')).toBeVisible();

    await viewSelect.selectOption('head-feature-matrix');
    await expect(q3Panel.getByTestId('q3-report-matrix-view')).toBeVisible();
    await expect(q3Panel.getByTestId('q3-report-matrix-cell-7-0')).toBeVisible();
  });

  test('restores the selected dashboard tab from the URL and falls back to Overview for invalid values', async ({ page }) => {
    await stubDashboardApis(page);

    await page.goto('/dashboard?tab=q3');

    await expect(page.getByRole('tab', { name: 'Q3' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toBeVisible();

    await page.reload();

    await expect(page.getByRole('tab', { name: 'Q3' })).toHaveAttribute('aria-selected', 'true');
    await expect(page).toHaveURL(/tab=q3/);
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toBeVisible();

    await page.goto('/dashboard?tab=not_a_real_tab');

    await expect(page.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('dashboard-main-panel')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Q3 Per-Head Specialization' })).toHaveCount(0);
  });

  test('renders the dedicated Q3 report route with the same three current views', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/q3-report?view=frozen-delta&model=clip&layer=11&metric=coverage&percentile=80');

    await expect(page.getByRole('heading', { name: 'Q3 Report View' })).toBeVisible();
    await expect(page.getByRole('combobox', { name: 'View' })).toHaveValue('frozen-delta');
    await expect(page.getByRole('combobox', { name: 'Model' })).toHaveValue('clip');
    await expect(page.getByRole('combobox', { name: 'Layer' })).toHaveValue('11');
    await expect(page.getByRole('combobox', { name: 'Metric' })).toHaveValue('coverage');
    await expect(page.getByRole('combobox', { name: 'Percentile' })).toHaveValue('80');
    await expect(page.getByTestId('q3-report-delta-view')).toBeVisible();
    await expect(page.getByTestId('q3-report-delta-lora')).toBeVisible();
    await expect(page.getByTestId('q3-report-delta-full')).toBeVisible();

    await page.getByRole('combobox', { name: 'View' }).selectOption('head-feature-matrix');
    await expect(page).toHaveURL(/view=head-feature-matrix/);
    await expect(page.getByTestId('q3-report-matrix-view')).toBeVisible();
    await page.getByTestId('q3-report-matrix-cell-7-1').click();
    await expect(page).toHaveURL(/head=1/);
    await expect(page).toHaveURL(/feature_label=7/);
    await expect(page.getByTestId('q3-report-selected-cell')).toContainText('Door');

    await page.reload();

    await expect(page).toHaveURL(/view=head-feature-matrix/);
    await expect(page).toHaveURL(/head=1/);
    await expect(page).toHaveURL(/feature_label=7/);
    await expect(page.getByTestId('q3-report-matrix-view')).toBeVisible();
  });

  test('renders /q3 as a removed route', async ({ page }) => {
    await stubDashboardApis(page);
    await page.goto('/q3?primary_model=clip&secondary_model=mae');

    await expect(page.getByText('404')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Open Dashboard' })).toHaveAttribute('href', '/dashboard');
    await expect(page.getByRole('heading', { name: 'Q3 Advanced Workspace' })).toHaveCount(0);
  });

  test('keeps the Q2 quick action context-preserving for direct /q2 navigation', async ({ page }) => {
    await stubDashboardApis(page);
    await stubQ2SummaryApi(page);

    await page.goto('/dashboard');
    await page.getByRole('combobox').first().selectOption('coverage');
    await page.getByRole('combobox').nth(1).selectOption('80');
    await page.getByTestId('leaderboard-row-clip').click();

    const quickActionsCard = page
      .getByRole('heading', { name: 'Quick Actions' })
      .locator('xpath=ancestor::div[contains(@class,"rounded")]')
      .first();
    await quickActionsCard.getByRole('link', {
      name: /Q2 Fine-Tuning Analysis/,
    }).click();

    await expect(page).toHaveURL(/\/q2\?metric=coverage&percentile=80&model=clip/);
    await expect(page.getByRole('heading', { name: 'Q2 Strategy-Aware Attention Shift' })).toBeVisible();
    await expect(page.getByText('No Q2 rows available for current filters.')).toBeVisible();
    await expect(page.getByText('Select a specific strategy to inspect per-image deltas.')).toBeVisible();
  });

  test('renders the embedded Q2 dashboard tab', async ({ page }) => {
    await stubDashboardApis(page);
    await stubQ2SummaryApi(page);

    await page.goto('/dashboard');
    await page.getByTestId('dashboard-page-tab-q2').click();

    await expect(page).toHaveURL(/\/dashboard\?tab=q2/);
    await expect(page.getByTestId('dashboard-q2-panel')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Q2 Strategy-Aware Attention Shift' })).toBeVisible();
    await expect(page.getByText('No Q2 rows available for current filters.')).toBeVisible();
    await expect(page.getByText('Select a specific model and strategy to inspect per-image deltas.')).toBeVisible();
  });
});
