import { expect, test, type Page } from '@playwright/test';

const IMAGE_ID = 'Q2034923_wd0.jpg';
const TYPED_IMAGE_ID = 'Q908802_wd0.jpg';

function getModelResult(
  payload: {
    results: Array<{ model: string; iou: number; mse: number }>;
  },
  model: string,
) {
  const result = payload.results.find((entry) => entry.model === model);
  expect(result).toBeTruthy();
  return result!;
}

function getSelectByLabel(page: Page, label: string) {
  return page
    .locator('div.flex.flex-col')
    .filter({ has: page.getByText(label, { exact: true }) })
    .locator('select')
    .first();
}

function getCompareImageInput(page: Page) {
  return page.locator('#compare-image-input');
}

async function clickBbox(page: Page, index: number) {
  await page.getByTestId(`bbox-hitbox-${index}`).first().evaluate((element) => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
}

async function stubVariantCompareApis(page: import('@playwright/test').Page) {
  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          image_id: IMAGE_ID,
          thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
        },
      ]),
    });
  });

  await page.route(`**/api/images/${IMAGE_ID}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        image_url: `/api/images/${IMAGE_ID}/file`,
        thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
        available_models: ['dinov2'],
        annotation: {
          image_id: IMAGE_ID,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
          bboxes: [],
        },
      }),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12 },
        methods: { dinov2: ['cls'] },
        default_methods: { dinov2: 'cls' },
      }),
    });
  });

  await page.route('**/api/compare/variants?**', async (route) => {
    const url = new URL(route.request().url());
    const model = url.searchParams.get('model') ?? 'dinov2';
    const leftVariant = url.searchParams.get('left_variant') ?? 'frozen';
    const rightVariant = url.searchParams.get('right_variant') ?? 'lora';
    const variantLabels: Record<string, string> = {
      frozen: 'Frozen (Pretrained)',
      linear_probe: 'Linear Probe',
      lora: 'LoRA',
      full: 'Full Fine-tune',
    };
    const buildVariantPayload = (variant: string) => ({
      model_key: variant === 'frozen' ? model : `${model}_finetuned_${variant}`,
      strategy: variant === 'frozen' ? null : variant,
      label: variantLabels[variant] ?? variant,
      available: true,
      url: `/api/attention/${IMAGE_ID}/overlay?model=${variant === 'frozen' ? model : `${model}_finetuned_${variant}`}&layer=0&method=cls`,
    });

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        model,
        layer: 'layer0',
        method: 'cls',
        show_bboxes: true,
        left: buildVariantPayload(leftVariant),
        right: buildVariantPayload(rightVariant),
        note: 'ok',
      }),
    });
  });

  await page.route('**/api/metrics/q2_summary?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        metric: 'mse',
        label: 'MSE',
        direction: 'lower',
        percentile_dependent: false,
        selected_percentile: null,
        analyzed_layer: 11,
        timestamp: null,
        rows: [],
        strategy_comparisons: [],
      }),
    });
  });
}

async function stubVariantShiftApi(
  page: import('@playwright/test').Page,
  options?: { available?: boolean; reason?: string }
) {
  await page.route('**/api/compare/variants/shift?**', async (route) => {
    const url = new URL(route.request().url());
    const model = url.searchParams.get('model') ?? 'dinov2';
    const comparedVariant = (url.searchParams.get('compared_variant') ?? 'lora') as 'linear_probe' | 'lora' | 'full';
    const variantLabels: Record<string, string> = {
      linear_probe: 'Linear Probe',
      lora: 'LoRA',
      full: 'Full Fine-tune',
    };
    const available = options?.available ?? true;

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        model,
        layer: 'layer0',
        method: 'cls',
        available,
        reason: available ? null : (options?.reason ?? 'Compared variant attention is not cached for this selection.'),
        baseline_variant: 'frozen',
        compared_variant: comparedVariant,
        baseline_model_key: model,
        compared_model_key: `${model}_finetuned_${comparedVariant}`,
        operation: 'compared_variant_attention - frozen_attention',
        shape: available ? [2, 2] : [],
        shift: available ? [0.25, -0.1, 0.0, 0.45] : [],
        min_value: available ? -0.1 : null,
        max_value: available ? 0.45 : null,
        max_abs_value: available ? 0.45 : null,
        label: variantLabels[comparedVariant],
      }),
    });
  });
}

async function stubVariantCompareLandingApis(page: import('@playwright/test').Page) {
  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'siglip2', 'clip'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12, siglip2: 12, clip: 12 },
        methods: { dinov2: ['cls'], siglip2: ['mean'], clip: ['cls'] },
        default_methods: { dinov2: 'cls', siglip2: 'mean', clip: 'cls' },
      }),
    });
  });
}

async function stubCompareControlApis(page: Page) {
  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          image_id: IMAGE_ID,
          thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
        },
      ]),
    });
  });

  await page.route(`**/api/images/${IMAGE_ID}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        image_url: `/api/images/${IMAGE_ID}/file`,
        thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
        available_models: ['dinov2', 'clip'],
        annotation: {
          image_id: IMAGE_ID,
          styles: ['gothic'],
          style_names: ['Gothic'],
          num_bboxes: 1,
          bboxes: [],
        },
      }),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'clip'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12, clip: 12 },
        methods: { dinov2: ['cls'], clip: ['cls'] },
        default_methods: { dinov2: 'cls', clip: 'cls' },
      }),
    });
  });

  await page.route('**/api/compare/models?**', async (route) => {
    const url = new URL(route.request().url());
    const models = url.searchParams.getAll('models');
    const method = url.searchParams.get('method') ?? 'cls';
    const results = models.map((model, index) => ({
      image_id: IMAGE_ID,
      model,
      layer: 'layer0',
      percentile: 90,
      iou: index === 0 ? 0.12 : 0.18,
      coverage: index === 0 ? 0.31 : 0.34,
      mse: index === 0 ? 0.0112 : 0.0104,
      kl: index === 0 ? 4.21 : 3.88,
      emd: index === 0 ? 0.29 : 0.24,
      attention_area: 0.12,
      annotation_area: 0.09,
      method,
    }));

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        models,
        layer: 'layer0',
        percentile: 90,
        selection: {
          mode: 'union',
          bbox_index: null,
          bbox_label: null,
        },
        results,
        heatmap_urls: {},
        unavailable_models: {},
      }),
    });
  });

  await page.route('**/api/compare/variants?**', async (route) => {
    const url = new URL(route.request().url());
    const model = url.searchParams.get('model') ?? 'dinov2';
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: IMAGE_ID,
        model,
        layer: 'layer0',
        method: 'cls',
        show_bboxes: true,
        left: {
          model_key: model,
          strategy: null,
          label: 'Frozen (Pretrained)',
          available: true,
          url: `/api/attention/${IMAGE_ID}/overlay?model=${model}&layer=0&method=cls`,
        },
        right: {
          model_key: `${model}_finetuned_full`,
          strategy: 'full',
          label: 'Full Fine-tune',
          available: true,
          url: `/api/attention/${IMAGE_ID}/overlay?model=${model}_finetuned_full&layer=0&method=cls`,
        },
        note: 'ok',
      }),
    });
  });
}

async function stubCompareImageSelectorApis(page: Page) {
  const galleryImages = [
    {
      image_id: IMAGE_ID,
      thumbnail_url: `/api/images/${IMAGE_ID}/thumbnail`,
      styles: ['gothic'],
      style_names: ['Gothic'],
      num_bboxes: 1,
    },
    {
      image_id: TYPED_IMAGE_ID,
      thumbnail_url: `/api/images/${TYPED_IMAGE_ID}/thumbnail`,
      styles: ['roman'],
      style_names: ['Romanesque'],
      num_bboxes: 2,
    },
  ];

  await page.route('**/api/images?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(galleryImages),
    });
  });

  await page.route(/\/api\/images\/[^/]+\.jpg$/, async (route) => {
    const url = new URL(route.request().url());
    const requestedImageId = decodeURIComponent(url.pathname.split('/').pop() ?? '');
    const matchedImage = galleryImages.find((image) => image.image_id === requestedImageId);

    if (!matchedImage) {
      await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'Not found' }) });
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: matchedImage.image_id,
        image_url: `/api/images/${matchedImage.image_id}/file`,
        thumbnail_url: matchedImage.thumbnail_url,
        available_models: ['dinov2', 'clip'],
        annotation: {
          image_id: matchedImage.image_id,
          styles: matchedImage.styles,
          style_names: matchedImage.style_names,
          num_bboxes: matchedImage.num_bboxes,
          bboxes: [],
        },
      }),
    });
  });

  await page.route('**/api/attention/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: ['dinov2', 'clip'],
        num_layers: 12,
        num_layers_per_model: { dinov2: 12, clip: 12 },
        methods: { dinov2: ['cls'], clip: ['cls'] },
        default_methods: { dinov2: 'cls', clip: 'cls' },
      }),
    });
  });

  await page.route('**/api/compare/models?**', async (route) => {
    const url = new URL(route.request().url());
    const models = url.searchParams.getAll('models');
    const selectedImageId = url.searchParams.get('image') ?? IMAGE_ID;
    const method = url.searchParams.get('method') ?? 'cls';

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image_id: selectedImageId,
        models,
        layer: 'layer0',
        percentile: 90,
        selection: {
          mode: 'union',
          bbox_index: null,
          bbox_label: null,
        },
        results: models.map((model, index) => ({
          image_id: selectedImageId,
          model,
          layer: 'layer0',
          percentile: 90,
          iou: index === 0 ? 0.12 : 0.18,
          coverage: index === 0 ? 0.31 : 0.34,
          mse: index === 0 ? 0.0112 : 0.0104,
          kl: index === 0 ? 4.21 : 3.88,
          emd: index === 0 ? 0.29 : 0.24,
          attention_area: 0.12,
          annotation_area: 0.09,
          method,
        })),
        heatmap_urls: {},
        unavailable_models: {},
      }),
    });
  });
}

test.describe('Compare page', () => {
  test('removes the View Details CTA and keeps the right controls when switching modes', async ({ page }) => {
    await stubCompareControlApis(page);

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=models`);

    await expect(page.getByRole('heading', { name: 'Model Comparison' })).toBeVisible();
    await expect(page.getByRole('link', { name: /View Details/i })).toHaveCount(0);
    await expect(page.getByText('Left Model', { exact: true })).toBeVisible();
    await expect(page.getByText('Right Model', { exact: true })).toBeVisible();
    await expect(page.getByTestId('comparison-method-context')).toContainText(
      'Comparing with shared method: cls'
    );

    await getSelectByLabel(page, 'Comparison Type').selectOption('variants');

    await expect(page).toHaveURL(/type=variants/);
    await expect(page.getByText('Model', { exact: true })).toBeVisible();
    await expect(page.getByText('Metric', { exact: true })).toBeVisible();
    await expect(page.getByText('Left Variant', { exact: true })).toBeVisible();
    await expect(page.getByText('Right Variant', { exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: /View Details/i })).toHaveCount(0);
  });

  test('lets users type an exact filename from the combined image picker', async ({ page }) => {
    await stubCompareImageSelectorApis(page);

    await page.goto('/compare?type=models');

    const imageInput = getCompareImageInput(page);
    const imageSuggestions = page.locator('#compare-image-options option');

    await expect(imageInput).toHaveAttribute('list', 'compare-image-options');
    await expect(imageSuggestions).toHaveCount(2);
    await expect(imageSuggestions.nth(1)).toHaveAttribute('value', TYPED_IMAGE_ID);

    await imageInput.fill('Q908802');
    await expect(page).not.toHaveURL(/image=/);

    await imageInput.fill(TYPED_IMAGE_ID);
    await expect(page).toHaveURL(/image=Q908802_wd0\.jpg/);
    await expect(imageInput).toHaveValue(TYPED_IMAGE_ID);

    await imageInput.fill(IMAGE_ID);
    await expect(page).toHaveURL(/image=Q2034923_wd0\.jpg/);
    await expect(imageInput).toHaveValue(IMAGE_ID);
  });

  test('preserves a shared attention method when both selected models support it', async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
    await page.getByRole('combobox').nth(1).selectOption('rollout');

    await page.getByRole('link', { name: 'Compare' }).click();

    const comparisonResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && response.url().includes('method=rollout')
        && response.status() === 200
    );
    await getCompareImageInput(page).fill(IMAGE_ID);
    await comparisonResponse;

    await expect(page.getByTestId('comparison-method-context')).toContainText(
      'Comparing with shared method: rollout'
    );
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*rollout$/)
    ).toHaveCount(2);
  });

  test('clamps the shared layer when switching to lower-depth model pairs', async ({ page }) => {
    await page.goto(`/image/${encodeURIComponent(IMAGE_ID)}`);
    await page.getByTestId('layer-last').click();
    await expect(page.getByTestId('active-layer-indicator')).toContainText('Focused: Layer 11');

    await page.getByRole('link', { name: 'Compare' }).click();
    await getCompareImageInput(page).fill(IMAGE_ID);

    await expect(page.getByText('Model Comparison')).toBeVisible();
    await getSelectByLabel(page, 'Left Model').selectOption('siglip');
    await getSelectByLabel(page, 'Right Model').selectOption('resnet50');

    await expect(page.getByText('Failed to load comparison')).toHaveCount(0);
    await expect(getSelectByLabel(page, 'Left Model')).toHaveValue('siglip');
    await expect(getSelectByLabel(page, 'Right Model')).toHaveValue('resnet50');
    await expect(page.getByTestId('comparison-method-context')).toContainText(
      "Using each model's default attention method because cls is not shared by both selected models."
    );
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*mean$/)
    ).toHaveCount(1);
    await expect(
      page.locator('.grid.grid-cols-2').getByText(/^Method:\s*gradcam$/)
    ).toHaveCount(1);
    await expect(page.getByText(/KL:/)).toHaveCount(2);
    await expect(page.getByText(/EMD:/)).toHaveCount(2);
  });

  test('switches compare metrics to bbox scope when a bounding box is selected', async ({ page }) => {
    const initialCompareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && !response.url().includes('bbox_index=')
        && response.status() === 200
    );

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=models`);
    const initialPayload = await (await initialCompareResponse).json();

    const leftModel = await getSelectByLabel(page, 'Left Model').inputValue();
    const initialLeft = getModelResult(initialPayload, leftModel);

    const bboxCompareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/models')
        && response.url().includes('bbox_index=0')
        && response.status() === 200
    );

    await clickBbox(page, 0);
    const bboxPayload = await (await bboxCompareResponse).json();
    const bboxLeft = getModelResult(bboxPayload, leftModel);

    expect(bboxPayload.selection.mode).toBe('bbox');
    expect(bboxPayload.selection.bbox_index).toBe(0);
    expect(bboxLeft.iou).not.toBe(initialLeft.iou);
    expect(bboxLeft.mse).not.toBe(initialLeft.mse);

    const leftMetrics = page.getByTestId('comparison-metrics-left');
    await expect(leftMetrics).toContainText('Feature-level metrics');
    await expect(leftMetrics).toContainText(bboxPayload.selection.bbox_label);
    await expect(leftMetrics).toContainText(`IoU: ${bboxLeft.iou.toFixed(3)}`);
    await expect(leftMetrics).toContainText(`MSE: ${bboxLeft.mse.toFixed(4)}`);
  });

  test('shows per-model unavailable messaging instead of stale whole-image metrics', async ({ page }) => {
    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=models`);
    await expect(page.getByTestId('comparison-metrics-left')).toContainText('Whole-image metrics');

    const leftModel = await getSelectByLabel(page, 'Left Model').inputValue();
    const rightModel = await getSelectByLabel(page, 'Right Model').inputValue();
    const rightMetrics = page.getByTestId('comparison-metrics-right');
    const previousRightText = (await rightMetrics.textContent()) ?? '';

    await page.route('**/api/compare/models?**', async (route) => {
      const url = new URL(route.request().url());
      if (url.searchParams.get('bbox_index') !== '0') {
        await route.continue();
        return;
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          image_id: IMAGE_ID,
          models: [leftModel, rightModel],
          layer: 'layer0',
          percentile: 90,
          selection: {
            mode: 'bbox',
            bbox_index: 0,
            bbox_label: 'Window',
          },
          results: [
            {
              image_id: IMAGE_ID,
              model: leftModel,
              layer: 'layer0',
              percentile: 90,
              iou: 0.222,
              coverage: 0.444,
              mse: 0.0123,
              kl: 0.0456,
              emd: 0.0678,
              attention_area: 0.12,
              annotation_area: 0.09,
              method: 'cls',
            },
          ],
          heatmap_urls: {},
          unavailable_models: {
            [rightModel]: `Feature-level metrics unavailable because cached attention is missing for ${rightModel}/layer0/cls/${IMAGE_ID}.`,
          },
        }),
      });
    });

    await clickBbox(page, 0);

    await expect(page.getByTestId('comparison-metrics-right-unavailable')).toContainText(
      'Feature-level metrics unavailable because cached attention is missing'
    );
    await expect(rightMetrics).toContainText('Feature-level metrics');
    await expect(rightMetrics).not.toContainText('Whole-image metrics');
    await expect(rightMetrics).not.toContainText(previousRightText.trim());
  });

  test('normalizes legacy fine-tuning URLs into variant compare and disables percentile for threshold-free metrics', async ({ page }) => {
    await stubVariantCompareApis(page);

    const compareResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/compare/variants')
        && response.url().includes('left_variant=frozen')
        && response.url().includes('right_variant=lora')
        && response.status() === 200
    );

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=frozen&model=dinov2&strategy=lora&metric=mse&percentile=90`);
    await compareResponse;

    await expect(page).toHaveURL(/type=variants/);
    await expect(page).toHaveURL(/left_variant=frozen/);
    await expect(page).toHaveURL(/right_variant=lora/);
    await expect(page.getByText(/threshold-free, so percentile stays visible/)).toBeVisible();
    await expect(getSelectByLabel(page, 'Percentile')).toBeDisabled();
  });

  test('shows the shift-map view for frozen-vs-adapted pairs and hides it for adapted-only pairs', async ({ page }) => {
    await stubVariantCompareApis(page);
    await stubVariantShiftApi(page);

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=variants&model=dinov2&left_variant=frozen&right_variant=lora`);

    const shiftButton = page.getByRole('button', { name: 'Shift map' });
    await expect(shiftButton).toBeVisible();
    await shiftButton.click();

    await expect(page.getByText(/This map always shows LoRA minus Frozen/)).toBeVisible();
    await expect(page.getByText(/Red means more attention after fine-tuning, blue means less/)).toBeVisible();
    await expect(page.getByText(/photo background is shown in grayscale and dimmed/)).toBeVisible();
    await expect(page.getByAltText(/shift map/i)).toHaveClass(/grayscale/);
    await expect(page.getByAltText(/shift map/i)).toHaveClass(/brightness-50/);

    await getSelectByLabel(page, 'Left Variant').selectOption('linear_probe');
    await getSelectByLabel(page, 'Right Variant').selectOption('full');

    await expect(page.getByRole('button', { name: 'Shift map' })).toHaveCount(0);
  });

  test('shows shift-specific unavailable messaging without failing the compare page', async ({ page }) => {
    await stubVariantCompareApis(page);
    await stubVariantShiftApi(page, {
      available: false,
      reason: 'Compared variant attention is not cached for this model/layer/image. Generate fine-tuned attention caches for the requested strategy first.',
    });

    await page.goto(`/compare?image=${encodeURIComponent(IMAGE_ID)}&type=variants&model=dinov2&left_variant=frozen&right_variant=full`);

    await page.getByRole('button', { name: 'Shift map' }).click();

    await expect(page.getByText('Attention shift unavailable')).toBeVisible();
    await expect(page.getByText(/Generate fine-tuned attention caches for the requested strategy first/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Side by side' })).toBeVisible();
  });

  test('keeps the Q2-selected model available before an image is chosen', async ({ page }) => {
    await stubVariantCompareLandingApis(page);

    await page.goto('/compare?type=variants&model=siglip2&metric=emd&left_variant=frozen&right_variant=full');

    const modelSelect = getSelectByLabel(page, 'Model');
    await expect(modelSelect).toHaveValue('siglip2');
    await expect(modelSelect.locator('option')).toContainText(['dinov2', 'siglip2', 'clip']);
    await expect(page.getByText('Select an image above to start comparing')).toBeVisible();
  });
});
