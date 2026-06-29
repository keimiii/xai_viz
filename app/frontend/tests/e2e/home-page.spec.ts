import { expect, test, type Page } from '@playwright/test';

const GALLERY_FIXTURES = [
  {
    image_id: 'Q908802_wd0.jpg',
    thumbnail_url: '/api/images/Q908802_wd0.jpg/thumbnail',
    styles: ['gothic'],
    style_names: ['Gothic'],
    num_bboxes: 2,
  },
  {
    image_id: 'Q2034923_wd0.jpg',
    thumbnail_url: '/api/images/Q2034923_wd0.jpg/thumbnail',
    styles: ['gothic'],
    style_names: ['Gothic'],
    num_bboxes: 1,
  },
  {
    image_id: 'Q18785543_wd0.jpg',
    thumbnail_url: '/api/images/Q18785543_wd0.jpg/thumbnail',
    styles: ['roman'],
    style_names: ['Romanesque'],
    num_bboxes: 3,
  },
] as const;

async function stubGalleryApis(page: Page) {
  await page.route(/\/api\/images\/styles$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(['Gothic', 'Romanesque']),
    });
  });

  await page.route(/\/api\/images(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url());
    const style = url.searchParams.get('style');
    const images = style
      ? GALLERY_FIXTURES.filter((image) => image.style_names.includes(style))
      : GALLERY_FIXTURES;

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(images),
    });
  });
}

test('filters gallery filenames live and restores results when cleared', async ({ page }) => {
  await stubGalleryApis(page);

  await page.goto('/');

  const filenameInput = page.getByLabel('Filename');

  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q2034923_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q18785543_wd0.jpg')).toBeVisible();

  await filenameInput.fill('908802');
  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q2034923_wd0.jpg')).toHaveCount(0);
  await expect(page.getByText('Q18785543_wd0.jpg')).toHaveCount(0);

  await filenameInput.fill('Q908802_WD0');
  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();

  await filenameInput.fill('missing-file');
  await expect(page.getByText('No images match the current filename and style filters.')).toBeVisible();

  await filenameInput.fill('');
  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q2034923_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q18785543_wd0.jpg')).toBeVisible();
});

test('composes filename search with the style filter', async ({ page }) => {
  await stubGalleryApis(page);

  await page.goto('/');

  const filenameInput = page.getByLabel('Filename');
  const styleSelect = page.locator('select').first();

  await styleSelect.selectOption('Gothic');
  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q2034923_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q18785543_wd0.jpg')).toHaveCount(0);

  await filenameInput.fill('Q908802');
  await expect(page.getByText('Q908802_wd0.jpg')).toBeVisible();
  await expect(page.getByText('Q2034923_wd0.jpg')).toHaveCount(0);

  await filenameInput.fill('Q18785543');
  await expect(page.getByText('No images match the current filename and style filters.')).toBeVisible();
});
