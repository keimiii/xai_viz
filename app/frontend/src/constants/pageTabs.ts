import type { PageTab } from '../types';

export const DEFAULT_PAGE_TAB: PageTab = 'main';

export function parsePageTab(value: string | null | undefined): PageTab {
  if (value === 'q2') return 'q2';
  if (value === 'q3') return 'q3';
  return DEFAULT_PAGE_TAB;
}
