import type { PageTab } from '../../types';

interface PageTabOption {
  value: PageTab;
  label: string;
  id?: string;
  panelId?: string;
  dataTestId?: string;
}

interface PageTabsProps {
  label: string;
  tabs: PageTabOption[];
  activeTab: PageTab;
  onChange: (tab: PageTab) => void;
  className?: string;
}

export function PageTabs({
  label,
  tabs,
  activeTab,
  onChange,
  className = '',
}: PageTabsProps) {
  return (
    <div
      role="tablist"
      aria-label={label}
      className={`inline-flex w-full max-w-max rounded-xl border border-slate-200 bg-white p-1 shadow-sm ${className}`}
    >
      {tabs.map((tab) => {
        const isSelected = activeTab === tab.value;

        return (
          <button
            key={tab.value}
            id={tab.id}
            type="button"
            role="tab"
            aria-selected={isSelected}
            aria-controls={tab.panelId}
            tabIndex={isSelected ? 0 : -1}
            data-testid={tab.dataTestId}
            onClick={() => onChange(tab.value)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              isSelected
                ? 'bg-primary-600 text-white shadow-sm'
                : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
