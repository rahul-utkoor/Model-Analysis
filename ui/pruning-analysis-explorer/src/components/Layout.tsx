import type { ReactNode } from 'react';
import { SearchPanel } from './SearchPanel';
import type { ModelSummary, SearchMatch } from '../types';

interface Props {
  models: ModelSummary[];
  selectedModel?: string;
  onSelectModel: (id: string) => void;
  onSearchResult: (match: SearchMatch) => void;
  children: ReactNode;
}

export function Layout({ models, selectedModel, onSelectModel, onSearchResult, children }: Props) {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">PA</div>
          <div>
            <h1>Pruning Analysis</h1>
            <p>Static report explorer</p>
          </div>
        </div>
        <div className="side-section">
          <h2>Models</h2>
          <div className="model-list">
            {models.map((model) => (
              <button
                key={model.id}
                className={`model-button ${selectedModel === model.id ? 'active' : ''}`}
                onClick={() => onSelectModel(model.id)}
              >
                <span>{model.display_name}</span>
                <small>{model.valid_plans} valid plans</small>
              </button>
            ))}
          </div>
        </div>
        <SearchPanel selectedModel={selectedModel} onResult={onSearchResult} />
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
