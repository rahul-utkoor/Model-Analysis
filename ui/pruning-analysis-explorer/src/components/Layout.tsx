import type { ReactNode } from 'react';
import { BarChart3, BookOpen, FileText, FlaskConical, LayoutDashboard } from 'lucide-react';
import { SearchPanel } from './SearchPanel';
import type { ModelSummary, SearchMatch } from '../types';

export type AppView = 'dashboard' | 'teaching-flow' | 'models' | 'case-studies' | 'reports';

interface Props {
  models: ModelSummary[];
  activeView: AppView;
  selectedModel?: string;
  onSelectView: (view: AppView) => void;
  onSelectModel: (id: string) => void;
  onSearchResult: (match: SearchMatch) => void;
  children: ReactNode;
}

const navigation: Array<{ id: AppView; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'teaching-flow', label: 'Teaching Flow', icon: BookOpen },
  { id: 'models', label: 'Models', icon: BarChart3 },
  { id: 'case-studies', label: 'Case Studies', icon: FlaskConical },
  { id: 'reports', label: 'Reports', icon: FileText },
];

export function Layout({ models, activeView, selectedModel, onSelectView, onSelectModel, onSearchResult, children }: Props) {
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
        <nav className="primary-nav" aria-label="Primary">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-button ${activeView === id ? 'active' : ''}`} onClick={() => onSelectView(id)}>
              <Icon aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        {activeView === 'models' ? (
          <>
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
          </>
        ) : (
          <div className="sidebar-note">Static evidence and proof reporting only. No pruning is executed.</div>
        )}
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
