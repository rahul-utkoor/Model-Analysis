import { Search } from 'lucide-react';
import { useState } from 'react';
import { api } from '../api';
import type { SearchMatch } from '../types';
import { StatusBadge } from './StatusBadge';

export function SearchPanel({ selectedModel, onResult }: { selectedModel?: string; onResult: (match: SearchMatch) => void }) {
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState<SearchMatch[]>([]);
  const [busy, setBusy] = useState(false);

  async function runSearch(value = query) {
    if (!value.trim()) {
      setMatches([]);
      return;
    }
    setBusy(true);
    try {
      const result = await api.search(value, selectedModel);
      setMatches(result.matches);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="side-section">
      <h2>Search</h2>
      <div className="search-box">
        <Search size={16} />
        <input
          value={query}
          placeholder="Feed Forward, MLP, valid"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') runSearch();
          }}
        />
      </div>
      <button className="secondary-button" onClick={() => runSearch()} disabled={busy}>
        {busy ? 'Searching' : 'Search'}
      </button>
      <div className="search-results">
        {matches.slice(0, 12).map((match) => (
          <button key={`${match.model_id}-${match.layer}-${match.node_slug}`} className="search-result" onClick={() => onResult(match)}>
            <span>{match.display_name}</span>
            <small>
              {match.model} / layer {match.layer}
            </small>
            <StatusBadge value={match.pruning_class} />
          </button>
        ))}
      </div>
    </div>
  );
}
