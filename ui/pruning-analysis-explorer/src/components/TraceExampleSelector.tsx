import type { EvidenceTraceExample } from '../types';

interface Props {
  examples: EvidenceTraceExample[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export function TraceExampleSelector({ examples, selectedId, onSelect }: Props) {
  return (
    <div className="trace-example-selector" role="tablist" aria-label="Evidence trace examples">
      {examples.map((example) => (
        <button
          className={`trace-example-button ${selectedId === example.id ? 'active' : ''}`}
          key={example.id}
          onClick={() => onSelect(example.id)}
          role="tab"
          aria-selected={selectedId === example.id}
        >
          <strong>{example.title}</strong>
          <small>{example.pattern}</small>
        </button>
      ))}
    </div>
  );
}
