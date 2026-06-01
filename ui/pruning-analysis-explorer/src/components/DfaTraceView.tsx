import { CheckCircle2, RotateCcw } from 'lucide-react';
import { useEffect, useState } from 'react';

interface Props {
  facts: string[];
  compact?: boolean;
}

export function DfaTraceView({ facts, compact = false }: Props) {
  const [visible, setVisible] = useState(1);

  useEffect(() => setVisible(1), [facts]);

  function reveal() {
    setVisible((value) => Math.min(value + 1, facts.length));
  }

  return (
    <div className={`dfa-trace-view ${compact ? 'compact' : ''}`}>
      <div className="reveal-step-list">
        {facts.slice(0, visible).map((fact, index) => (
          <div className="reveal-step" key={fact}>
            <span>{index + 1}</span>
            <code>{fact}</code>
          </div>
        ))}
      </div>
      <button className="secondary-action" onClick={visible === facts.length ? () => setVisible(1) : reveal}>
        {visible === facts.length ? <RotateCcw aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />}
        {visible === facts.length ? 'Reset trace' : 'Reveal propagation'}
      </button>
    </div>
  );
}
