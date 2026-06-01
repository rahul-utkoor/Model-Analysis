import { ArrowRight } from 'lucide-react';
import type { EvidenceTraceExample } from '../types';

interface Props {
  example: EvidenceTraceExample;
}

export function PatternMatchView({ example }: Props) {
  const blocked = example.verdict === 'blocked_as_expected';
  return (
    <section className="pattern-match-view">
      <div>
        <small>Raw graph</small>
        <div className="pattern-before">
          {example.pattern_match.before.map((item, index) => (
            <span key={`${item}-${index}`}>{item}</span>
          ))}
        </div>
      </div>
      <ArrowRight aria-hidden="true" />
      <div>
        <small>Matched pattern</small>
        <strong className={`pattern-token ${blocked ? 'blocked' : ''}`}>{example.pattern_match.after}</strong>
      </div>
      <ul className="compact-list">
        {example.pattern_match.why.map((reason) => <li key={reason}>{reason}</li>)}
      </ul>
    </section>
  );
}
