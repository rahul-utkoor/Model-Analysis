import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { PipelineStep } from '../types';

interface Props {
  step: PipelineStep;
  index: number;
}

export function PipelineStepCard({ step, index }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className={`teaching-step ${expanded ? 'expanded' : ''}`}>
      <button className="teaching-step-button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        <span className="step-number">{index + 1}</span>
        <span>
          <strong>{step.title}</strong>
          <small>{step.summary}</small>
        </span>
        {expanded ? <ChevronUp aria-hidden="true" /> : <ChevronDown aria-hidden="true" />}
      </button>
      {expanded ? (
        <ul className="compact-list">
          {step.details.map((detail) => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
