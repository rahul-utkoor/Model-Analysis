import { ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react';
import type { TraceStep } from '../types';

interface Props {
  steps: TraceStep[];
  activeIndex: number;
  onChange: (index: number) => void;
}

export function TraceTimeline({ steps, activeIndex, onChange }: Props) {
  return (
    <section className="trace-timeline">
      <div className="trace-step-list">
        {steps.map((step, index) => (
          <button className={`trace-step trace-step-${step.kind} ${activeIndex === index ? 'trace-step-active' : ''}`} key={`${step.kind}-${step.fact}`} onClick={() => onChange(index)}>
            <span>{index + 1}</span>
            <small>{step.kind.replace('_', ' ')}</small>
            <code>{step.fact}</code>
          </button>
        ))}
      </div>
      <div className="trace-controls">
        <button className="secondary-action" disabled={activeIndex === 0} onClick={() => onChange(activeIndex - 1)}><ArrowLeft aria-hidden="true" /> Previous</button>
        <button className="secondary-action" onClick={() => onChange(0)}><RotateCcw aria-hidden="true" /> Reset</button>
        <button className="primary-action" disabled={activeIndex === steps.length - 1} onClick={() => onChange(activeIndex + 1)}>Next <ArrowRight aria-hidden="true" /></button>
      </div>
    </section>
  );
}
