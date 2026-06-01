import { ArrowLeft, ArrowRight } from 'lucide-react';
import type { PipelineStage } from '../types';

interface Props {
  stages: PipelineStage[];
  activeIndex: number;
  onChange: (index: number) => void;
}

export function PipelineStageStepper({ stages, activeIndex, onChange }: Props) {
  return (
    <div className="pipeline-stepper">
      <div className="pipeline-stepper-track">
        {stages.map((stage, index) => (
          <button className={`stepper-node ${index === activeIndex ? 'active' : ''}`} key={stage.id} onClick={() => onChange(index)}>
            <span>{index + 1}</span>
            <small>{stage.title}</small>
          </button>
        ))}
      </div>
      <div className="stepper-actions">
        <button className="secondary-action" disabled={activeIndex === 0} onClick={() => onChange(activeIndex - 1)}>
          <ArrowLeft aria-hidden="true" /> Previous
        </button>
        <button className="primary-action" disabled={activeIndex === stages.length - 1} onClick={() => onChange(activeIndex + 1)}>
          Next <ArrowRight aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
