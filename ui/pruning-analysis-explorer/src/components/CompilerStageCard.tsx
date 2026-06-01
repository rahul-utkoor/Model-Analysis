import { AffineEquationPanel } from './AffineEquationPanel';
import { AxisRelationView } from './AxisRelationView';
import { DfaTraceView } from './DfaTraceView';
import { GraphMiniView } from './GraphMiniView';
import type { PipelineStage } from '../types';

interface Props {
  stage: PipelineStage;
}

export function CompilerStageCard({ stage }: Props) {
  return (
    <article className="stage-card">
      <div className="stage-card-heading">
        <span className="stage-kind">{stage.kind}</span>
        <h2>{stage.title}</h2>
        <p>{stage.short}</p>
      </div>
      {stage.visual ? <GraphMiniView nodes={stage.visual.nodes} edges={stage.visual.edges} /> : null}
      {stage.equations ? <AffineEquationPanel equations={stage.equations} /> : null}
      {stage.relations ? <AxisRelationView relations={stage.relations} /> : null}
      {stage.pattern ? <div className="pattern-token">{stage.pattern}</div> : null}
      {stage.facts ? <DfaTraceView facts={stage.facts} /> : null}
      <div className="stage-claims">
        <div><small>What is proven?</small><p>{stage.proven}</p></div>
        {stage.not_claimed ? <div><small>What is not claimed?</small><p>{stage.not_claimed}</p></div> : null}
      </div>
    </article>
  );
}
