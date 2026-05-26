import type { ModelDetail } from '../types';

const stages = [
  ['Op Semantics', 'Primitive TensorOps are annotated as projections, contractions, residuals, LayerNorms, activations, or metadata flow.'],
  ['Region Pruning Semantics', 'Abstract regions receive pruning roles, repair obligations, blockers, and protected dimensions.'],
  ['Opportunity Ranking', 'Regions are ranked as safe, constrained, blocked, auxiliary, or unknown with reasons.'],
  ['Plan Synthesis', 'Safe MLP/FFN candidates become symbolic plans over an intermediate_dim index set.'],
  ['Plan Validation', 'Plans are checked for required actions, semantic agreement, hidden preservation, residual and LayerNorm protection.'],
  ['Layer/Subgraph Atlas', 'Reports project the full-model analysis into ordered learner-facing subgraphs and ONNX evidence.']
];

export function PipelineOverview({ detail }: { detail?: ModelDetail }) {
  const summary = detail?.model_summary ?? {};
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>Pipeline Overview</h2>
        <p>How the selected model is lowered from local op facts into validated symbolic pruning plans.</p>
      </div>
      <div className="pipeline-list">
        {stages.map(([title, body], index) => (
          <div className="pipeline-step" key={title}>
            <span>{index + 1}</span>
            <div>
              <h3>{title}</h3>
              <p>{body}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="pipeline-counts">
        <span>{summary.ranking?.safe ?? 0} safe candidates</span>
        <span>{summary.plans?.ready_symbolic ?? 0} ready plans</span>
        <span>{summary.plan_validation?.valid ?? 0} valid plans</span>
      </div>
    </section>
  );
}
