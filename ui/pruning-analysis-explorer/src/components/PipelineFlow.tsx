import { Ban, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';
import { AffineEquationPanel } from './AffineEquationPanel';
import { AxisRelationView } from './AxisRelationView';
import { CompilerStageCard } from './CompilerStageCard';
import { DfaTraceView } from './DfaTraceView';
import { GraphMiniView } from './GraphMiniView';
import { PipelineStageStepper } from './PipelineStageStepper';
import type { PipelineExample, PipelineFlowResponse, ProofSummaryResponse } from '../types';

interface Props {
  flow?: PipelineFlowResponse;
  proof?: ProofSummaryResponse;
}

function ExampleCard({ example, blocked = false }: { example: PipelineExample; blocked?: boolean }) {
  return (
    <article className="transformation-card">
      <div className="transformation-heading">
        <span className={`pattern-token ${blocked ? 'blocked' : ''}`}>{example.pattern}</span>
        <h3>{example.title}</h3>
      </div>
      <GraphMiniView nodes={example.nodes} edges={example.edges} dimensions={example.dimensions} />
      <AffineEquationPanel equations={example.equations} />
      <AxisRelationView relations={example.relations} />
      <DfaTraceView facts={example.facts} compact />
    </article>
  );
}

export function PipelineFlow({ flow, proof }: Props) {
  const [activeStage, setActiveStage] = useState(0);

  if (!flow) return null;

  return (
    <>
      <section className="pipeline-hero">
        <div>
          <p className="eyebrow">Compiler pipeline</p>
          <h1>{flow.title}</h1>
          <p>{flow.summary}</p>
        </div>
        <div className="proof-badge">
          <strong>{flow.aggregate.proven_plans} / {flow.aggregate.expected_plans}</strong>
          <span>plans proven</span>
        </div>
      </section>
      <section className="pipeline-section">
        <div className="section-heading">
          <h2>Stage View</h2>
          <p>Advance through the local evidence pipeline one transformation at a time.</p>
        </div>
        <PipelineStageStepper stages={flow.stages} activeIndex={activeStage} onChange={setActiveStage} />
        <CompilerStageCard stage={flow.stages[activeStage]} />
      </section>
      <section className="pipeline-section">
        <div className="section-heading">
          <h2>Transformation Views</h2>
          <p>Reveal each worklist trace from seed fact to fixed point or blocker.</p>
        </div>
        <div className="transformation-grid">
          <ExampleCard example={flow.examples.ffn} />
          <ExampleCard example={flow.examples.attention_value} />
          <ExampleCard example={flow.examples.qk_blocker} blocked />
        </div>
      </section>
      <section className="pipeline-section">
        <div className="section-heading">
          <h2>Proof Matrix</h2>
          <p>QK contractions remain blockers and are not counted as propagation plans.</p>
        </div>
        <div className="table-scroll">
          <table className="data-table proof-table">
            <thead><tr><th>Model</th><th>FFN</th><th>Attention Value</th><th>Total</th><th>Verdict</th></tr></thead>
            <tbody>
              {proof?.models.map((model) => (
                <tr key={model.model_name}>
                  <td>{model.model_name}</td>
                  <td>{model.ffn_proven} / {model.layers}</td>
                  <td>{model.attention_value_proven} / {model.layers}</td>
                  <td>{model.proven_plans} / {model.expected_plans}</td>
                  <td><span className="badge badge-complete"><CheckCircle2 aria-hidden="true" /> complete</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted-note"><Ban aria-hidden="true" /> MLIR supplies local access evidence. The analysis framework owns pruning semantics and fixed-point propagation.</p>
      </section>
    </>
  );
}
