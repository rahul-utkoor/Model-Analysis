import { ArrowDown, Ban, CheckCircle2 } from 'lucide-react';
import { ConceptCard } from './ConceptCard';
import { EvidenceHierarchy } from './EvidenceHierarchy';
import { PipelineStepCard } from './PipelineStepCard';
import type { OverviewResponse, ProofSummaryResponse, TeachingFlowResponse } from '../types';

interface Props {
  overview?: OverviewResponse;
  flow?: TeachingFlowResponse;
  proof?: ProofSummaryResponse;
}

const diagram = ['ONNX Subgraph', 'ONNX-MLIR Lowering', 'Native MLIR Dependence Evidence', 'Axis-Transfer Summary', 'Pattern Recognition', 'DFA Worklist Propagation', 'Proof Verdict'];

export function TeachingFlow({ overview, flow, proof }: Props) {
  const summary = overview?.final_summary ?? flow?.summary;
  const sectionsById = new Map(flow?.sections.map((section) => [section.id, section]));
  return (
    <>
      <section className="overview-hero teaching-hero">
        <div>
          <p className="eyebrow">Teaching Flow</p>
          <h1>Static Pruning Propagation Analysis</h1>
          <p>From dead axes to compiler-style propagation proofs.</p>
        </div>
        <div className="result-badge">
          <strong>{summary?.proven_plans ?? 0} / {summary?.expected_plans ?? 0}</strong>
          <span>propagation plans proven</span>
        </div>
      </section>
      <section className="teaching-section">
        <div className="section-heading">
          <h2>Evidence Pipeline</h2>
          <p>Selected local subgraphs become proof obligations. Each stage narrows the claim before DFA propagation.</p>
        </div>
        <div className="pipeline-diagram">
          {diagram.map((item, index) => (
            <div className="pipeline-diagram-item" key={item}>
              <div>{item}</div>
              {index < diagram.length - 1 ? <ArrowDown aria-hidden="true" /> : null}
            </div>
          ))}
        </div>
      </section>
      <section className="teaching-section">
        <div className="section-heading">
          <h2>Core Conceptual Split</h2>
          <p>Sparsity is not the same as deadness.</p>
        </div>
        <div className="concept-grid">
          <ConceptCard title="Sparse-weight pruning" eyebrow="Shape preserving">
            <ul className="compact-list">
              <li>Creates zeros in existing tensors.</li>
              <li>Keeps tensor shapes unchanged.</li>
              <li>Does not necessarily create dead channels.</li>
            </ul>
          </ConceptCard>
          <ConceptCard title="Structural pruning" eyebrow="Compiler visible">
            <ul className="compact-list">
              <li>Removes or makes whole axes dead.</li>
              <li>Requires propagation and coordinated repair.</li>
              <li>Creates graph-transformation obligations.</li>
            </ul>
          </ConceptCard>
        </div>
      </section>
      <section className="teaching-section">
        <div className="section-heading">
          <h2>Propagation Examples</h2>
          <p>Axis-transfer evidence selects a rule; the worklist computes the fixed point.</p>
        </div>
        <div className="example-grid">
          <ConceptCard title="MLP / FFN" eyebrow="FFN_INTERMEDIATE_CHAIN">
            <div className="code-flow">{'hidden -> intermediate -> intermediate -> hidden'}</div>
            <p>{'op3 input[j] DEAD -> op2 output/input[j] DEAD -> op1 output[j] DEAD'}</p>
          </ConceptCard>
          <ConceptCard title="Attention value path" eyebrow="ATTENTION_VALUE_PATH">
            <div className="code-flow">{'value projection -> attention context -> output projection'}</div>
            <p>{'out projection input[d] DEAD -> context value axis[d] DEAD -> value projection output[d] DEAD'}</p>
          </ConceptCard>
          <ConceptCard title="QK blocker" eyebrow="QK_SCORE_BLOCKER">
            <div className="code-flow">Score[q,k] += Q[q,d] * K[k,d]</div>
            <p><Ban aria-hidden="true" /> d is reduced and mixed, so simple Q/K propagation is BLOCKED.</p>
          </ConceptCard>
        </div>
      </section>
      {summary ? <EvidenceHierarchy summary={summary} /> : null}
      <section className="teaching-section">
        <div className="section-heading">
          <h2>All-Model Proof</h2>
          <p>FFN and attention value-path plans are complete for the five supported models.</p>
        </div>
        <div className="table-scroll">
          <table className="data-table proof-table">
            <thead><tr><th>Model</th><th>FFN</th><th>Attention Value Path</th><th>Total</th><th>Verdict</th></tr></thead>
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
      </section>
      <section className="teaching-section">
        <div className="section-heading">
          <h2>Guided Detail</h2>
          <p>Expand a step to review the claim made at that stage.</p>
        </div>
        <div className="teaching-step-list">
          {overview?.pipeline_steps.map((step, index) => <PipelineStepCard key={step.id} step={step} index={index} />)}
        </div>
      </section>
      <section className="teaching-section limitations">
        <div className="section-heading">
          <h2>Limitations</h2>
          <p>Static proof reporting has explicit scope boundaries.</p>
        </div>
        <ul className="compact-list">
          {sectionsById.get('limits')?.points.map((point) => <li key={point}>{point}</li>)}
          <li>MLIR is local evidence, not the pruning framework itself.</li>
        </ul>
      </section>
    </>
  );
}
