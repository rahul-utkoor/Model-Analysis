import { ArrowRight, Ban, CheckCircle2, FileText } from 'lucide-react';
import { GraphMiniView } from './GraphMiniView';
import type { AppView } from './Layout';
import type { OverviewResponse, PipelineFlowResponse, ProofSummaryResponse } from '../types';

interface Props {
  overview?: OverviewResponse;
  flow?: PipelineFlowResponse;
  proof?: ProofSummaryResponse;
  onSelectView: (view: AppView) => void;
}

export function OverviewDashboard({ overview, flow, proof, onSelectView }: Props) {
  const summary = overview?.final_summary;
  return (
    <>
      <section className="overview-hero">
        <div>
          <p className="eyebrow">Compiler-style static analysis</p>
          <h1>{overview?.title ?? 'Static Pruning Propagation Analysis'}</h1>
          <p>{overview?.subtitle ?? 'From dead axes to compiler-style propagation proofs.'}</p>
          <div className="hero-actions">
            <button className="primary-action" onClick={() => onSelectView('pipeline-flow')}>
              <ArrowRight aria-hidden="true" /> Pipeline Flow
            </button>
            <button className="secondary-action" onClick={() => onSelectView('reports')}>
              <FileText aria-hidden="true" /> Reports
            </button>
          </div>
        </div>
        <div className="result-badge">
          <strong>{summary?.proven_plans ?? 0} / {summary?.expected_plans ?? 0}</strong>
          <span>propagation plans proven</span>
        </div>
      </section>
      <section className="dashboard-band">
        <div className="overview-metrics">
          <div><CheckCircle2 /><strong>{summary?.native_mlir_evidence ?? 0}</strong><span>native MLIR evidence</span></div>
          <div><ArrowRight /><strong>{summary?.fallback ?? 0}</strong><span>fallback proofs</span></div>
        </div>
      </section>
      <section className="dashboard-band">
        <div className="section-heading">
          <h2>Core Transformations</h2>
        </div>
        <div className="dashboard-example-grid">
          {flow ? (
            <>
              <article className="dashboard-example"><strong>FFN propagation</strong><GraphMiniView nodes={flow.examples.ffn.nodes} edges={flow.examples.ffn.edges} /></article>
              <article className="dashboard-example"><strong>Attention value path</strong><GraphMiniView nodes={flow.examples.attention_value.nodes} edges={flow.examples.attention_value.edges} /></article>
              <article className="dashboard-example"><strong><Ban aria-hidden="true" /> QK blocker</strong><GraphMiniView nodes={flow.examples.qk_blocker.nodes} edges={flow.examples.qk_blocker.edges} /></article>
            </>
          ) : null}
        </div>
      </section>
      <section className="dashboard-band">
        <div className="section-heading"><h2>Model Proofs</h2></div>
        <div className="table-scroll">
          <table className="data-table proof-table">
            <thead><tr><th>Model</th><th>FFN</th><th>Attention Value</th><th>Total</th></tr></thead>
            <tbody>{proof?.models.map((model) => <tr key={model.model_name}><td>{model.model_name}</td><td>{model.ffn_proven} / {model.layers}</td><td>{model.attention_value_proven} / {model.layers}</td><td>{model.proven_plans} / {model.expected_plans}</td></tr>)}</tbody>
          </table>
        </div>
      </section>
    </>
  );
}
