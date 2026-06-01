import { ArrowRight, BookOpen, CheckCircle2, FileText, Network } from 'lucide-react';
import type { AppView } from './Layout';
import type { OverviewResponse, ProofSummaryResponse } from '../types';

interface Props {
  overview?: OverviewResponse;
  proof?: ProofSummaryResponse;
  onSelectView: (view: AppView) => void;
}

export function OverviewDashboard({ overview, proof, onSelectView }: Props) {
  const summary = overview?.final_summary;
  return (
    <>
      <section className="overview-hero">
        <div>
          <p className="eyebrow">Compiler-style static analysis</p>
          <h1>{overview?.title ?? 'Static Pruning Propagation Analysis'}</h1>
          <p>{overview?.subtitle ?? 'From dead axes to compiler-style propagation proofs.'}</p>
          <div className="hero-actions">
            <button className="primary-action" onClick={() => onSelectView('teaching-flow')}>
              <BookOpen aria-hidden="true" /> Teaching Flow
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
        <div className="section-heading">
          <h2>Proof At A Glance</h2>
          <p>Static evidence is converted into axis relations, semantic patterns, and DFA fixed-point conclusions.</p>
        </div>
        <div className="overview-metrics">
          <div><CheckCircle2 /><strong>{summary?.native_mlir_evidence ?? 0}</strong><span>native MLIR evidence</span></div>
          <div><Network /><strong>{proof?.models.length ?? 0}</strong><span>supported models</span></div>
          <div><ArrowRight /><strong>{summary?.fallback ?? 0}</strong><span>fallback proofs</span></div>
        </div>
      </section>
      <section className="dashboard-band">
        <div className="section-heading">
          <h2>Professor Walkthrough</h2>
          <p>A concise route through the complete analysis story.</p>
        </div>
        <ol className="walkthrough-list">
          {['Teaching Flow', 'MLP/FFN example', 'Attention value-path', 'QK blocker', 'All-model proof', 'Reports'].map((item) => <li key={item}>{item}</li>)}
        </ol>
      </section>
    </>
  );
}
