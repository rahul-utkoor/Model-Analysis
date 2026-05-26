import { Download, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import type { SubgraphDetailResponse } from '../types';
import { EvidencePanel } from './EvidencePanel';
import { MarkdownBlock } from './MarkdownBlock';
import { PlanPanel } from './PlanPanel';
import { StatusBadge } from './StatusBadge';
import { ValidationPanel } from './ValidationPanel';

const tabs = ['Summary', 'Explanation', 'Primitive Ops', 'Op Semantics', 'Region Semantics', 'Ranking', 'Plan', 'Validation', 'Artifacts'];

export function SubgraphDetail({ detail }: { detail?: SubgraphDetailResponse }) {
  const [tab, setTab] = useState('Summary');
  if (!detail) {
    return (
      <section className="panel detail-panel empty-detail">
        <h2>Select a subgraph</h2>
        <p className="muted">Choose an abstract node to inspect semantics, ranking, plan, validation, and artifacts.</p>
      </section>
    );
  }
  const analysis = detail.analysis ?? {};
  const cls = analysis.classification ?? {};
  return (
    <section className="panel detail-panel">
      <div className="detail-header">
        <div>
          <h2>{analysis.display_name}</h2>
          <p>{analysis.semantic_category}</p>
        </div>
        <div className="badge-row">
          <StatusBadge value={cls.pruning_class} />
          <StatusBadge value={cls.plan_status} />
          <StatusBadge value={cls.validation_status} />
        </div>
      </div>
      <div className="tabs">
        {tabs.map((item) => (
          <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>
            {item}
          </button>
        ))}
      </div>
      <div className="tab-body">{renderTab(tab, detail)}</div>
    </section>
  );
}

function renderTab(tab: string, detail: SubgraphDetailResponse) {
  const analysis = detail.analysis ?? {};
  if (tab === 'Summary') {
    return (
      <div className="summary-tab">
        <p className="verdict">{analysis.verdict ?? analysis.explanation}</p>
        <div className="metric-grid compact">
          <Small label="Primitive ops" value={analysis.primitive_ops?.length ?? 0} />
          <Small label="Ranking rows" value={analysis.local_ranking?.length ?? 0} />
          <Small label="Plans" value={analysis.local_plans?.length ?? 0} />
          <Small label="Validations" value={analysis.local_validations?.length ?? 0} />
        </div>
        <p><strong>Why no plan:</strong> {analysis.why_no_plan || 'not applicable'}</p>
      </div>
    );
  }
  if (tab === 'Explanation') return <MarkdownBlock text={detail.explanation_md} />;
  if (tab === 'Primitive Ops') return <EvidencePanel title="Primitive TensorIR / ONNX ops" rows={analysis.primitive_ops ?? []} columns={['topological_index', 'source_name', 'op_type']} />;
  if (tab === 'Op Semantics') return <EvidencePanel title="Op semantics" rows={analysis.local_op_semantics ?? []} columns={['source_name', 'semantic_kind', 'semantic_category', 'parameterized', 'direct_pruning']} />;
  if (tab === 'Region Semantics') return <EvidencePanel title="Region semantics" rows={analysis.local_region_semantics ?? []} columns={['region_name', 'source_region_type', 'semantic_category', 'pruning_role', 'blockers', 'repairs']} />;
  if (tab === 'Ranking') return <EvidencePanel title="Ranking evidence" rows={analysis.local_ranking ?? []} columns={['candidate_kind', 'pruning_class', 'rank_score', 'confidence', 'target_dimension', 'reason']} />;
  if (tab === 'Plan') return <PlanPanel plans={analysis.local_plans ?? []} />;
  if (tab === 'Validation') return <ValidationPanel validations={analysis.local_validations ?? []} />;
  return <Artifacts artifacts={detail.artifact_paths ?? {}} />;
}

function Artifacts({ artifacts }: { artifacts: Record<string, { path: string; url: string }> }) {
  return (
    <div className="stack">
      {artifacts.svg ? (
        <div className="svg-preview">
          <img src={artifacts.svg.url} alt="Subgraph SVG" />
        </div>
      ) : (
        <p className="muted">No SVG preview available.</p>
      )}
      {Object.entries(artifacts).map(([kind, artifact]) => (
        <div className="artifact-row" key={kind}>
          <strong>{kind.toUpperCase()}</strong>
          <code>{artifact.path}</code>
          <a href={artifact.url} target="_blank" rel="noreferrer"><ExternalLink size={16} /> Open</a>
          <a href={artifact.url} download><Download size={16} /> Download</a>
        </div>
      ))}
      {artifacts.onnx ? <p className="muted">Netron command: <code>netron {artifacts.onnx.path}</code></p> : null}
    </div>
  );
}

function Small({ label, value }: { label: string; value: number }) {
  return (
    <div className="small-metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
