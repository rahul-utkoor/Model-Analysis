import type { ArtifactBundle } from '../types';

export function EvidenceSummaryPanel({ bundle }: { bundle: ArtifactBundle }) {
  const blocker = bundle.evidence.dfa_verdict.includes('blocked');
  return (
    <div className="evidence-summary-panel stack">
      <div className="evidence-summary-grid">
        <Summary label="Pattern" value={bundle.evidence.pattern} />
        <Summary label="Evidence tier" value={bundle.evidence.evidence_tier} tone={bundle.evidence.evidence_tier.startsWith('native') ? 'native' : 'neutral'} />
        <Summary label="DFA verdict" value={bundle.evidence.dfa_verdict} tone={blocker ? 'blocker' : 'native'} />
      </div>
      <div>
        <h4>Axis relations</h4>
        <div className="axis-relation-list">
          {bundle.evidence.axis_relations.length ? bundle.evidence.axis_relations.map((relation) => <code key={`${relation.source}-${relation.target}`}>{relation.source} -&gt; {relation.target} = {relation.relation}</code>) : <span className="muted">No local axis summary available.</span>}
        </div>
      </div>
      {bundle.warnings.length ? <div>{bundle.warnings.map((warning) => <p className="artifact-warning" key={warning}>{warning}</p>)}</div> : null}
    </div>
  );
}

function Summary({ label, value, tone = 'neutral' }: { label: string; value: string; tone?: string }) {
  return <div><span>{label}</span><strong className={`evidence-value ${tone}`}>{value.replace(/_/g, ' ')}</strong></div>;
}
