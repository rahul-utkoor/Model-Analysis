import type { ModelDetail } from '../types';
import { StatusBadge } from './StatusBadge';

export function ModelOverview({ detail, diagnosis }: { detail?: ModelDetail; diagnosis?: Record<string, any> }) {
  if (!detail) return null;
  const summary = detail.model_summary ?? {};
  const ranking = summary.ranking ?? {};
  const plans = summary.plans ?? {};
  const validation = summary.plan_validation ?? {};
  const gaps = diagnosis?.gaps ?? [];

  return (
    <section className="panel">
      <div className="section-heading">
        <h2>{detail.model_name}</h2>
        <p>Full-model static pruning-analysis summary.</p>
      </div>
      <div className="metric-grid compact">
        <SmallMetric label="Layers" value={summary.layers_generated} />
        <SmallMetric label="Subgraphs" value={summary.total_subgraphs} />
        <SmallMetric label="Safe candidates" value={ranking.safe} />
        <SmallMetric label="MLP safe" value={ranking.mlp_safe_candidates} />
        <SmallMetric label="Plans" value={plans.total_plans} />
        <SmallMetric label="Valid plans" value={validation.valid} />
      </div>
      <div className="summary-columns">
        <SummaryList title="Ranking" data={ranking} />
        <SummaryList title="Plans" data={plans} />
        <SummaryList title="Validation" data={validation} />
      </div>
      <div className="gap-strip">
        <strong>Remaining gaps</strong>
        {gaps.length ? gaps.map((gap: any) => <StatusBadge key={gap.gap_id ?? gap.gap_type} value={gap.gap_type} tone="unknown" />) : <span className="muted">None reported.</span>}
      </div>
    </section>
  );
}

function SmallMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="small-metric">
      <strong>{value ?? 0}</strong>
      <span>{label}</span>
    </div>
  );
}

function SummaryList({ title, data }: { title: string; data: Record<string, any> }) {
  const entries = Object.entries(data ?? {}).filter(([, value]) => typeof value !== 'object').slice(0, 8);
  return (
    <div className="summary-list">
      <h3>{title}</h3>
      {entries.map(([key, value]) => (
        <div key={key}>
          <span>{key}</span>
          <strong>{String(value)}</strong>
        </div>
      ))}
    </div>
  );
}
