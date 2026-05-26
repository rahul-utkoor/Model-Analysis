import { EvidencePanel } from './EvidencePanel';

export function PlanPanel({ plans }: { plans: any[] }) {
  if (!plans?.length) return <p className="muted">No symbolic pruning plan is attached to this subgraph.</p>;
  return (
    <div className="stack">
      {plans.map((plan) => (
        <div className="subcard" key={plan.plan_id}>
          <h3>{plan.plan_kind}</h3>
          <p className="muted">Status: {plan.plan_status} / target: {plan.target_dimension}</p>
          <p><strong>Index set:</strong> {plan.symbolic_index_set?.name ?? 'n/a'}</p>
          <EvidencePanel title="Actions" rows={plan.actions ?? []} columns={['action_type', 'target_source_name', 'target_axis', 'dimension']} />
        </div>
      ))}
    </div>
  );
}
