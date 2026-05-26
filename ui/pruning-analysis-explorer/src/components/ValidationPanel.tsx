import { StatusBadge } from './StatusBadge';

export function ValidationPanel({ validations }: { validations: any[] }) {
  if (!validations?.length) return <p className="muted">No plan validation is attached to this subgraph.</p>;
  return (
    <div className="stack">
      {validations.map((validation) => (
        <div className="subcard" key={validation.validation_id}>
          <div className="inline-heading">
            <h3>Validation</h3>
            <StatusBadge value={validation.validation_status} />
            <StatusBadge value={`score ${validation.validation_score}`} tone={validation.validation_status} />
          </div>
          <div className="check-groups">
            <CheckList title="Failed checks" items={validation.failed_checks} tone="invalid" />
            <CheckList title="Warnings" items={validation.warning_checks} tone="warning" />
          </div>
        </div>
      ))}
    </div>
  );
}

function CheckList({ title, items, tone }: { title: string; items: string[]; tone: string }) {
  return (
    <div>
      <h4>{title}</h4>
      {items?.length ? items.map((item) => <StatusBadge key={item} value={item} tone={tone} />) : <span className="muted">none</span>}
    </div>
  );
}
