import type { AxisRelation } from '../types';

interface Props {
  relations: AxisRelation[];
}

export function AxisRelationView({ relations }: Props) {
  return (
    <div className="axis-relation-view">
      {relations.map((relation) => (
        <span className={`relation-pill relation-${relation.relation.toLowerCase().replace(/[^a-z]+/g, '-')}`} key={`${relation.source}-${relation.target}-${relation.relation}`}>
          <code>{relation.source}</code>
          <span>to</span>
          <code>{relation.target}</code>
          <strong>{relation.relation}</strong>
        </span>
      ))}
    </div>
  );
}
