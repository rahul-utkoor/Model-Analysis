import { useMemo, useState } from 'react';
import type { SubgraphSummary } from '../types';
import { StatusBadge } from './StatusBadge';

export function SubgraphTable({
  subgraphs,
  selectedNode,
  onSelect
}: {
  subgraphs: SubgraphSummary[];
  selectedNode?: string;
  onSelect: (node: string) => void;
}) {
  const [text, setText] = useState('');
  const [klass, setKlass] = useState('all');
  const [validOnly, setValidOnly] = useState(false);
  const filtered = useMemo(() => {
    return subgraphs.filter((item) => {
      const haystack = `${item.display_name} ${item.semantic_category} ${item.pruning_class} ${item.plan_status} ${item.validation_status}`.toLowerCase();
      if (text && !haystack.includes(text.toLowerCase())) return false;
      if (klass !== 'all' && item.pruning_class !== klass) return false;
      if (validOnly && item.validation_status !== 'valid') return false;
      return true;
    });
  }, [subgraphs, text, klass, validOnly]);

  return (
    <section className="panel subgraph-table-panel">
      <div className="section-heading tight">
        <h2>Subgraphs</h2>
        <p>Ordered abstract nodes for the selected layer/block.</p>
      </div>
      <div className="filters">
        <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Filter subgraphs" />
        <select value={klass} onChange={(event) => setKlass(event.target.value)}>
          <option value="all">All classes</option>
          <option value="safe">Safe</option>
          <option value="constrained">Constrained</option>
          <option value="blocked">Blocked</option>
          <option value="auxiliary">Auxiliary</option>
          <option value="unknown">Unknown</option>
        </select>
        <label className="check-filter">
          <input type="checkbox" checked={validOnly} onChange={(event) => setValidOnly(event.target.checked)} />
          Valid plan
        </label>
      </div>
      <table className="data-table clickable">
        <thead>
          <tr>
            <th>#</th>
            <th>Abstract node</th>
            <th>Category</th>
            <th>Class</th>
            <th>Plan</th>
            <th>Validation</th>
            <th>Artifact</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((item) => (
            <tr key={item.node_slug} className={selectedNode === item.node_slug ? 'selected' : ''} onClick={() => onSelect(item.node_slug)}>
              <td>{item.ordinal}</td>
              <td>{item.display_name}</td>
              <td>{item.semantic_category}</td>
              <td><StatusBadge value={item.pruning_class} /></td>
              <td>{item.plan_status}</td>
              <td><StatusBadge value={item.validation_status} /></td>
              <td>{item.onnx_status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
