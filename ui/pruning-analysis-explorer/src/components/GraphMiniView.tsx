import { ArrowRight } from 'lucide-react';

interface Props {
  nodes: string[];
  edges: string[][];
  dimensions?: string[];
}

export function GraphMiniView({ nodes, edges, dimensions }: Props) {
  return (
    <div className="graph-mini-view">
      <div className="graph-node-row">
        {nodes.map((node, index) => (
          <div className="graph-node-wrap" key={node}>
            <div className="graph-node">{node}</div>
            {dimensions?.[index] ? <small>{dimensions[index]}</small> : null}
          </div>
        ))}
      </div>
      <div className="graph-edge-list">
        {edges.map(([source, target]) => (
          <span key={`${source}-${target}`}><strong>{source}</strong><ArrowRight aria-hidden="true" /><strong>{target}</strong></span>
        ))}
      </div>
    </div>
  );
}
