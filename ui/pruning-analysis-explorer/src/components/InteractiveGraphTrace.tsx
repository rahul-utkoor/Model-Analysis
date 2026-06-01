import { ArrowRight } from 'lucide-react';
import type { EvidenceTraceExample, TraceStep } from '../types';

interface Props {
  example: EvidenceTraceExample;
  step: TraceStep;
}

function isActiveEdge(step: TraceStep, source: string, target: string) {
  return step.active_edges.some(([edgeSource, edgeTarget]) => edgeSource === source && edgeTarget === target);
}

export function InteractiveGraphTrace({ example, step }: Props) {
  return (
    <section className="trace-graph">
      <div className="trace-graph-header">
        <span>Graph transition</span>
        <code>{step.fact}</code>
      </div>
      <div className="trace-node-grid">
        {example.graph.nodes.map((node) => {
          const active = step.active_nodes.includes(node.id);
          const state = active && ['blocker', 'blocked'].includes(step.kind) ? 'blocked' : active && step.kind !== 'seed' ? 'dead' : active ? 'active' : 'normal';
          return (
            <article className={`trace-node trace-node-${state}`} key={node.id}>
              <small>{node.op}</small>
              <strong>{node.label}</strong>
              <span>{node.axis_role}</span>
              {node.shape ? <code>{node.shape}</code> : null}
            </article>
          );
        })}
      </div>
      <div className="trace-edge-grid">
        {example.graph.edges.map((edge) => (
          <div className={`trace-edge ${isActiveEdge(step, edge.source, edge.target) ? 'trace-edge-active' : ''}`} key={`${edge.source}-${edge.target}`}>
            <strong>{edge.source}</strong>
            <ArrowRight aria-hidden="true" />
            <strong>{edge.target}</strong>
            <span>axis {edge.axis}</span>
            <em>{edge.relation}</em>
          </div>
        ))}
      </div>
    </section>
  );
}
