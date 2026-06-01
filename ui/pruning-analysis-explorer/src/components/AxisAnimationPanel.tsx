import type { EvidenceTraceExample } from '../types';

interface AxisRow {
  axis: string;
  cells: string[];
  state: string;
}

function rowsFor(example: EvidenceTraceExample): { headings: string[]; rows: AxisRow[] } {
  if (example.id === 'ffn_intermediate') {
    return { headings: ['Axis', 'Producer', 'Activation', 'Consumer', 'State'], rows: [
      { axis: 'j', cells: ['expansion.output', 'activation.in / out', 'contraction.input'], state: 'DEAD propagated' },
      { axis: 'h', cells: ['hidden input', 'unchanged', 'hidden output'], state: 'PROTECTED' },
    ] };
  }
  if (example.id === 'attention_value_path') {
    return { headings: ['Axis', 'Value Projection', 'Context', 'Output Projection', 'State'], rows: [
      { axis: 'd', cells: ['value.output', 'context.value', 'output.input'], state: 'DEAD propagated' },
      { axis: 'k', cells: ['key position', 'reduced', 'not propagated'], state: 'REDUCED' },
    ] };
  }
  return { headings: ['Axis', 'Q', 'K', 'Score', 'State'], rows: [
    { axis: 'd', cells: ['q.head_dim', 'k.head_dim', 'reduced away'], state: 'BLOCKED' },
    { axis: 'q / k', cells: ['query position', 'key position', 'score output'], state: 'LIVE' },
  ] };
}

export function AxisAnimationPanel({ example, stepIndex }: { example: EvidenceTraceExample; stepIndex: number }) {
  const { headings, rows } = rowsFor(example);
  return (
    <section className="axis-animation-panel">
      <table className="data-table dense">
        <thead><tr>{headings.map((heading) => <th key={heading}>{heading}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr className={index === 0 && stepIndex > 0 ? 'axis-row-active' : ''} key={row.axis}>
              <td><code>{row.axis}</code></td>
              {row.cells.map((cell) => <td key={cell}>{cell}</td>)}
              <td><span className={`axis-state axis-state-${row.state.toLowerCase().replace(/[^a-z]+/g, '-')}`}>{row.state}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
