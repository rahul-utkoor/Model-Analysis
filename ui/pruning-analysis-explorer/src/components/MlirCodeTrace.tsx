import type { EvidenceTraceExample } from '../types';

interface Props {
  example: EvidenceTraceExample;
}

function highlightCode(exampleId: string, code: string) {
  return code.split(/(\b[jdk]\b)/g).map((part, index) => {
    const isAxis = ['j', 'd', 'k'].includes(part);
    const reduced = (exampleId === 'qk_score_blocker' && part === 'd') || (exampleId === 'attention_value_path' && part === 'k');
    return isAxis ? <mark className={reduced ? 'axis-token-reduced' : 'axis-token-preserved'} key={`${part}-${index}`}>{part}</mark> : part;
  });
}

export function MlirCodeTrace({ example }: Props) {
  return (
    <section className="mlir-code-trace">
      {example.mlir.map((snippet) => (
        <article key={snippet.title}>
          <small>{snippet.title}</small>
          <code>{highlightCode(example.id, snippet.code)}</code>
          <span>{snippet.relation}</span>
        </article>
      ))}
      <div className="axis-legend">
        <span><i className="legend-preserved" /> preserved axis</span>
        <span><i className="legend-reduced" /> reduced / mixed axis</span>
      </div>
    </section>
  );
}
