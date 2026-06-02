import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import type { ArtifactBundle } from '../types';
import { CodeBlock } from './CodeBlock';

type JsonKind = 'native_json' | 'python_json';

export function DependenceJsonViewer({ bundle }: { bundle: ArtifactBundle }) {
  const available = (['native_json', 'python_json'] as JsonKind[]).filter((kind) => bundle.dependence[kind]);
  const [selected, setSelected] = useState<JsonKind>(available[0] ?? 'native_json');
  const [text, setText] = useState('');
  const [error, setError] = useState<string>();
  const summary = useMemo(() => summarize(text), [text]);

  useEffect(() => {
    const path = bundle.dependence[selected];
    if (!path) return;
    setText('');
    setError(undefined);
    api.artifactText(path).then((payload) => setText(payload.text)).catch((err) => setError(String(err)));
  }, [bundle.dependence, selected]);

  if (!available.length) return <p className="artifact-warning">No native or Python dependence JSON was found for this subgraph.</p>;
  return (
    <div className="dependence-json-viewer stack">
      <div className="mlir-artifact-selector">
        {available.map((kind) => <button className={selected === kind ? 'active' : ''} key={kind} onClick={() => setSelected(kind)}>{kind.replace(/_/g, ' ')}</button>)}
      </div>
      <div className="dependence-metrics">
        <Metric label="relations" value={summary.relations} />
        <Metric label="preserved" value={summary.preserved} />
        <Metric label="reduced" value={summary.reduced} />
        <Metric label="mixed / blocked" value={summary.blocked} />
      </div>
      {text ? <CodeBlock text={text} language="json" /> : <p className="muted">Loading dependence JSON...</p>}
      {error ? <p className="artifact-warning">{error}</p> : null}
    </div>
  );
}

function summarize(text: string) {
  try {
    const relations = JSON.parse(text).relations ?? [];
    return {
      relations: relations.length,
      preserved: relations.filter((relation: any) => relation.relation_kind === 'preserved').length,
      reduced: relations.filter((relation: any) => relation.relation_kind === 'reduced').length,
      blocked: relations.filter((relation: any) => ['mixed', 'blocked'].includes(relation.relation_kind)).length,
    };
  } catch {
    return { relations: 0, preserved: 0, reduced: 0, blocked: 0 };
  }
}

function Metric({ label, value }: { label: string; value: number }) {
  return <div><strong>{value}</strong><span>{label}</span></div>;
}
