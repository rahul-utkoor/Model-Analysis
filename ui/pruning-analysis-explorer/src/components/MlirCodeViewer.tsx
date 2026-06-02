import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import type { MlirArtifact } from '../types';
import { CodeBlock } from './CodeBlock';

export function MlirCodeViewer({ artifacts }: { artifacts: MlirArtifact[] }) {
  const [selectedPath, setSelectedPath] = useState(artifacts[0]?.path);
  const [text, setText] = useState('');
  const [error, setError] = useState<string>();
  const selected = useMemo(() => artifacts.find((artifact) => artifact.path === selectedPath) ?? artifacts[0], [artifacts, selectedPath]);

  useEffect(() => {
    setSelectedPath(artifacts[0]?.path);
  }, [artifacts]);

  useEffect(() => {
    if (!selected) return;
    setText('');
    setError(undefined);
    api.artifactText(selected.path).then((payload) => setText(payload.text)).catch((err) => setError(String(err)));
  }, [selected]);

  if (!artifacts.length) return <p className="artifact-warning">No pre-generated MLIR files were found for this subgraph.</p>;
  return (
    <div className="mlir-code-viewer stack">
      <div className="mlir-artifact-selector">
        {artifacts.map((artifact) => <button className={selected?.path === artifact.path ? 'active' : ''} key={artifact.path} onClick={() => setSelectedPath(artifact.path)}>{artifact.stage.replace(/_/g, ' ')}</button>)}
      </div>
      {selected ? (
        <>
          <p className="muted-note"><code>{selected.path}</code></p>
          <div className="dialect-badges">{selected.dialect_hints.map((hint) => <span key={hint}>{hint}</span>)}</div>
          {text ? <CodeBlock text={text} language="mlir" highlightLines /> : <p className="muted">Loading MLIR text...</p>}
        </>
      ) : null}
      {error ? <p className="artifact-warning">{error}</p> : null}
    </div>
  );
}
