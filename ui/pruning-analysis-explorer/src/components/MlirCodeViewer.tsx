import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import type { ArtifactTextResponse, MlirArtifact } from '../types';
import { CodeBlock } from './CodeBlock';

type ViewMode = 'focused' | 'full';
type FocusKind = 'affine' | 'loops' | 'loads' | 'stores' | 'matmul';

const COUNT_KEYS = ['affine.for', 'affine.load', 'affine.store', 'scf.for', 'memref.load', 'memref.store', 'krnl.matmul', 'linalg.matmul', 'onnx.MatMul', 'onnx.Gemm'];

export function MlirCodeViewer({ artifacts }: { artifacts: MlirArtifact[] }) {
  const defaultArtifact = useMemo(() => chooseRichestArtifact(artifacts), [artifacts]);
  const [selectedPath, setSelectedPath] = useState(defaultArtifact?.path);
  const [payload, setPayload] = useState<ArtifactTextResponse>();
  const [mode, setMode] = useState<ViewMode>('focused');
  const [focus, setFocus] = useState<FocusKind>('affine');
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string>();
  const selected = useMemo(() => artifacts.find((artifact) => artifact.path === selectedPath) ?? defaultArtifact, [artifacts, defaultArtifact, selectedPath]);

  useEffect(() => {
    setSelectedPath(defaultArtifact?.path);
  }, [defaultArtifact]);

  useEffect(() => {
    if (!selected) return;
    setPayload(undefined);
    setError(undefined);
    api.artifactText(selected.path, mode === 'focused' ? { focus, context: 4 } : undefined).then(setPayload).catch((err) => setError(String(err)));
  }, [focus, mode, selected]);

  if (!artifacts.length) return <p className="artifact-warning">No pre-generated MLIR files were found for this subgraph.</p>;
  const affineCount = count(selected, 'affine.for') + count(selected, 'affine.load') + count(selected, 'affine.store');
  const fallbackCount = count(selected, 'krnl.matmul') + count(selected, 'linalg.matmul') + count(selected, 'onnx.MatMul') + count(selected, 'onnx.Gemm');
  return (
    <div className="mlir-code-viewer stack">
      <div className="mlir-view-banner">Showing affine/load/store regions, not just the file header.</div>
      <div className="mlir-artifact-selector">
        {artifacts.map((artifact) => <button className={selected?.path === artifact.path ? 'active' : ''} key={artifact.path} onClick={() => setSelectedPath(artifact.path)}>{artifact.stage.replace(/_/g, ' ')} <small>{artifact.line_count} lines</small></button>)}
      </div>
      {selected ? (
        <>
          <p className="muted-note"><code>{selected.path}</code></p>
          <div className="dialect-badges">{selected.dialect_hints.map((hint) => <span key={hint}>{hint}</span>)}</div>
          <div className="mlir-summary-badges">
            {COUNT_KEYS.filter((key) => count(selected, key)).map((key) => <span key={key}><strong>{count(selected, key)}</strong> {key}</span>)}
            {selected.first_interesting_line ? <span><strong>line {selected.first_interesting_line}</strong> first match</span> : null}
          </div>
          <div className="mlir-toolbar">
            <div className="segmented-control" aria-label="MLIR view mode">
              <button className={mode === 'focused' ? 'active' : ''} onClick={() => setMode('focused')}>Focused loop/access regions</button>
              <button className={mode === 'full' ? 'active' : ''} onClick={() => setMode('full')}>Full file</button>
            </div>
            <input aria-label="Search MLIR" placeholder="Search MLIR, e.g. affine.load" value={search} onChange={(event) => setSearch(event.target.value)} />
          </div>
          {mode === 'focused' ? (
            <div className="mlir-jump-row">
              <button disabled={!count(selected, 'affine.for')} onClick={() => setFocus('loops')}>Jump to first affine.for</button>
              <button disabled={!count(selected, 'affine.load') && !count(selected, 'memref.load')} onClick={() => setFocus('loads')}>Jump to first load</button>
              <button disabled={!count(selected, 'affine.store') && !count(selected, 'memref.store')} onClick={() => setFocus('stores')}>Jump to first store</button>
              <button disabled={!fallbackCount} onClick={() => setFocus('matmul')}>Show ONNX/Gemm fallback</button>
              <button onClick={() => setFocus('affine')}>Show affine regions</button>
            </div>
          ) : null}
          {!affineCount && mode === 'focused' ? <p className="artifact-warning">No affine loopnest found in this MLIR artifact. Showing fallback high-level operations when available.</p> : null}
          {payload?.warnings.map((warning) => <p className="artifact-warning" key={warning}>{warning}</p>)}
          {payload && mode === 'focused' && payload.sections.length ? (
            <div className="mlir-sections">
              {payload.sections.map((section) => (
                <section className="mlir-section" key={`${section.start_line}-${section.end_line}`}>
                  <h4 className="mlir-section-title">{section.title} <span>lines {section.start_line}-{section.end_line}</span></h4>
                  <CodeBlock text={section.text} language="mlir" label="focused region" lineStart={section.start_line} matchLines={section.match_lines} searchTerm={search} highlightLines />
                </section>
              ))}
            </div>
          ) : null}
          {payload && mode === 'focused' && !payload.sections.length ? <p className="artifact-warning">No loop, access, or fallback operation matched this focus.</p> : null}
          {payload && mode === 'full' ? <CodeBlock text={payload.text} language="mlir" label="full file" searchTerm={search} highlightLines /> : null}
          {!payload ? <p className="muted">Loading MLIR text...</p> : null}
        </>
      ) : null}
      {error ? <p className="artifact-warning">{error}</p> : null}
    </div>
  );
}

function count(artifact: MlirArtifact | undefined, key: string) {
  return artifact?.interesting_counts[key] ?? 0;
}

function chooseRichestArtifact(artifacts: MlirArtifact[]) {
  return [...artifacts].sort((left, right) => score(right) - score(left))[0];
}

function score(artifact: MlirArtifact) {
  return count(artifact, 'affine.for') * 100 + count(artifact, 'affine.load') * 10 + count(artifact, 'affine.store') * 10 + (artifact.stage === 'lowered_affine' ? 1 : 0);
}
