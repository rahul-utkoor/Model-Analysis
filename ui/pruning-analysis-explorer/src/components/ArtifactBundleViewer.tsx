import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ArtifactBundle } from '../types';
import { DependenceJsonViewer } from './DependenceJsonViewer';
import { EvidenceSummaryPanel } from './EvidenceSummaryPanel';
import { MlirCodeViewer } from './MlirCodeViewer';
import { OnnxGraphViewer } from './OnnxGraphViewer';

type BundleTab = 'ONNX Graph' | 'DOT' | 'MLIR' | 'Dependence' | 'Evidence';

interface Props {
  model: string;
  layer: number;
  subgraph: string;
  compact?: boolean;
  fallbackArtifacts?: Record<string, { path: string; url: string }>;
}

const tabs: BundleTab[] = ['ONNX Graph', 'DOT', 'MLIR', 'Dependence', 'Evidence'];

export function ArtifactBundleViewer({ model, layer, subgraph, compact = false }: Props) {
  const [bundle, setBundle] = useState<ArtifactBundle>();
  const [tab, setTab] = useState<BundleTab>('ONNX Graph');
  const [error, setError] = useState<string>();

  useEffect(() => {
    setBundle(undefined);
    setError(undefined);
    setTab('ONNX Graph');
    api.artifactBundle(model, layer, subgraph).then(setBundle).catch((err) => setError(String(err)));
  }, [model, layer, subgraph]);

  return (
    <section className={`artifact-bundle-viewer ${compact ? 'compact' : ''}`}>
      <header>
        <div><p className="eyebrow">Read-only artifact bundle</p><h3>{bundle?.title ?? subgraph.replace(/_/g, ' ')}</h3></div>
        <code>{model} / layer {layer}</code>
      </header>
      <div className="artifact-tabs">
        {tabs.map((item) => <button className={tab === item ? 'active' : ''} key={item} onClick={() => setTab(item)}>{item}</button>)}
      </div>
      {!bundle && !error ? <p className="muted">Loading artifact bundle...</p> : null}
      {error ? <p className="artifact-warning">{error}</p> : null}
      {bundle && tab === 'ONNX Graph' ? <OnnxGraphViewer bundle={bundle} /> : null}
      {bundle && tab === 'DOT' ? <OnnxGraphViewer bundle={bundle} mode="dot" /> : null}
      {bundle && tab === 'MLIR' ? <MlirCodeViewer artifacts={bundle.mlir.artifacts} /> : null}
      {bundle && tab === 'Dependence' ? <DependenceJsonViewer bundle={bundle} /> : null}
      {bundle && tab === 'Evidence' ? <EvidenceSummaryPanel bundle={bundle} /> : null}
    </section>
  );
}
