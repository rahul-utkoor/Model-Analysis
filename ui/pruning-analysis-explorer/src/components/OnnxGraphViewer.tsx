import { Download, ExternalLink } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ArtifactBundle } from '../types';
import { CodeBlock } from './CodeBlock';

interface Props {
  bundle: ArtifactBundle;
  mode?: 'graph' | 'dot';
}

export function OnnxGraphViewer({ bundle, mode = 'graph' }: Props) {
  const [dot, setDot] = useState<string>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (mode !== 'dot' && bundle.links.svg) return;
    if (!bundle.paths.dot) return;
    api.artifactText(bundle.paths.dot).then((payload) => setDot(payload.text)).catch((err) => setError(String(err)));
  }, [bundle.paths.dot, bundle.links.svg, mode]);

  return (
    <div className="onnx-graph-viewer stack">
      <div className="artifact-link-row">
        {bundle.links.onnx ? <a href={bundle.links.onnx} target="_blank" rel="noreferrer"><ExternalLink size={15} /> Open ONNX</a> : null}
        {bundle.links.onnx ? <a href={bundle.links.onnx} download><Download size={15} /> Download ONNX</a> : null}
        {bundle.links.svg ? <a href={bundle.links.svg} target="_blank" rel="noreferrer"><ExternalLink size={15} /> Open SVG</a> : null}
        {bundle.links.dot ? <a href={bundle.links.dot} target="_blank" rel="noreferrer"><ExternalLink size={15} /> Open DOT</a> : null}
      </div>
      {bundle.paths.onnx ? <p className="muted-note"><code>{bundle.paths.onnx}</code></p> : null}
      {mode === 'graph' && bundle.links.svg ? <div className="onnx-graph-preview"><img src={bundle.links.svg} alt={`${bundle.title} ONNX graph`} /></div> : null}
      {(mode === 'dot' || !bundle.links.svg) && dot ? <CodeBlock text={dot} language="dot" /> : null}
      {!bundle.links.svg && !bundle.paths.dot ? <p className="artifact-warning">No SVG or DOT preview is available. Use the ONNX link when present.</p> : null}
      {error ? <p className="artifact-warning">{error}</p> : null}
    </div>
  );
}
