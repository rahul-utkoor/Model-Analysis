import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';

const HIGHLIGHT_TERMS = ['affine.for', 'affine.load', 'affine.store', 'memref.load', 'memref.store', 'linalg.matmul', 'krnl.matmul', 'onnx.MatMul', 'onnx.Gemm'];

interface Props {
  text: string;
  language?: string;
  highlightLines?: boolean;
}

export function CodeBlock({ text, language = 'text', highlightLines = false }: Props) {
  const [copied, setCopied] = useState(false);
  const lines = useMemo(() => text.split('\n'), [text]);

  async function copy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="code-block">
      <div className="code-block-toolbar">
        <span>{language}</span>
        <button onClick={copy} title="Copy code">{copied ? <Check size={15} /> : <Copy size={15} />} {copied ? 'Copied' : 'Copy'}</button>
      </div>
      <pre>{lines.map((line, index) => <code className={highlightLines && HIGHLIGHT_TERMS.some((term) => line.includes(term)) ? 'code-line-highlight' : ''} key={`${index}-${line}`}>{line}{'\n'}</code>)}</pre>
    </div>
  );
}
