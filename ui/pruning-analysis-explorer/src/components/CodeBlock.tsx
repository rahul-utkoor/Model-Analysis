import { Check, Copy } from 'lucide-react';
import { useMemo, useState } from 'react';

const HIGHLIGHT_TERMS = ['affine.for', 'affine.load', 'affine.store', 'memref.load', 'memref.store', 'linalg.matmul', 'krnl.matmul', 'onnx.MatMul', 'onnx.Gemm'];

interface Props {
  text: string;
  language?: string;
  highlightLines?: boolean;
  lineStart?: number;
  matchLines?: number[];
  searchTerm?: string;
  label?: string;
}

export function CodeBlock({ text, language = 'text', highlightLines = false, lineStart = 1, matchLines = [], searchTerm = '', label }: Props) {
  const [copied, setCopied] = useState(false);
  const lines = useMemo(() => text.split('\n'), [text]);
  const matchSet = useMemo(() => new Set(matchLines), [matchLines]);
  const normalizedSearch = searchTerm.trim().toLowerCase();

  async function copy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="code-block">
      <div className="code-block-toolbar">
        <span>{label ? `${language} / ${label}` : language}</span>
        <button onClick={copy} title="Copy code">{copied ? <Check size={15} /> : <Copy size={15} />} {copied ? 'Copied' : 'Copy'}</button>
      </div>
      <pre>{lines.map((line, index) => {
        const lineNo = lineStart + index;
        const classes = ['code-line'];
        if ((highlightLines && HIGHLIGHT_TERMS.some((term) => line.includes(term))) || matchSet.has(lineNo)) classes.push('code-line-highlight');
        if (normalizedSearch && line.toLowerCase().includes(normalizedSearch)) classes.push('code-line-search-hit');
        return <code className={classes.join(' ')} key={`${lineNo}-${line}`}><span className="code-line-number">{lineNo}</span><span className="code-line-text">{line || ' '}</span></code>;
      })}</pre>
    </div>
  );
}
