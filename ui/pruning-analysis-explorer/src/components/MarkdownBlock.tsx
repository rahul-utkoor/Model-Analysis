export function MarkdownBlock({ text }: { text: string }) {
  if (!text) return <p className="muted">No explanation text available.</p>;
  return <pre className="markdown-block">{text}</pre>;
}
