import type { ReactNode } from 'react';

interface Props {
  title: string;
  eyebrow?: string;
  children: ReactNode;
}

export function ConceptCard({ title, eyebrow, children }: Props) {
  return (
    <article className="concept-card">
      {eyebrow ? <small>{eyebrow}</small> : null}
      <h3>{title}</h3>
      {children}
    </article>
  );
}
