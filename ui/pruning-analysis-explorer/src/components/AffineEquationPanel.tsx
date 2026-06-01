interface Props {
  equations: string[];
  title?: string;
}

export function AffineEquationPanel({ equations, title = 'Affine access form' }: Props) {
  return (
    <div className="affine-equation-panel">
      <small>{title}</small>
      {equations.map((equation) => <code className="affine-equation" key={equation}>{equation}</code>)}
    </div>
  );
}
