import type { PruningClass } from '../types';

export function StatusBadge({ value, tone }: { value: string | number | undefined; tone?: PruningClass }) {
  const label = String(value ?? 'unknown');
  const key = (tone ?? label).toString().toLowerCase().replace(/_/g, '-');
  return <span className={`badge badge-${key}`}>{label}</span>;
}
