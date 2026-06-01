import type { FinalSummary } from '../types';

interface Props {
  summary: FinalSummary;
}

const tiers = [
  ['native_mlir_dependence_evidence', 'Strongest', 'Native affine/load/store dependence facts'],
  ['actual_loop_access_evidence', 'Concrete', 'Python extraction over emitted MLIR accesses'],
  ['high_level_mlir_dialect_evidence', 'Fallback', 'High-level MLIR operator structure'],
  ['onnx_hint_fallback', 'Fallback', 'Conservative ONNX graph and shape hints'],
];

export function EvidenceHierarchy({ summary }: Props) {
  return (
    <section className="teaching-section">
      <div className="section-heading">
        <h2>Evidence Hierarchy</h2>
        <p>Local evidence is ranked explicitly. The current complete proof uses native MLIR evidence for every plan.</p>
      </div>
      <div className="evidence-tier-list">
        {tiers.map(([name, rank, note], index) => (
          <div className={`evidence-tier ${index === 0 ? 'active' : ''}`} key={name}>
            <span>{index + 1}</span>
            <div>
              <strong>{name}</strong>
              <small>{rank}: {note}</small>
            </div>
          </div>
        ))}
      </div>
      <div className="inline-metrics">
        <div><strong>{summary.native_mlir_evidence}</strong><span>native MLIR</span></div>
        <div><strong>{summary.fallback}</strong><span>fallback</span></div>
      </div>
    </section>
  );
}
