import { CaseStudyCard } from './CaseStudyCard';
import type { CaseStudiesResponse } from '../types';

interface Props {
  studies?: CaseStudiesResponse;
}

export function CaseStudies({ studies }: Props) {
  return (
    <section>
      <div className="page-heading">
        <p className="eyebrow">Selected proof stories</p>
        <h1>Case Studies</h1>
        <p>Focused reports for explaining where propagation succeeds, where it blocks, and how local evidence improved.</p>
      </div>
      <div className="case-study-grid">
        {studies?.case_studies.map((study) => <CaseStudyCard key={study.id} study={study} />)}
      </div>
    </section>
  );
}
