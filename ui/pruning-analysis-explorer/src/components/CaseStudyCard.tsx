import { ExternalLink, FileText } from 'lucide-react';
import type { CaseStudy } from '../types';

interface Props {
  study: CaseStudy;
}

export function CaseStudyCard({ study }: Props) {
  return (
    <article className="case-study-card">
      <div>
        <h3>{study.title}</h3>
        <p>{study.summary}</p>
      </div>
      <div className="case-study-numbers">
        {Object.entries(study.key_numbers).map(([label, value]) => <span key={label}><strong>{value}</strong>{label.replace(/_/g, ' ')}</span>)}
      </div>
      {study.available ? (
        <a className="text-action" href={study.report_url} target="_blank" rel="noreferrer">
          <ExternalLink aria-hidden="true" /> Open report
        </a>
      ) : (
        <span className="text-action disabled"><FileText aria-hidden="true" /> Report unavailable</span>
      )}
    </article>
  );
}
