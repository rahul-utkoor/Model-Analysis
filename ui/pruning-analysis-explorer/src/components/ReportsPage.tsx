import { ExternalLink, FileText } from 'lucide-react';
import { useState } from 'react';
import { api } from '../api';
import type { ReportTextResponse } from '../types';

const reports = [
  ['Final report', 'final/static_pruning_propagation_final_report.md'],
  ['Claims', 'final/static_pruning_propagation_claims.md'],
  ['All-model proof', 'all_model_plan_proof/index.md'],
  ['BERT 24-plan proof', 'bert_24_plan_proof/index.md'],
  ['Formalization notes', 'formalization/static_pruning_propagation_notes.md'],
  ['Teaching slide outline', 'formalization/teaching_slide_outline.md'],
  ['Paper methodology outline', 'formalization/paper_methodology_outline.md'],
];

export function ReportsPage() {
  const [selected, setSelected] = useState<ReportTextResponse>();
  const [error, setError] = useState<string>();

  function openReport(path: string) {
    setError(undefined);
    api.reportText(path).then(setSelected).catch((err) => setError(String(err)));
  }

  return (
    <section>
      <div className="page-heading">
        <p className="eyebrow">Read-only report library</p>
        <h1>Reports</h1>
        <p>Open the consolidated proof, claims, formalization notes, and teaching outlines.</p>
      </div>
      <div className="reports-layout">
        <div className="report-list">
          {reports.map(([title, path]) => (
            <article className="report-row" key={path}>
              <FileText aria-hidden="true" />
              <div>
                <strong>{title}</strong>
                <small>{path}</small>
              </div>
              <button className="icon-action" onClick={() => openReport(path)} title={`Preview ${title}`}>
                <ExternalLink aria-hidden="true" />
              </button>
            </article>
          ))}
        </div>
        <section className="report-preview">
          <div className="section-heading tight">
            <h2>{selected?.path ?? 'Report preview'}</h2>
            <p>{selected ? 'Read-only text view' : 'Select a report to view its Markdown text.'}</p>
          </div>
          {error ? <div className="error-banner">{error}</div> : null}
          {selected ? <pre>{selected.text}</pre> : null}
        </section>
      </div>
    </section>
  );
}
