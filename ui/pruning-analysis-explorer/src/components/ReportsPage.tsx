import { ExternalLink, EyeOff, FileText } from 'lucide-react';
import { useState } from 'react';
import { api } from '../api';
import type { ReportTextResponse } from '../types';

const reportGroups = [
  ['Final', [['Final report', 'final/static_pruning_propagation_final_report.md'], ['Claims', 'final/static_pruning_propagation_claims.md']]],
  ['Formalization', [['Formalization notes', 'formalization/static_pruning_propagation_notes.md'], ['Slide outline', 'formalization/teaching_slide_outline.md'], ['Paper methodology', 'formalization/paper_methodology_outline.md']]],
  ['Model proofs', [['All-model proof', 'all_model_plan_proof/index.md'], ['BERT 24-plan proof', 'bert_24_plan_proof/index.md']]],
  ['Diagnostics', [['OPT FFN native diagnosis', 'opt_ffn_native_diagnosis/index.md']]],
];

export function ReportsPage() {
  const [selected, setSelected] = useState<ReportTextResponse>();
  const [previewOpen, setPreviewOpen] = useState(false);
  const [error, setError] = useState<string>();

  function openReport(path: string) {
    setError(undefined);
    api.reportText(path).then((report) => {
      setSelected(report);
      setPreviewOpen(true);
    }).catch((err) => setError(String(err)));
  }

  return (
    <section>
      <div className="page-heading">
        <p className="eyebrow">Read-only report library</p>
        <h1>Reports</h1>
        <p>Open concise proof, formalization, and diagnostic reports as needed.</p>
      </div>
      <div className="reports-layout">
        <div className="report-list">
          {reportGroups.map(([group, reports]) => (
            <section className="report-group" key={group as string}>
              <h2>{group as string}</h2>
              {(reports as string[][]).map(([title, path]) => (
                <article className="report-row" key={path}>
                  <FileText aria-hidden="true" />
                  <div><strong>{title}</strong><small>{path}</small></div>
                  <button className="icon-action" onClick={() => openReport(path)} title={`Preview ${title}`}><ExternalLink aria-hidden="true" /></button>
                </article>
              ))}
            </section>
          ))}
        </div>
        <section className={`report-preview ${previewOpen ? 'open' : 'collapsed'}`}>
          <div className="section-heading tight">
            <h2>{selected?.path ?? 'Report preview'}</h2>
            <p>{selected ? 'Read-only text view' : 'Select a report to open the preview.'}</p>
          </div>
          {previewOpen ? <button className="secondary-action" onClick={() => setPreviewOpen(false)}><EyeOff aria-hidden="true" /> Collapse preview</button> : null}
          {error ? <div className="error-banner">{error}</div> : null}
          {selected && previewOpen ? <pre>{selected.text}</pre> : null}
        </section>
      </div>
    </section>
  );
}
