import { CheckCircle2, ShieldAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { AxisAnimationPanel } from './AxisAnimationPanel';
import { ArtifactBundleViewer } from './ArtifactBundleViewer';
import { InteractiveGraphTrace } from './InteractiveGraphTrace';
import { MlirCodeTrace } from './MlirCodeTrace';
import { PatternMatchView } from './PatternMatchView';
import { TraceExampleSelector } from './TraceExampleSelector';
import { TraceTimeline } from './TraceTimeline';
import type { EvidenceArtifactMap, EvidenceTracesResponse } from '../types';

interface Props {
  data?: EvidenceTracesResponse;
  artifactMap?: EvidenceArtifactMap;
  selectedExample?: string;
  onSelectExample: (id: string) => void;
}

type EvidenceTab = 'pattern' | 'mlir' | 'axis' | 'verdict';

export function EvidenceTracePage({ data, artifactMap, selectedExample, onSelectExample }: Props) {
  const [activeStep, setActiveStep] = useState(0);
  const [activeTab, setActiveTab] = useState<EvidenceTab>('pattern');
  const example = useMemo(() => data?.examples.find((item) => item.id === selectedExample) ?? data?.examples[0], [data, selectedExample]);

  useEffect(() => {
    setActiveStep(0);
    setActiveTab('pattern');
  }, [example?.id]);

  if (!data || !example) return null;

  const blocked = example.verdict === 'blocked_as_expected';
  return (
    <section className="evidence-trace-page">
      <header className="trace-page-heading">
        <div><p className="eyebrow">Pattern laboratory</p><h1>{data.summary.title}</h1><p>{data.summary.description}</p></div>
        <div className="proof-badge"><strong>{data.summary.plans_proven}</strong><span>native-backed plans</span></div>
      </header>
      <TraceExampleSelector examples={data.examples} selectedId={example.id} onSelect={onSelectExample} />
      <div className="trace-layout">
        <InteractiveGraphTrace example={example} step={example.dfa_trace[activeStep]} />
        <section className="trace-evidence-panel">
          <div className="trace-tabs" role="tablist" aria-label="Evidence panels">
            {(['pattern', 'mlir', 'axis', 'verdict'] as EvidenceTab[]).map((tab) => <button className={activeTab === tab ? 'active' : ''} key={tab} onClick={() => setActiveTab(tab)}>{tab}</button>)}
          </div>
          {activeTab === 'pattern' ? <PatternMatchView example={example} /> : null}
          {activeTab === 'mlir' ? <MlirCodeTrace example={example} /> : null}
          {activeTab === 'axis' ? <AxisAnimationPanel example={example} stepIndex={activeStep} /> : null}
          {activeTab === 'verdict' ? (
            <div className={`trace-verdict ${blocked ? 'blocked' : 'proven'}`}>
              {blocked ? <ShieldAlert aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />}
              <div><strong>{example.verdict.replace(/_/g, ' ')}</strong><code>{example.pattern}</code></div>
              <ul className="compact-list">{example.not_claimed.map((item) => <li key={item}>{item}</li>)}</ul>
            </div>
          ) : null}
        </section>
      </div>
      <TraceTimeline steps={example.dfa_trace} activeIndex={activeStep} onChange={setActiveStep} />
      {artifactMap?.[example.id] ? (
        <section className="real-artifact-panel">
          <div className="section-heading"><p className="eyebrow">Real artifact</p><h2>Generated ONNX and MLIR evidence</h2></div>
          <ArtifactBundleViewer compact model={artifactMap[example.id].model} layer={artifactMap[example.id].layer} subgraph={artifactMap[example.id].subgraph} />
        </section>
      ) : <p className="artifact-warning">No artifact bundle mapping is available for this trace.</p>}
    </section>
  );
}
