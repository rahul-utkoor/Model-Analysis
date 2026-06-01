import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { CaseStudies } from './components/CaseStudies';
import { CoverageDashboard } from './components/CoverageDashboard';
import { LayerNavigator } from './components/LayerNavigator';
import { Layout, type AppView } from './components/Layout';
import { ModelOverview } from './components/ModelOverview';
import { OverviewDashboard } from './components/OverviewDashboard';
import { PipelineOverview } from './components/PipelineOverview';
import { ReportsPage } from './components/ReportsPage';
import { SubgraphDetail } from './components/SubgraphDetail';
import { SubgraphTable } from './components/SubgraphTable';
import { TeachingFlow } from './components/TeachingFlow';
import {
  chooseDefaultSubgraph,
  makePreviousSubgraphIntent,
  subgraphBelongsToLoadedLayer,
  type PreviousSubgraphIntent,
} from './selection';
import type {
  CaseStudiesResponse,
  CoverageResponse,
  LayerSummary,
  ModelDetail,
  ModelSummary,
  OverviewResponse,
  ProofSummaryResponse,
  SearchMatch,
  SubgraphDetailResponse,
  SubgraphSummary,
  TeachingFlowResponse,
} from './types';

type LoadedSubgraphContext = {
  model: string;
  layer: number;
};

type PendingNavigation = {
  model: string;
  layer?: number;
  nodeSlug?: string;
};

function sameContext(ctx: LoadedSubgraphContext | undefined, model?: string, layer?: number): boolean {
  return Boolean(ctx && model && layer !== undefined && ctx.model === model && ctx.layer === layer);
}

export default function App() {
  const [activeView, setActiveView] = useState<AppView>('dashboard');
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [coverage, setCoverage] = useState<CoverageResponse>();
  const [overview, setOverview] = useState<OverviewResponse>();
  const [proofSummary, setProofSummary] = useState<ProofSummaryResponse>();
  const [teachingFlow, setTeachingFlow] = useState<TeachingFlowResponse>();
  const [caseStudies, setCaseStudies] = useState<CaseStudiesResponse>();
  const [selectedModel, setSelectedModel] = useState<string>();
  const [modelDetail, setModelDetail] = useState<ModelDetail>();
  const [diagnosis, setDiagnosis] = useState<Record<string, any>>();
  const [layers, setLayers] = useState<LayerSummary[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<number>();
  const [subgraphs, setSubgraphs] = useState<SubgraphSummary[]>([]);
  const [subgraphsContext, setSubgraphsContext] = useState<LoadedSubgraphContext>();
  const [selectedNode, setSelectedNode] = useState<string>();
  const [subgraphDetail, setSubgraphDetail] = useState<SubgraphDetailResponse>();
  const [error, setError] = useState<string>();

  const modelRequestId = useRef(0);
  const layerRequestId = useRef(0);
  const detailRequestId = useRef(0);
  const previousIntentRef = useRef<PreviousSubgraphIntent | undefined>(undefined);
  const pendingNavigationRef = useRef<PendingNavigation | undefined>(undefined);

  useEffect(() => {
    Promise.all([api.models(), api.coverage(), api.overview(), api.proofSummary(), api.teachingFlow(), api.caseStudies()])
      .then(([modelRows, coverageData, overviewData, proofData, flowData, caseStudyData]) => {
        setModels(modelRows);
        setCoverage(coverageData);
        setOverview(overviewData);
        setProofSummary(proofData);
        setTeachingFlow(flowData);
        setCaseStudies(caseStudyData);
        if (modelRows.length) setSelectedModel(modelRows[0].id);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;

    const requestId = ++modelRequestId.current;
    setSelectedLayer(undefined);
    setSelectedNode(undefined);
    setSubgraphDetail(undefined);
    setSubgraphs([]);
    setSubgraphsContext(undefined);
    setModelDetail(undefined);
    setDiagnosis(undefined);

    Promise.all([api.model(selectedModel), api.layers(selectedModel), api.diagnosis(selectedModel)])
      .then(([detail, layerRows, diagnosisData]) => {
        if (requestId !== modelRequestId.current) return;

        setModelDetail(detail);
        setLayers(layerRows);
        setDiagnosis(diagnosisData);

        if (!layerRows.length) {
          setSelectedLayer(undefined);
          return;
        }

        const pending = pendingNavigationRef.current;
        const requestedLayer = pending?.model === selectedModel ? pending.layer : undefined;
        const nextLayer =
          requestedLayer !== undefined && layerRows.some((row) => row.layer_index === requestedLayer)
            ? requestedLayer
            : layerRows[0].layer_index;
        setSelectedLayer(nextLayer);
      })
      .catch((err) => {
        if (requestId !== modelRequestId.current) return;
        setError(String(err));
      });
  }, [selectedModel]);

  useEffect(() => {
    if (!selectedModel || selectedLayer === undefined) return;

    const requestId = ++layerRequestId.current;
    const modelAtRequest = selectedModel;
    const layerAtRequest = selectedLayer;

    setSelectedNode(undefined);
    setSubgraphDetail(undefined);
    setSubgraphs([]);
    setSubgraphsContext(undefined);

    api.subgraphs(modelAtRequest, layerAtRequest)
      .then((rows) => {
        if (requestId !== layerRequestId.current) return;
        if (selectedModel !== modelAtRequest || selectedLayer !== layerAtRequest) return;

        setSubgraphs(rows);
        setSubgraphsContext({ model: modelAtRequest, layer: layerAtRequest });

        const pending = pendingNavigationRef.current;
        const exactPendingNode = pending?.model === modelAtRequest && pending.layer === layerAtRequest ? pending.nodeSlug : undefined;
        const nextNode = chooseDefaultSubgraph(rows, previousIntentRef.current, exactPendingNode);

        previousIntentRef.current = undefined;
        if (pending?.model === modelAtRequest && pending.layer === layerAtRequest) {
          pendingNavigationRef.current = undefined;
        }

        setSelectedNode(nextNode);
      })
      .catch((err) => {
        if (requestId !== layerRequestId.current) return;
        setError(String(err));
      });
  }, [selectedModel, selectedLayer]);

  useEffect(() => {
    if (!selectedModel || selectedLayer === undefined || !selectedNode) {
      setSubgraphDetail(undefined);
      return;
    }

    // Critical guard: do not request details until the subgraph list currently
    // loaded for this exact model/layer contains the selected node. This prevents
    // stale calls such as /layers/1/subgraphs/12_layer_0_feed_forward.
    if (!sameContext(subgraphsContext, selectedModel, selectedLayer) || !subgraphBelongsToLoadedLayer(subgraphs, selectedNode)) {
      setSubgraphDetail(undefined);
      return;
    }

    const requestId = ++detailRequestId.current;
    const modelAtRequest = selectedModel;
    const layerAtRequest = selectedLayer;
    const nodeAtRequest = selectedNode;

    setSubgraphDetail(undefined);

    api.subgraph(modelAtRequest, layerAtRequest, nodeAtRequest)
      .then((detail) => {
        if (requestId !== detailRequestId.current) return;
        if (selectedModel !== modelAtRequest || selectedLayer !== layerAtRequest || selectedNode !== nodeAtRequest) return;
        setSubgraphDetail(detail);
      })
      .catch((err) => {
        if (requestId !== detailRequestId.current) return;
        setError(String(err));
      });
  }, [selectedModel, selectedLayer, selectedNode, subgraphs, subgraphsContext]);

  const activeLayer = useMemo(() => layers.find((layer) => layer.layer_index === selectedLayer), [layers, selectedLayer]);

  function handleSelectModel(modelId: string) {
    setActiveView('models');
    if (modelId === selectedModel) return;
    previousIntentRef.current = undefined;
    pendingNavigationRef.current = undefined;
    setSelectedModel(modelId);
  }

  function handleSelectLayer(layer: number) {
    if (layer === selectedLayer) return;
    previousIntentRef.current = makePreviousSubgraphIntent(subgraphs, selectedNode);
    pendingNavigationRef.current = selectedModel ? { model: selectedModel, layer } : undefined;
    setSelectedLayer(layer);
  }

  function handleSelectNode(node: string) {
    if (!subgraphBelongsToLoadedLayer(subgraphs, node)) return;
    setSelectedNode(node);
    setSubgraphDetail(undefined);
  }

  function handleSearchResult(match: SearchMatch) {
    setActiveView('models');
    previousIntentRef.current = undefined;
    pendingNavigationRef.current = { model: match.model_id, layer: match.layer, nodeSlug: match.node_slug };
    setSelectedNode(undefined);
    setSubgraphDetail(undefined);
    setSubgraphs([]);
    setSubgraphsContext(undefined);

    if (match.model_id !== selectedModel) {
      setSelectedModel(match.model_id);
    } else {
      setSelectedLayer(match.layer);
    }
  }

  return (
    <Layout
      models={models}
      activeView={activeView}
      selectedModel={selectedModel}
      onSelectView={setActiveView}
      onSelectModel={handleSelectModel}
      onSearchResult={handleSearchResult}
    >
      {error ? <div className="error-banner">{error}</div> : null}
      {activeView === 'dashboard' ? (
        <>
          <OverviewDashboard overview={overview} proof={proofSummary} onSelectView={setActiveView} />
          <CoverageDashboard coverage={coverage} />
        </>
      ) : null}
      {activeView === 'teaching-flow' ? <TeachingFlow overview={overview} flow={teachingFlow} proof={proofSummary} /> : null}
      {activeView === 'case-studies' ? <CaseStudies studies={caseStudies} /> : null}
      {activeView === 'reports' ? <ReportsPage /> : null}
      {activeView === 'models' ? (
        <>
          <CoverageDashboard coverage={coverage} />
          <ModelOverview detail={modelDetail} diagnosis={diagnosis} />
          <PipelineOverview detail={modelDetail} />
          <div className="workspace-grid">
            <LayerNavigator layers={layers} selectedLayer={selectedLayer} onSelect={handleSelectLayer} />
            <div className="analysis-column">
              {activeLayer ? (
                <section className="panel layer-summary-card">
                  <h2>Layer {activeLayer.layer_index}</h2>
                  <p>
                    {activeLayer.total_subgraphs} subgraphs / {activeLayer.safe} safe / {activeLayer.constrained} constrained / {activeLayer.blocked} blocked / {activeLayer.valid_plan_subgraphs} valid plan subgraphs
                  </p>
                </section>
              ) : null}
              <SubgraphDetail detail={subgraphDetail} />
              <SubgraphTable subgraphs={subgraphs} selectedNode={selectedNode} onSelect={handleSelectNode} />
            </div>
          </div>
        </>
      ) : null}
    </Layout>
  );
}
