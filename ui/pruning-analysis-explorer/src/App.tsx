import { useEffect, useMemo, useState } from 'react';
import { api } from './api';
import { CoverageDashboard } from './components/CoverageDashboard';
import { LayerNavigator } from './components/LayerNavigator';
import { Layout } from './components/Layout';
import { ModelOverview } from './components/ModelOverview';
import { PipelineOverview } from './components/PipelineOverview';
import { SubgraphDetail } from './components/SubgraphDetail';
import { SubgraphTable } from './components/SubgraphTable';
import type { CoverageResponse, LayerSummary, ModelDetail, ModelSummary, SearchMatch, SubgraphDetailResponse, SubgraphSummary } from './types';

export default function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [coverage, setCoverage] = useState<CoverageResponse>();
  const [selectedModel, setSelectedModel] = useState<string>();
  const [modelDetail, setModelDetail] = useState<ModelDetail>();
  const [diagnosis, setDiagnosis] = useState<Record<string, any>>();
  const [layers, setLayers] = useState<LayerSummary[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<number>();
  const [subgraphs, setSubgraphs] = useState<SubgraphSummary[]>([]);
  const [selectedNode, setSelectedNode] = useState<string>();
  const [subgraphDetail, setSubgraphDetail] = useState<SubgraphDetailResponse>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    Promise.all([api.models(), api.coverage()])
      .then(([modelRows, coverageData]) => {
        setModels(modelRows);
        setCoverage(coverageData);
        if (modelRows.length) setSelectedModel(modelRows[0].id);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    setSelectedLayer(undefined);
    setSelectedNode(undefined);
    setSubgraphDetail(undefined);
    Promise.all([api.model(selectedModel), api.layers(selectedModel), api.diagnosis(selectedModel)])
      .then(([detail, layerRows, diagnosisData]) => {
        setModelDetail(detail);
        setLayers(layerRows);
        setDiagnosis(diagnosisData);
        if (layerRows.length) setSelectedLayer(layerRows[0].layer_index);
      })
      .catch((err) => setError(String(err)));
  }, [selectedModel]);

  useEffect(() => {
    if (!selectedModel || selectedLayer === undefined) return;
    setSelectedNode(undefined);
    setSubgraphDetail(undefined);
    api.subgraphs(selectedModel, selectedLayer)
      .then((rows) => {
        setSubgraphs(rows);
        const preferred = rows.find((row) => row.validation_status === 'valid') ?? rows[0];
        if (preferred) setSelectedNode(preferred.node_slug);
      })
      .catch((err) => setError(String(err)));
  }, [selectedModel, selectedLayer]);

  useEffect(() => {
    if (!selectedModel || selectedLayer === undefined || !selectedNode) return;
    api.subgraph(selectedModel, selectedLayer, selectedNode)
      .then(setSubgraphDetail)
      .catch((err) => setError(String(err)));
  }, [selectedModel, selectedLayer, selectedNode]);

  const activeLayer = useMemo(() => layers.find((layer) => layer.layer_index === selectedLayer), [layers, selectedLayer]);

  function handleSearchResult(match: SearchMatch) {
    setSelectedModel(match.model_id);
    setSelectedLayer(match.layer);
    setSelectedNode(match.node_slug);
  }

  return (
    <Layout models={models} selectedModel={selectedModel} onSelectModel={setSelectedModel} onSearchResult={handleSearchResult}>
      {error ? <div className="error-banner">{error}</div> : null}
      <CoverageDashboard coverage={coverage} />
      <ModelOverview detail={modelDetail} diagnosis={diagnosis} />
      <PipelineOverview detail={modelDetail} />
      <div className="workspace-grid">
        <LayerNavigator layers={layers} selectedLayer={selectedLayer} onSelect={setSelectedLayer} />
        <div className="middle-column">
          {activeLayer ? (
            <section className="panel layer-summary-card">
              <h2>Layer {activeLayer.layer_index}</h2>
              <p>
                {activeLayer.total_subgraphs} subgraphs / {activeLayer.safe} safe / {activeLayer.constrained} constrained / {activeLayer.blocked} blocked / {activeLayer.valid_plan_subgraphs} valid plan subgraphs
              </p>
            </section>
          ) : null}
          <SubgraphTable subgraphs={subgraphs} selectedNode={selectedNode} onSelect={setSelectedNode} />
        </div>
        <SubgraphDetail detail={subgraphDetail} />
      </div>
    </Layout>
  );
}
