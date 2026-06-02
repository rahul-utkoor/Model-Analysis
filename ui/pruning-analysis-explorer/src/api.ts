import type {
  CaseStudiesResponse,
  CoverageResponse,
  ArtifactBundle,
  ArtifactTextResponse,
  EvidenceArtifactMap,
  EvidenceTracesResponse,
  LayerSummary,
  ModelDetail,
  ModelSummary,
  OverviewResponse,
  PipelineFlowResponse,
  ProofSummaryResponse,
  ReportTextResponse,
  SearchMatch,
  SubgraphDetailResponse,
  SubgraphSummary,
} from './types';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  models: () => getJson<ModelSummary[]>('/api/models'),
  coverage: () => getJson<CoverageResponse>('/api/coverage'),
  overview: () => getJson<OverviewResponse>('/api/overview'),
  proofSummary: () => getJson<ProofSummaryResponse>('/api/proof-summary'),
  pipelineFlow: () => getJson<PipelineFlowResponse>('/api/pipeline-flow'),
  evidenceTraces: () => getJson<EvidenceTracesResponse>('/api/evidence-traces'),
  evidenceArtifactMap: () => getJson<EvidenceArtifactMap>('/api/evidence-artifact-map'),
  artifactBundle: (model: string, layer: number, subgraph: string) => {
    const params = new URLSearchParams({ model, layer: String(layer), subgraph });
    return getJson<ArtifactBundle>(`/api/artifact-bundle?${params.toString()}`);
  },
  artifactText: (path: string) => getJson<ArtifactTextResponse>(`/api/artifact-text?path=${encodeURIComponent(path)}`),
  caseStudies: () => getJson<CaseStudiesResponse>('/api/case-studies'),
  reportText: (path: string) => getJson<ReportTextResponse>(`/api/report-text?path=${encodeURIComponent(path)}`),
  model: (modelId: string) => getJson<ModelDetail>(`/api/models/${encodeURIComponent(modelId)}`),
  layers: (modelId: string) => getJson<LayerSummary[]>(`/api/models/${encodeURIComponent(modelId)}/layers`),
  subgraphs: (modelId: string, layer: number) => getJson<SubgraphSummary[]>(`/api/models/${encodeURIComponent(modelId)}/layers/${layer}/subgraphs`),
  subgraph: (modelId: string, layer: number, node: string) =>
    getJson<SubgraphDetailResponse>(`/api/models/${encodeURIComponent(modelId)}/layers/${layer}/subgraphs/${encodeURIComponent(node)}`),
  ranking: (modelId: string) => getJson<Record<string, any>>(`/api/models/${encodeURIComponent(modelId)}/ranking`),
  plans: (modelId: string) => getJson<Record<string, any>>(`/api/models/${encodeURIComponent(modelId)}/plans`),
  validation: (modelId: string) => getJson<Record<string, any>>(`/api/models/${encodeURIComponent(modelId)}/validation`),
  diagnosis: (modelId: string) => getJson<Record<string, any>>(`/api/models/${encodeURIComponent(modelId)}/diagnosis`),
  status: (modelId: string) => getJson<Record<string, any>>(`/api/models/${encodeURIComponent(modelId)}/status`),
  search: (query: string, modelId?: string, layer?: number) => {
    const params = new URLSearchParams({ q: query });
    if (modelId) params.set('model', modelId);
    if (layer !== undefined) params.set('layer', String(layer));
    return getJson<{ matches: SearchMatch[] }>(`/api/search?${params.toString()}`);
  }
};
