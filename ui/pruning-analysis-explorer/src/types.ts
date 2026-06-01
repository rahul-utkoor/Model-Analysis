export type PruningClass = 'safe' | 'constrained' | 'blocked' | 'auxiliary' | 'unknown' | string;

export interface ModelSummary {
  id: string;
  display_name: string;
  num_layers: number;
  num_subgraphs: number;
  safe_candidates: number;
  mlp_safe_candidates: number;
  plans: number;
  valid_plans: number;
  status: string;
}

export interface CoverageResponse {
  coverage: Record<string, unknown>;
  table: Array<{
    model: string;
    status: string;
    layers: number;
    subgraphs: number;
    safe: number;
    plans: number;
    valid_plans: number;
  }>;
}

export interface LayerSummary {
  layer_index: number;
  total_subgraphs: number;
  safe: number;
  constrained: number;
  blocked: number;
  auxiliary: number;
  unknown: number;
  valid_plan_subgraphs: number;
  onnx_exported?: number;
}

export interface SubgraphSummary {
  ordinal: number;
  node_slug: string;
  display_name: string;
  semantic_category: string;
  pruning_class: PruningClass;
  plan_status: string;
  validation_status: string;
  onnx_status: string;
}

export interface SubgraphDetailResponse {
  analysis: Record<string, any>;
  explanation_md: string;
  artifact_paths: Record<string, { path: string; url: string }>;
}

export interface ModelDetail {
  id: string;
  model_name: string;
  model_summary: Record<string, any>;
  safe_opportunities?: any[];
  constrained_opportunities?: any[];
  blocked_structures?: any[];
  auxiliary_structures?: any[];
  missing_artifacts?: any[];
}

export interface SearchMatch {
  model: string;
  model_id: string;
  layer: number;
  node_slug: string;
  display_name: string;
  semantic_category: string;
  pruning_class: string;
  validation_status: string;
}

export interface FinalSummary {
  expected_plans: number;
  proven_plans: number;
  native_mlir_evidence: number;
  fallback: number;
  unsupported: number;
  partial: number;
  missing: number;
  failed: number;
}

export interface PipelineStep {
  id: string;
  title: string;
  summary: string;
  details: string[];
}

export interface OverviewResponse {
  title: string;
  subtitle: string;
  final_summary: FinalSummary;
  pipeline_steps: PipelineStep[];
  teaching_takeaways: string[];
  warnings: string[];
}

export interface ModelProofSummary {
  model_name: string;
  layers: number;
  expected_plans: number;
  proven_plans: number;
  ffn_proven: number;
  attention_value_proven: number;
  native_mlir_evidence: number;
  fallback: number;
  verdict: string;
}

export interface ProofSummaryResponse {
  models: ModelProofSummary[];
  aggregate: FinalSummary;
  warnings: string[];
}

export interface AxisRelation {
  source: string;
  target: string;
  relation: string;
}

export interface GraphVisual {
  type: 'graph';
  nodes: string[];
  edges: string[][];
}

export interface PipelineStage {
  id: string;
  title: string;
  kind: string;
  short: string;
  example?: string;
  visual?: GraphVisual;
  equations?: string[];
  relations?: AxisRelation[];
  pattern?: string;
  facts?: string[];
  proven: string;
  not_claimed?: string;
}

export interface PipelineExample {
  title: string;
  pattern: string;
  nodes: string[];
  dimensions: string[];
  edges: string[][];
  equations: string[];
  relations: AxisRelation[];
  facts: string[];
}

export interface PipelineFlowResponse {
  title: string;
  summary: string;
  aggregate: Pick<FinalSummary, 'expected_plans' | 'proven_plans' | 'native_mlir_evidence' | 'fallback'>;
  stages: PipelineStage[];
  examples: {
    ffn: PipelineExample;
    attention_value: PipelineExample;
    qk_blocker: PipelineExample;
  };
  warnings: string[];
}

export interface CaseStudy {
  id: string;
  title: string;
  summary: string;
  report_path: string;
  report_url: string;
  key_numbers: Record<string, string>;
  available: boolean;
}

export interface CaseStudiesResponse {
  case_studies: CaseStudy[];
}

export interface ReportTextResponse {
  path: string;
  format: string;
  text: string;
}
