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
