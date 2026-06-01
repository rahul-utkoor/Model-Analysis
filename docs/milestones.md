# Milestones

## Milestone 1: Repository Infrastructure

Status: complete.

Implemented:

- Git-backed Python project scaffold
- Model registry in `configs/models.yaml`
- Download scripts for Hugging Face models
- ONNX export script
- Lightweight inspection script
- Initial tests for paths and registry
- README setup and usage instructions

Primary commands:

```bash
python scripts/download_models.py --model all
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/inspect_model.py --model bert-base-uncased
```

## Milestone 2: Structural Inventory Generation

Status: complete.

Implemented:

- PyTorch structural inventory summaries
- ONNX graph summaries
- Conservative pruning hints
- Report writing helpers
- Structural inventory CLI
- Tests for inventory, ONNX analysis, and reporting

Primary commands:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model all --require-onnx
```

## Milestone 3: Pruning Dependency Graph Construction

Status: complete.

Implemented:

- Prunable unit and dependency edge data model
- Dependency graph construction from PyTorch structural summaries
- Optional ONNX graph augmentation
- Coupled group and independent-unit detection
- Ambiguous/manual-review item detection
- Dependency graph analyzer
- Dependency graph CLI
- Tests for dependency graph and analyzer behavior

Primary commands:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model all --require-onnx --verbose
```

## Milestone 4: Pruning Action Simulation and Dimension Propagation

Status: complete.

Implemented:

- Dry-run pruning action schema
- Propagation step and pruning plan schema
- Conservative propagation engine over dependency graphs
- Edge-specific semantics for QKV, MLP, residual, normalization, embedding, feeds, shape, and propagation-only edges
- Candidate action generation
- Manual and candidate-action CLI scripts
- Markdown and JSON pruning plan reports
- Tests for action serialization, propagation, candidate generation, and reporting

Primary commands:

```bash
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
python scripts/simulate_pruning_action.py --model bert-base-uncased --target-unit <unit_id> --dim out_features --indices 0,1,2,3 --verbose
```

## Milestone 5: PyTorch-to-ONNX Correspondence and Shape Evidence

Status: complete.

Implemented:

- Parameter-to-initializer evidence
- Module-to-node correspondence reports
- Static ONNX tensor and node shape evidence
- Dependency graph validation with correspondence/shape support
- Evidence-enriched pruning action simulation
- Correspondence CLI and docs

Primary commands:

```bash
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
python scripts/simulate_pruning_action.py --model bert-base-uncased --target-unit <unit_id> --dim out_features --indices 0,1,2,3 --use-evidence --verbose
```

## Milestone 6: Proposed Next Step

Status: complete.

Implemented:

- Reversible PyTorch `nn.Linear` out_features pruning
- Reversible PyTorch `nn.Linear` in_features pruning
- Linear-only pruning plan executor
- Dry-run and only-target execution modes
- Pruning execution reports
- Structural pruning diffs
- Rollback manifests
- Tests for Linear surgery, execution, diffs, and rollback

Primary commands:

```bash
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.attention.self.query --dim out_features --indices 0,1,2,3 --only-target --dry-run --verbose
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.attention.self.query --dim out_features --indices 0,1,2,3 --only-target --allow-ambiguous --verbose
```

## Milestone 7: Paired Linear Structural Repair and Forward Smoke Validation

Status: complete.

Implemented:

- Paired Linear repair plan data model
- MLP expansion/projection repair detection from explicit `mlp_hidden_coupling`
- Atomic source `out_features` plus target `in_features` Linear repair
- Repair plan and repair transaction reports
- Forward smoke validation module and standalone CLI
- Optional before/after smoke validation during pruning execution
- Tests for repair detection, paired pruning atomicity, and smoke validation

Primary commands:

```bash
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense --dim out_features --indices 0,1,2,3 --repair-pairs --write-repair-plan-only --allow-ambiguous --verbose
python scripts/execute_pruning_plan.py --model bert-base-uncased --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense --dim out_features --indices 0,1,2,3 --repair-pairs --dry-run --allow-ambiguous --verbose
python scripts/run_forward_smoke_test.py --model bert-base-uncased --device cpu --verbose
```

## Milestone 8: BERT MLP Block-Level Executable Pruning

Status: complete.

Implemented:

- BERT-style MLP block target detection
- Architecture-specific prune specs for intermediate-dimension pruning
- Executable `intermediate.dense` `out_features` plus `output.dense` `in_features` pruning
- Block-level reports, diffs, validation reports, and rollback manifests
- Target-list CLI and BERT MLP pruning CLI
- Tiny BERT-like tests for detection, dry-run, execution, forward pass, and rejection cases

Primary commands:

```bash
python scripts/list_bert_mlp_targets.py --model bert-base-uncased
python scripts/prune_bert_mlp_block.py --model bert-base-uncased --layer 0 --indices 0,1,2,3 --dry-run --smoke-test-before --verbose
python scripts/prune_bert_mlp_block.py --model bert-base-uncased --layer 0 --indices 0,1,2,3 --smoke-test-before --smoke-test-after --verbose
```

## Milestone 9: Compiler-Style Pruning Opportunity Analysis

Status: complete.

Implemented:

- Pruning opportunity IR for dimensions, propagation constraints, opportunities, and model pruning maps
- Dimension extraction from dependency graphs
- Constraint extraction from dependency edges and optional validation evidence
- Opportunity inference for local Linear outputs, MLP intermediate dimensions, attention QKV structures, embeddings, ONNX MatMul/Gemm candidates, and blocked residual hidden-size regions
- Structural risk maps
- Per-model pruning map CLI
- Cross-model pruning map comparison CLI
- Tests for opportunity extraction, constraint typing, risk detection, and comparison matrices

Primary commands:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_pruning_map.py --model all --verbose
python scripts/compare_pruning_maps.py --models all
```

## Milestone 10: Dimension Variable IR and Symbolic Propagation Constraints

Status: complete.

Implemented:

- Symbolic Dimension IR data model
- Dimension variables from model pruning maps
- Pruning-index variables for symbolic pruning selections
- Constraint equations for MLP, QKV, residual, LayerNorm, tied-parameter, reshape, and unknown mappings
- Union-find equivalence classes for equality, same-index, and tied dimensions
- Blocked-dimension and unresolved-constraint tracking
- MLIR-like `.pir` textual dump
- Dimension IR build and comparison CLIs
- Tests for Dimension IR construction, text dumps, and cross-model comparison

Primary commands:

```bash
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model all --verbose
python scripts/compare_dimension_irs.py --models all
```

## Milestone 11: Proposed Next Step

Status: complete.

Implemented:

- Symbolic pruning request model
- Constraint adjacency and slicing helpers over Dimension IR
- Static legality checker for symbolic/concrete pruning requests
- Constraint satisfaction classification
- Forward and backward propagation slices
- Minimal structural repair-set computation
- Blocked-region explanations
- Dimension listing, legality check, and blocked-region CLIs
- Tests for graph traversal, legality statuses, repair sets, invalid requests, and blocked-region mitigation

Primary commands:

```bash
python scripts/list_pruning_dimensions.py --model bert-base-uncased --contains intermediate.dense
python scripts/check_pruning_legality.py --model bert-base-uncased --dimension-var <dimension_var_id> --count 4 --verbose
python scripts/explain_blocked_regions.py --model bert-base-uncased
```

## Milestone 12: Demo Track and Research Walkthrough

Status: complete.

Implemented:

- Guided demo documentation under `demos/`
- Milestone-by-milestone walkthroughs for Milestones 1 through 11
- Full research pipeline walkthrough
- Research glossary for pruning/compiler terms
- Demo shell scripts for the main analysis path
- Formal demo-track documentation under `docs/demo_track.md`
- Lightweight tests that verify demo documentation entry points

Primary commands:

```bash
bash demo_scripts/run_demo_01_setup_check.sh
PYTHON=python MODEL=bert-base-uncased bash demo_scripts/run_full_analysis_pipeline.sh
```

The demo track makes Tensor IR, Structural Region Tree, Region-Aware Dimension IR, region-aware legality analysis, pruning maps, and Dimension IR the main learning path. Executable pruning modules are documented as optional experimental backends.

## Milestone 13: k-Node and Join-Aware ONNX Subgraph Structural Analysis

Status: complete.

Implemented:

- Directed ONNX path enumeration for lengths one through five
- Join-centered subgraph extraction for `Add`, `Sum`, and `Concat` branch merges
- Bias-add versus residual-add classification
- Residual-like join and post-join normalization evidence
- Local pattern classification for projection, MLP, attention, shape-transform, join, and residual structures
- Report-level pruning/dimension evidence
- Cross-model subgraph comparison
- Guided demo documentation and synthetic tests

Primary commands:

```bash
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_subgraphs.py --model all --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/compare_subgraphs.py --models all
```

This pass performs structural analysis over saved ONNX summaries only. Residual and join evidence is intended to refine future pruning maps and Dimension IR constraints.

## Milestone 14: DAG Motif and Multi-Join Region Subgraph Analysis

Status: complete.

Implemented:

- Fork-region detection for producer fanout
- Bounded reconvergence search and diamond-region extraction
- Join-fork-join detection for multi-join motifs such as `A,B -> C -> D,E -> F`
- Canonical op-type patterns for multi-branch regions
- Suggested fanout, branch compatibility, residual, and reshape constraints
- DAG region report and cross-model comparison CLIs
- Demo documentation and synthetic exact-motif tests

Primary commands:

```bash
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/analyze_dag_regions.py --model all --max-branch-depth 4 --verbose
python scripts/compare_dag_regions.py --models all
```

This pass captures fork, join, diamond, and join-fork-join regions but does not modify models.

## Milestone 15: Netron-Visualizable ONNX Subgraph Export

Status: complete.

Implemented:

- Path, join, and DAG-region record loading and filtered selection
- Standalone ONNX fragment extraction preserving nodes, boundaries, initializers, shapes where available, opsets, and provenance metadata
- Exact/filter-driven export CLI
- Curated Netron demo export CLI
- Export manifests and Netron command indexes that list the original full ONNX graph as the comparison baseline
- Demo documentation and synthetic ONNX extractor tests

Primary commands:

```bash
python scripts/export_demo_subgraphs.py --model bert-base-uncased --max-per-category 3 --verbose
python scripts/export_subgraph_onnx.py --model bert-base-uncased --kind dag_region --max-exports 5 --verbose
```

Extracted ONNX subgraphs are visualization artifacts and do not modify the source model.

### Static-Shape ONNX Visualization Export

The static-shape exporter is a Netron visualization aid adjacent to Milestone 15. It writes `data/models/onnx_static/<model>/model.static.onnx`, reconstructs Hugging Face inputs by keyword through ONNX wrapper modules, and records dropped tokenizer inputs and fixed shapes in metadata. Dynamic ONNX exports remain the analysis input.

```bash
./conda-env/bin/python scripts/export_static_shape_onnx.py --model bert-base-uncased --seq-len max --batch-size 1 --opset 17 --device cpu
./conda-env/bin/python scripts/export_static_shape_onnx.py --model all --seq-len 128 --batch-size 1 --opset 17 --device cpu --continue-on-error
```

## Milestone 16: Frontend-Independent Tensor Graph IR

Status: complete.

Implemented:

- Frontend-independent `TensorValue`, `TensorOp`, and `TensorGraph` data model
- Conservative canonical operation typing and region hints
- ONNX-summary importer as the first Tensor IR frontend
- Producer/consumer, fork, and join construction
- Markdown, JSON, statistics, and readable `.tir` dumps
- Cross-model Tensor IR comparison
- Guided demo and synthetic tests

Primary commands:

```bash
python scripts/build_tensor_ir.py --model bert-base-uncased --verbose
python scripts/build_tensor_ir.py --model all --verbose
python scripts/compare_tensor_ir.py --models all
```

ONNX is only one frontend representation. Tensor IR is the frontend-independent substrate intended for Structural Region Tree and pruning-propagation analysis.

## Milestone 17: Structural Region Tree over Tensor IR

Status: complete.

Implemented:

- `StructuralRegion`, `StructuralRegionInterface`, and `StructuralRegionTree` data model
- Tensor IR adjacency and region-boundary computation
- Primitive leaf construction and conservative semantic region detection
- Projection, activation, normalization, axis-transform, residual-merge, feed-forward, attention-skeleton, fork, and join candidates
- Priority-based overlap resolution with primitive leaves retained
- Preliminary pruning/propagation region interfaces
- Readable `.srtree` textual dumps and cross-model tree comparison
- Guided demo and synthetic tests

Primary commands:

```bash
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model all --verbose
python scripts/compare_structural_region_trees.py --models all
```

This builds a compiler-inspired structural hierarchy over Tensor IR. It does not modify models, execute pruning, rewrite ONNX, or evaluate quality.

## Milestone 18: Region-Aware Dimension IR

Status: complete.

Implemented:

- `RegionDimensionVariable`, `RegionConstraintEquation`, `RegionDimensionEquivalenceClass`, and `RegionDimensionIR` data model
- Symbolic dimensions derived from projection, feed-forward, residual merge, normalization, axis-transform, activation, attention, fork, and join region interfaces
- Region-scoped constraints for same-index MLP propagation, residual/normalization equality, transform mapping, attention-axis uncertainty, fanout, and join compatibility
- Equivalence-class construction over explicit equality-like region constraints
- Readable `.rdim` textual dump and cross-model comparison
- Guided demo and synthetic tests

Primary commands:

```bash
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python scripts/build_region_dimension_ir.py --model all --verbose
python scripts/compare_region_dimension_ir.py --models all
```

Region-Aware Dimension IR refines symbolic dimension reasoning using semantic structural regions. It does not modify models, execute pruning, rewrite ONNX, or evaluate quality.

## Milestone 19: Region-Aware Pruning Propagation Analysis

Status: complete.

Implemented:

- `RegionPruningRequest`, `RegionConstraintSatisfaction`, `RegionPropagationSlice`, `RegionRepairSetItem`, and `RegionLegalityCheckResult` model
- Inferred-direction constraint adjacency and forward/backward slice extraction over RegionDimensionIR
- Static legality classification for semantic-region pruning requests
- Minimal repair obligations for MLP, linear-bias, fanout, residual, normalization, axis-transform, attention, and join constraints
- Protected/blocked dimension explanations with mitigations
- Region dimension listing, legality-check, and blocked-dimension CLIs
- Guided demo and synthetic tests

Primary commands:

```bash
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 10
python scripts/explain_region_blocked_dimensions.py --model bert-base-uncased
python scripts/check_region_pruning_legality.py --model bert-base-uncased --dimension-var <region_dimension_var_id> --count 4 --verbose
```

This milestone performs static region-aware legality analysis and does not modify models, execute pruning, rewrite ONNX, or evaluate quality.

## Milestone 20: Semantic Fusion for Activation and Feed-Forward Regions

Status: complete.

Implemented:

- Tensor IR semantic-fusion report for decomposed GELU activation patterns
- Graph-structured `Div/Mul -> Erf -> Add -> Mul -> Mul` recovery with confidence labels
- Projection/GELU/projection feed-forward fusion detection
- Structural Region Tree integration with fused activation and feed-forward candidates
- Exclusion of proved GELU-internal additions from residual-merge blocking candidates
- Region Dimension IR evidence carrying fused projection metadata
- CLI reports, demo workflow, and synthetic regression tests

Primary commands:

```bash
python scripts/analyze_semantic_fusion.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 20
```

Semantic fusion recovers high-level activation/feed-forward regions without modifying models, rewriting ONNX, executing pruning, or evaluating quality.

## Local Region Tree Browsing and Export Tools

Status: complete.

Added after Milestone 20:

- Lazy API-backed Structural Region Tree browser
- Abstract structure collector and focused structure browser
- Optional graph-heavy structure viewer for exploratory use
- Lazy region-tree export/viewer for static HTTP serving
- MindNode tab-indented and OPML outline exporter
- Ignore rules for generated `viewer_data/`, abstract-structure reports, and MindNode outlines

Primary commands:

```bash
python abstract_structure_collector.py --model bert-base-uncased --write
python region_structure_api_server.py --model bert-base-uncased --port 8765
./conda-env/bin/python tools/export_region_tree_mindnode.py --model bert-base-uncased --label-mode semantic --include-counts --max-depth 3
```

These are visualization, browsing, and export tools only. They do not modify models, pruning logic, Tensor IR, RegionDimensionIR, or ONNX artifacts.

## Milestone 21: Step-by-Step Dataflow Control-Tree Construction Trace

Status: complete.

Implemented:

- Control-tree trace data model for active TensorOp/region graph snapshots
- Working graph collapse operation that redirects dataflow edges through newly created abstract regions
- Ordered trace candidates from semantic fusion, Structural Region Tree regions, or detector fallback
- JSON, Markdown, textual `.ctrace`, DOT graph, and MindNode outline outputs
- CLI workflow and synthetic tests for initialization, collapse, skip, DOT, text, and outline behavior

Primary commands:

```bash
python scripts/build_control_tree_trace.py --model bert-base-uncased --format all --max-dot-steps 20 --verbose
python tools/export_control_tree_trace_mindnode.py --model bert-base-uncased
```

This is an explanatory structural-analysis trace over Tensor IR. It does not modify models, execute pruning, rewrite ONNX, or evaluate quality.

## Milestone 22: Lightweight Stepwise Control-Tree Viewer

Status: complete.

Implemented:

- Standard-library lazy API server for control-tree trace browsing
- Standalone HTML viewer with step filters, paging, navigation, details, and local SVG collapse graphs
- Local graph extraction that avoids sending full step snapshots to the browser
- Pure helper tests for step summaries, filters, pagination, local graphs, skip steps, and next/previous navigation

Primary commands:

```bash
python scripts/build_control_tree_trace.py --model bert-base-uncased --format all --max-dot-steps 20 --verbose
python tools/control_tree_trace_api_server.py --model bert-base-uncased --port 8766
```

Open `http://127.0.0.1:8766/`. This viewer is an explanatory structural-analysis browser and does not modify models, execute pruning, rewrite ONNX, or evaluate quality.

## Milestone 23: Ordered Dataflow Control-Tree Browser

Status: complete.

Implemented:

- Standard-library lazy API server for final Structural Region Tree browsing
- Ordered child summaries using Tensor IR/source operation order rather than alphabetical region type
- Human-readable region and primitive TensorOp labels with pruning-role and source-order context
- Expandable standalone HTML tree browser with breadcrumb, search, quick filters, lazy children, and ordered leaf reveal
- Synthetic tests for ordering, labels, paths, leaf extraction, search, teaching text, and dimension summaries

Primary command:

```bash
python tools/ordered_control_tree_api_server.py --model bert-base-uncased --port 8767
```

Open `http://127.0.0.1:8767/`. This is visualization/reporting only and does not modify models, execute pruning, rewrite ONNX, export ONNX, or evaluate quality.

## Learner Abstract Node Expansion Reports

Status: complete.

Implemented:

- Learner-facing abstract-node expansion report generator
- Main compute and grouped shape/mask report views
- Clear separation between immediate expansion and recursive primitive leaf evidence
- Section-level expansion for model, embeddings, encoder layers, prediction head, and auxiliary flow
- ShapeMotifRegion grouping for predicate, mask, scalar, and axis plumbing
- Markdown/PDF/JSON outputs and synthetic tests

Primary commands:

```bash
./conda-env/bin/python tools/export_abstract_node_expansion_report.py --model bert-base-uncased --view main --max-leaf-names 30
./conda-env/bin/python tools/export_abstract_node_expansion_report.py --model bert-base-uncased --view shape --max-leaf-names 30
```

This is reporting/visualization only and does not modify models, execute pruning, rewrite ONNX, export ONNX, or evaluate quality.

## Milestone 25: Region-Level Pruning Propagation Semantics

Status: complete.

Implemented:

- Region Pruning Semantics IR
- JSON, Markdown, and `.rpsem` text reports
- Build, explain, and compare CLIs
- Conservative semantics for projection, GELU, feed-forward, attention, residual, LayerNorm, and shape/mask regions
- Attention-internal score/context MatMul and mask-add overrides to avoid treating dataflow contractions or mask application as directly prunable projections/residual merges
- Explicit `source_region_type` and `semantic_category` fields so reports distinguish structural classification from pruning-semantics interpretation
- Tight attention-mask categories so Axis/Fork/Join mask plumbing is not counted as true `attention_mask_add`
- Synthetic tests for opportunities, repairs, blockers, text dumps, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/build_region_pruning_semantics.py --model bert-base-uncased --verbose
./conda-env/bin/python scripts/explain_region_pruning_semantics.py --model bert-base-uncased --contains "Feed Forward" --limit 5
./conda-env/bin/python scripts/explain_region_pruning_semantics.py --model bert-base-uncased --blocked-only --limit 10
./conda-env/bin/python scripts/compare_region_pruning_semantics.py --models all
```

This is static reporting/analysis only and does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

## Milestone 25.4: Pruning-Relevant Op Semantics Annotation

Status: complete.

Implemented:

- Op Semantics IR over primitive Tensor IR operations
- JSON, Markdown, and `.opsem` text reports
- Build, explain, and compare CLIs
- Conservative op semantics for learned projections, bias adds, embeddings, GELU pieces, LayerNorm, residual Adds, attention score/context MatMuls, attention mask Adds/selects, axis transforms, metadata flow, and unknown ops
- Synthetic tests for classifier behavior, text dumps, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/build_op_semantics.py --model bert-base-uncased --verbose
./conda-env/bin/python scripts/explain_op_semantics.py --model bert-base-uncased --semantic-kind attention_score_matmul --limit 5
./conda-env/bin/python scripts/explain_op_semantics.py --model bert-base-uncased --category parameterized_projection --limit 10
./conda-env/bin/python scripts/compare_op_semantics.py --models all
```

This is static reporting/analysis only and does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

## Milestone 26: Region Pruning Opportunity Ranking

Status: complete.

Implemented:

- Pruning Opportunity Ranking IR over Region Pruning Semantics and optional Op Semantics
- JSON, Markdown, and `.rank` text reports
- Build, explain, and compare CLIs
- Deterministic conservative ranking into safe, constrained, blocked, auxiliary, and unknown candidates
- Op-semantics evidence attachment, missing-evidence warnings, and op-region disagreement warnings
- Synthetic tests for ranking policy, text reports, Markdown auxiliary suppression, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/rank_pruning_opportunities.py --model bert-base-uncased --verbose
./conda-env/bin/python scripts/explain_pruning_opportunity.py --model bert-base-uncased --class safe --limit 20
./conda-env/bin/python scripts/explain_pruning_opportunity.py --model bert-base-uncased --contains "Attention Score MatMul"
./conda-env/bin/python scripts/compare_pruning_opportunities.py --models all
```

This is static reporting/analysis only and does not modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

## Milestone 27: Candidate Plan Synthesis for Safe FFN Pruning

Status: complete.

Implemented:

- Symbolic Pruning Plan IR for safe feed-forward `intermediate_dim` candidates
- JSON, Markdown, and `.plan` text reports
- Build, explain, and compare CLIs
- Plan actions for intermediate projection output pruning, intermediate bias repair, GELU propagation, FFN output projection input repair, hidden-dimension preservation, and forbidden residual/LayerNorm hidden pruning
- Synthetic tests for plan synthesis, text reports, incomplete-evidence handling, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/synthesize_pruning_plans.py --model bert-base-uncased --verbose
./conda-env/bin/python scripts/explain_pruning_plan.py --model bert-base-uncased --status ready_symbolic --limit 20
./conda-env/bin/python scripts/explain_pruning_plan.py --model bert-base-uncased --contains "Layer 0 Feed Forward"
./conda-env/bin/python scripts/compare_pruning_plans.py --models all
```

This is static reporting/analysis only. Plans are parameterized by symbolic index sets; they do not choose concrete indices, modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

## Milestone 28: Pruning Plan Validation and Consistency Checking

Status: complete.

Implemented:

- Pruning Plan Validation IR over symbolic plans, ranking, region semantics, and op semantics
- JSON, Markdown, and `.pvalid` text reports
- Build, explain, and compare CLIs
- Static FFN plan checks for candidate safety, ready status, symbolic index set, required actions, op-semantics agreement, repair consistency, hidden-dimension preservation, forbidden actions, blockers, and unknown critical ops
- Synthetic tests for valid plans, missing actions, wrong op semantics, missing repairs, forbidden-action violations, text dumps, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/validate_pruning_plans.py --model bert-base-uncased --verbose
./conda-env/bin/python scripts/explain_pruning_plan_validation.py --model bert-base-uncased --status valid --limit 20
./conda-env/bin/python scripts/explain_pruning_plan_validation.py --model bert-base-uncased --contains "Layer 0 Feed Forward"
./conda-env/bin/python scripts/compare_pruning_plan_validation.py --models all
```

This is static reporting/analysis only. Validation does not choose concrete indices, modify models, execute pruning, rewrite ONNX, export ONNX, download models, or evaluate quality.

## Milestone 29: Encoder-Layer Subgraph Evidence and Validation Pack

Status: complete.

Implemented:

- Layer Subgraph Validation Pack IR for learner-facing encoder-layer nodes
- Per-node analysis folders with primitive ops, op semantics, region semantics, ranking, plan, validation, and explanation slices
- Best-effort ONNX visualization fragment export under `artifacts/layer_subgraphs/`
- JSON, Markdown, and text dump reports plus explain/compare CLIs
- Synthetic tests for selection, deduplication, learner ordering, classification, local slices, ONNX failure handling, Markdown, and comparison summaries

Primary commands:

```bash
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py --model bert-base-uncased --layer 0 --export-onnx --render-svg --verbose
./conda-env/bin/python scripts/explain_layer_subgraph_validation.py --model bert-base-uncased --layer 0 --contains "Feed Forward"
./conda-env/bin/python scripts/explain_layer_subgraph_validation.py --model bert-base-uncased --layer 0 --class safe
./conda-env/bin/python scripts/compare_layer_subgraph_validation.py --models bert-base-uncased --layer 0
```

This is static reporting/visualization only. ONNX fragments are evidence artifacts for Netron and are not treated as standalone full models.

## Milestone 30: Full-Model and Cross-Model Structured Analysis Reports

Status: complete.

Implemented:

- Full-model static analysis report tree under `reports/model_analysis_reports/<model>/`
- Per-layer polished reports and per-subgraph explanation slices
- Cross-model summary reports under `reports/model_analysis_reports/cross_model/`
- Visualization-only subgraph artifacts under `artifacts/model_analysis_subgraphs/`
- Learner-facing report fixes for attention skeleton, attention softmax, attention output projection, contextual LayerNorm names, and "why no plan" explanations

Primary commands:

```bash
./conda-env/bin/python scripts/build_full_model_analysis_report.py --model bert-base-uncased --layers all --export-onnx-subgraphs --render-svg --verbose
./conda-env/bin/python scripts/build_all_model_analysis_reports.py --models all --layers all --no-export-onnx-subgraphs --verbose
./conda-env/bin/python scripts/compare_model_analysis_reports.py --models all --verbose
./conda-env/bin/python scripts/explain_model_analysis_report.py --model bert-base-uncased --section feedforward
```

This is static reporting/visualization only. It does not choose pruning indices, modify models, execute pruning, rewrite full ONNX models, download models, or evaluate accuracy.

## Milestone 31: Cross-Model Artifact Completion and Static Coverage Study

Status: complete.

Implemented:

- Static pipeline model-status manifests under `reports/static_pipeline_status/`
- Cross-model static coverage study under `reports/static_coverage_study/`
- Orchestrator commands that mark existing stages, skipped prerequisites, failed builders, and not-applicable validation stages
- Optional downstream static analysis/report building when prerequisites already exist locally
- Synthetic tests for stage status, orchestrator continuation, coverage aggregation, and coverage Markdown

Primary commands:

```bash
./conda-env/bin/python scripts/build_static_pipeline_for_model.py --model bert-base-uncased --build-missing-analysis --build-layer-packs --verbose
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
./conda-env/bin/python scripts/report_static_pipeline_coverage.py --models all --verbose
./conda-env/bin/python scripts/explain_static_pipeline_status.py --model opt-125m
```

This is a static coverage/generalization study. It distinguishes complete support from partial/skipped support and records where new model-specific semantics are needed.

## Milestone 32: Cross-Model Rule-Gap Diagnosis and Generic FFN Matching

Status: complete.

Implemented:

- Rule-gap diagnosis reports for incomplete plans, invalid validations, missing FFN fusion, skipped layer grouping, and unknown op semantics
- Family detection for BERT, DistilBERT, OPT, GPT-2, ViT, and unknown models using op-semantic source paths
- Generic FFN evidence matching for `intermediate/output dense`, `ffn.lin1/lin2`, `fc1/fc2`, `mlp.fc1/fc2`, and `mlp.c_fc/c_proj`
- Generic plan synthesis/validation over expansion projection, activation propagation, contraction projection, hidden preservation, and fused Gemm bias actions
- Synthetic tests for cross-family matching, diagnosis, Markdown, and comparison reports

Primary commands:

```bash
./conda-env/bin/python scripts/diagnose_rule_gaps.py --models all --verbose
./conda-env/bin/python scripts/explain_rule_gap.py --model facebook/opt-125m
./conda-env/bin/python scripts/compare_rule_gaps.py --models all
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
./conda-env/bin/python scripts/report_static_pipeline_coverage.py --models all --verbose
```

This is static diagnosis/reporting only. Generic FFN matching improves symbolic plan evidence binding; it does not choose pruning indices, modify models, execute pruning, rewrite ONNX, download models, or evaluate accuracy.

## Milestone 33: Generalized FFN/MLP Region Fusion and Ranking

Status: complete.

Implemented:

- Generic MLP fusion from op semantics into synthesized `GenericMLPRegion` records
- Expansion/activation/contraction matching for DistilBERT `ffn.lin1/lin2`, ViT `mlp.fc1/fc2`, and GPT-2 `mlp.c_fc/c_proj`
- Region pruning semantics and ranking support for generic MLP safe/constrained candidates
- Static coverage and diagnosis summaries for recovered generic MLP regions and plans
- Regression checks preserving BERT and OPT valid FFN plans

Primary commands:

```bash
./conda-env/bin/python scripts/build_region_pruning_semantics.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/rank_pruning_opportunities.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/synthesize_pruning_plans.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/validate_pruning_plans.py --model distilbert-base-uncased --verbose
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
```

This is static analysis/reporting only. It recovers symbolic MLP pruning opportunities and validated plans where evidence is complete; it does not choose concrete indices or modify models.

## Milestone 34: Generic Transformer Block Layer Grouping and Subgraph Atlases

Status: complete.

Implemented:

- Generic transformer block grouping for BERT, DistilBERT, OPT, GPT-2, and ViT source-path families
- Layer/subgraph validation pack fallback from op semantics when BERT abstract-expansion records are unavailable
- Family-aware learner groups for attention projections, attention internals, residuals, LayerNorms, MLP blocks, MLP expansion, activation, and contraction
- Full-model report layer detection for encoder layers, decoder blocks, GPT-2 blocks, and ViT layers
- Validation summary aliases exposing both canonical `valid/warning/invalid/unknown` fields and backward-compatible `valid_plans/invalid_plans` fields

Primary commands:

```bash
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py --model facebook/opt-125m --layer 0 --export-onnx --render-svg --verbose
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py --model gpt2 --layer 0 --export-onnx --render-svg --verbose
./conda-env/bin/python scripts/build_full_model_analysis_report.py --model google/vit-base-patch16-224 --layers all --export-onnx-subgraphs --render-svg --verbose
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
```

This is static analysis/reporting/visualization only. ONNX subgraphs are evidence artifacts for inspection; they are not treated as standalone models and no pruning or model mutation is performed.

## Milestone 35: Interactive Static Analysis Explorer CLI

Status: complete.

Implemented:

- Read-only interactive CLI under `tools/interactive_analysis_explorer.py`
- Model, layer/block, and subgraph navigation over `reports/model_analysis_reports/`
- Subgraph inspection commands for explanations, primitive ops, op/region semantics, ranking, symbolic plans, validation checks, and ONNX paths
- Cross-model coverage comparison view
- Optional `--scripted` command sequence for smoke testing
- Stdlib-only helper module and tests for discovery, search, ONNX lookup, and validation summary aliases

Primary commands:

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py
./conda-env/bin/python tools/interactive_analysis_explorer.py --model bert-base-uncased --layer 0 --no-open
./conda-env/bin/python tools/interactive_analysis_explorer.py --model bert-base-uncased --layer 0 --no-open --scripted "nodes;subgraph Feed Forward;plan;validation;path;back;back"
```

This is read-only static analysis/reporting/visualization only. It does not regenerate reports, execute pruning, modify models, rewrite ONNX, download models, or evaluate accuracy.

## Milestone 36: React Web UI for Pruning Analysis Explorer

Status: complete.

Implemented:

- Python stdlib API server under `tools/analysis_ui_api_server.py`
- React + Vite + TypeScript frontend under `ui/pruning-analysis-explorer/`
- Cross-model coverage dashboard
- Model overview, pipeline overview, layer/block navigator, subgraph table, and subgraph detail panels
- Tabs for explanation, primitive ops, op/region semantics, ranking, symbolic plan, validation, and artifacts
- Safe artifact serving for ONNX/SVG/DOT/Markdown/JSON files
- Backend route tests and production frontend build

Primary commands:

```bash
cd ui/pruning-analysis-explorer
npm install
npm run build
cd ../..
./conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777
```

Open `http://127.0.0.1:8777/`.

This is read-only static analysis/reporting/visualization only. It does not execute pruning, choose indices, modify models, rewrite ONNX, download models, or evaluate accuracy.

## Milestone 37: Attention Value-Path Deadness Propagation

Status: complete.

Implemented:

- Static deadbranch propagation report over existing op semantics
- Generic MLP `expansion -> contraction` channel-deadness pairs
- Attention value-path `v_proj -> out_proj` pairs with reshape/transpose/context evidence
- Explicitly blocked query/key records for `QK^T` score contraction
- OPT SparseGPT alignment summary
- Text dumps, Markdown explanations, comparison report, CLI tools, tests, API endpoint, and optional static coverage integration

Primary commands:

```bash
./conda-env/bin/python scripts/analyze_deadbranch_propagation.py --model facebook/opt-125m --verbose
./conda-env/bin/python scripts/explain_deadbranch_propagation.py --model facebook/opt-125m --contains v_proj --limit 5
./conda-env/bin/python scripts/compare_deadbranch_propagation.py --models all --verbose
```

For OPT-125M, the expected report contains `12` FFN pairs, `12` attention value-path pairs, and `24` separately blocked Q/K records. This is static analysis/reporting only.

## Milestone 38: DFA Worklist Prototype for Static Pruning Propagation

Status: complete.

Implemented under `experimental/dfa_pruning_propagation/`:

- Explicit operation/axis graph IR
- Conservative pruning-fact lattice
- Compiler-style transfer functions
- Queue-based fixed-point worklist solver with trace diagnostics
- FFN, attention value-path, Q/K blocked, and residual/LayerNorm protected examples
- Markdown/text/JSON report rendering and standalone CLI
- Fast deterministic teaching-prototype tests

Primary commands:

```bash
python -m experimental.dfa_pruning_propagation.cli --example ffn --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-value --show-trace
python -m experimental.dfa_pruning_propagation.cli --example attention-qk --show-trace
python -m pytest -q experimental/dfa_pruning_propagation/tests
```

This is a separate experimental teaching prototype. It does not replace production analysis, select concrete indices, execute pruning, mutate models, download models, or evaluate accuracy.

## Milestone 40: Axis Transfer Summary Prototype

Status: complete.

Implemented under `experimental/axis_transfer_analysis/`:

- MLIR-inspired loop IV, tensor, indexed-access, operation, and region IR
- Access-derived axis summaries for preserved, reduced, permuted, broadcast, protected, blocked, and unknown relations
- FFN intermediate-chain recognition from projection, unary-preservation, and contraction accesses
- Attention value-path recognition from preserved `V.value_dim -> Context.value_context_dim` evidence
- Explicit Q/K score-contraction blocker derived from reduced feature-axis accesses
- Residual and LayerNorm hidden-axis protection summaries
- Markdown/text/JSON reports, standalone CLI, and deterministic tests

Primary commands:

```bash
python -m experimental.axis_transfer_analysis.cli --example qk-score --format markdown --show-relations
python -m experimental.axis_transfer_analysis.cli --example attention-context --format markdown --show-relations
python -m experimental.axis_transfer_analysis.cli --example ffn --format markdown --show-patterns
python -m experimental.axis_transfer_analysis.cli --example attention-value-path --format markdown --show-patterns
```

This is a separate experimental loop/access-analysis prototype. It does not replace production analysis, execute pruning, mutate models, download models, or evaluate accuracy.

## Milestone 41: Connect Axis Transfer Summaries to DFA Propagation

Status: complete.

Implemented under `experimental/pruning_analysis_bridge/`:

- End-to-end bridge from loop/access evidence to DFA fixed-point propagation
- Access-analysis and pattern-recognition trace records
- Generic-label DFA graph lowering for FFN chains, attention value paths, Q/K score blockers, residual merges, and LayerNorm protection
- Symbolic seed policies for consumer-input deadness and blocked hidden/head-axis attempts
- Combined Markdown/text/JSON reports showing upstream axis evidence and downstream DFA facts
- Standalone CLI, teaching README, and deterministic tests

Primary commands:

```bash
python -m experimental.pruning_analysis_bridge.cli --example ffn-from-access --format markdown --show-all
python -m experimental.pruning_analysis_bridge.cli --example attention-value-from-access --format markdown --show-all
python -m experimental.pruning_analysis_bridge.cli --example qk-blocked-from-access --format markdown --show-all
```

This is a separate experimental bridge prototype. It does not replace production analysis, execute pruning, mutate models, download models, or evaluate accuracy.

## Milestone 42: ONNX Subgraph to Axis-Transfer Bridge

Status: complete.

Implemented under `experimental/onnx_axis_bridge/`:

- Read-only local ONNX subgraph loader with tensor-shape extraction
- Topology, producer/consumer, operator-class, initializer, and shape summaries
- Conservative hints for FFN-like chains, Q/K score contractions, attention context, attention value paths, residual adds, and explicit LayerNorm
- Template lowering into the existing loop/access `RegionSpec`
- Reuse of axis-transfer summaries, pattern recognition, semantic DFA graph construction, and fixed-point propagation
- Markdown/text/JSON reports, standalone CLI, synthetic ONNX tests, and best-effort real artifact smoke support

Primary commands:

```bash
python -m experimental.onnx_axis_bridge.cli --help
python -m experimental.onnx_axis_bridge.cli --onnx artifacts/model_analysis_subgraphs/gpt2/layers/layer_0/03_gpt_2_block_0_mlp_block/subgraph.onnx --format markdown --show-all
python -m experimental.onnx_axis_bridge.cli --onnx artifacts/model_analysis_subgraphs/bert-base-uncased/layers/layer_0/05_layer_0_attention_score_matmul/subgraph.onnx --format markdown --show-all
```

This is a separate experimental ONNX-subgraph bridge, not full MLIR lowering. It does not replace production analysis, execute pruning, mutate models, download models, or evaluate accuracy.

## Milestone 43: ONNX-MLIR Access Semantics Bridge

Status: complete.

Implemented under `experimental/mlir_axis_bridge/`:

- Local discovery for ONNX-MLIR and `mlir-opt`
- Read-only lowering of selected ONNX subgraphs to ONNX dialect and lowered MLIR artifacts
- Recursive discovery of preserved MLIR stages and dialect hints
- Conservative text extraction for affine/scf loops plus affine/memref load-store accesses
- Axis-summary construction that distinguishes actual loop accesses, high-level MLIR dialect evidence, and ONNX-hint fallback
- Reuse of the existing axis-transfer recognizers and DFA propagation bridge
- Markdown/text/JSON reports, standalone CLI, and synthetic parser/bridge tests

Primary command:

```bash
python -m experimental.mlir_axis_bridge.cli \
  --onnx artifacts/model_analysis_subgraphs/gpt2/layers/layer_0/03_gpt_2_block_0_mlp_block/subgraph.onnx \
  --output-dir reports/mlir_axis_bridge/gpt2_layer0_mlp \
  --format markdown \
  --show-all
```

MLIR is used as a local evidence generator for selected subgraphs. This experiment does not replace production analysis, lower full models, execute pruning, mutate model weights, or evaluate accuracy.

## Milestone 44: Native MLIR Dependence Evidence Prototype

Status: complete.

Extended `experimental/mlir_axis_bridge/` with:

- A shared native-style dependence JSON model for Python extraction and future MLIR passes
- Loop-nesting context on extracted affine/scf load-store records
- Conservative Python summaries for preserved, reduced, and mixed indexed-access relations
- Native dependence JSON import and Python dependence JSON emission
- Evidence precedence for `native_mlir_dependence_evidence`, `actual_loop_access_evidence`, high-level MLIR evidence, and ONNX fallback
- Native dependence lowering into the existing axis-transfer and DFA bridge when supported motifs are proven
- An optional out-of-tree C++ `pruning-axis-dependence` pass scaffold under `experimental/mlir_axis_bridge/native/`

The native C++ pass is scaffold-only and is not required by automated tests. This remains an experimental local-evidence bridge for selected subgraphs, not a full MLIR rewrite.

## Milestone 45: Functional Native MLIR Axis Dependence Pass

Status: complete.

Extended `experimental/mlir_axis_bridge/native/` with:

- A standalone MLIR-linked `pruning-axis-dependence` executable
- Local parsing for selected MLIR artifacts with unregistered ONNX-MLIR dialect tolerance
- Affine/scf loop-IV tracking and affine/memref load-store extraction
- Conservative native JSON facts for preserved, reduced, and mixed access relations
- A local CMake build helper and parseable MLIR sample fixtures
- Expected JSON samples compatible with `native_dependence.py`
- Optional Python orchestration through `--run-native-pass`, `--native-pass-tool`, and `--native-output-dir`
- Evidence hierarchy reporting with clean fallback to Python affine extraction

The native analyzer remains a selected-subgraph local evidence generator. It does not transform MLIR, lower full models, execute pruning, mutate model weights, or replace production analysis.

## Milestone 46: Cross-Evidence Pruning Proof Report

Status: complete.

Added `experimental/pruning_proof_report/`:

- Selected layer-0 proof cases across transformer MLP, attention score/context, residual, and LayerNorm subgraphs
- One report schema for ONNX hints, MLIR dialects, native/Python/fallback evidence sources, axis relations, recognized patterns, DFA facts, verdicts, and limitations
- Evidence-source precedence from native MLIR dependence evidence through explicit ONNX-only fallback
- Aggregate Markdown and JSON reports plus per-case details
- Missing-artifact tolerance and a standalone CLI

This is a reporting and evaluation layer over selected local subgraphs. It does not execute pruning, mutate models, lower full models, or replace production analysis.

## Milestone 47: MLIR Evidence Coverage Study

Status: complete.

Added `experimental/mlir_evidence_coverage/`:

- Flexible discovery for local model-analysis ONNX subgraphs with sanitized model-directory and older-layout support
- A model-by-pattern coverage matrix across BERT, DistilBERT, OPT, GPT-2, and ViT
- Evidence tiers for native MLIR dependence facts, Python affine/access facts, high-level MLIR dialect evidence, ONNX fallback, and unavailable evidence
- Verdicts for native/access/fallback proofs, expected blockers, partial mappings, missing artifacts, unknown evidence, and failures
- Aggregate per-model and per-pattern summaries
- Markdown/JSON reports, per-case details, CLI, README, and deterministic tests

This is a reporting and evaluation layer over selected subgraphs. It does not execute pruning, mutate models, lower full models, or replace production analysis.

## Milestone 48: Full Attention Value-Path Subgraph Artifacts

Status: complete.

Added complete local attention value-path extraction:

- Semantic-anchor discovery from deadbranch propagation reports
- Source-ONNX connectivity recovery for value projection -> layout -> attention context -> layout -> output projection
- Seedability and value-axis mapping status in JSON and Markdown reports
- Best-effort ONNX, DOT, and SVG evidence artifact export
- Cross-model explanation and comparison CLIs
- Coverage-study discovery for the new artifact root
- A curated OPT layer-0 attention value-path proof case

These artifacts expose a seedable local chain to the existing experimental evidence stack. They do not execute pruning, choose channel indices, or mutate model weights.

## Milestone 49: BERT 24-Plan MLIR-Backed Propagation Proof

Status: complete.

Extended attention value-path evidence and reporting for BERT:

- Complete `attention.self.value -> context -> attention.output.dense` ONNX fragments
- Value-axis-preserving layout recovery and seedability status per encoder layer
- MLIR coverage discovery for BERT attention value paths
- A BERT-specific proof report joining 12 validated FFN plans with 12 attention value-path proofs
- Per-layer evidence tiers, verdicts, aggregate counts, and explicit QK blocker interpretation

This is static evidence and proof reporting only. It does not execute pruning, choose channel indices, or mutate model weights.

## Milestone 50: Formalize Static Pruning Propagation Analysis

Status: complete.

Added `experimental/formalization/`:

- Compiler-style static pruning propagation notes
- Sparse-weight versus structural-pruning distinction
- Axis-fact lattice, axis-transfer relations, semantic pattern recognition, and DFA worklist methodology
- BERT 24-plan case study generated from existing proof artifacts
- Teaching slide outline and paper methodology outline
- Graceful partial documentation when evidence inputs are absent
- Markdown/JSON index, CLI, deterministic tests, and demo

This is documentation and formalization only. It does not execute pruning, mutate models, or evaluate accuracy.

## Milestone 51: All-Model Propagation Plan Proof

Status: complete.

Added `experimental/all_model_plan_proof/`:

- A common plan-proof schema for BERT, DistilBERT, OPT, GPT-2, and ViT
- Per-layer FFN and attention value-path evidence cells backed by the existing MLIR coverage evaluator
- Separate QK blocker tables that remain outside pruning-plan totals
- Optional generation of missing separable attention value-path artifacts
- Explicit fused-QKV value-slice recovery gaps for GPT-2 and ViT
- Aggregate Markdown/JSON reports, per-model details, CLI, README, and deterministic tests
- An optional all-model snapshot in the static formalization bundle

This is static artifact, evidence, and proof reporting only. It does not execute pruning, choose channel indices, mutate model weights, or evaluate accuracy.

## Milestone 52: Fused-QKV Value-Slice Recovery

Status: complete.

Extended attention value-path extraction with:

- Source-ONNX graph recovery when deadbranch value-pair anchors are absent
- Conservative GPT-2 `c_attn -> Split -> value branch -> context -> attn/c_proj` recovery
- Explicit value-slice status, QKV layout, recovered slice operations, and evidence records
- Value-operand-specific context binding so recovery cannot follow the Q/K score branch
- ViT recovery through the local exported `v_proj -> context -> o_proj` path
- Coverage and all-model proof policy updates for GPT-2 and ViT
- Synthetic fused, separate, and ambiguous-branch tests plus all-layer artifact smoke coverage

Ambiguous fused QKV paths remain blocked. This is static artifact and proof generation only.

## Milestone 53: Final Static Pruning Propagation Research Report

Status: complete.

Added `experimental/final_report/`:

- Robust collection of all-model, BERT, formalization, MLIR coverage, deadbranch, value-path, plan, and validation reports
- A final research narrative covering the compiler-style model, evidence hierarchy, DFA propagation, SparseGPT alignment, claims, limitations, and next directions
- Machine-readable JSON summary, CSV case table, claims boundary document, and index
- Non-strict warning-based collection plus strict all-model-proof validation
- CLI wrapper, README, deterministic tests, demo, and ignored generated output root

This is final documentation and reporting only. It does not execute pruning, mutate model weights, or evaluate accuracy.

## Milestone 54: Upgrade OPT FFN Native MLIR Evidence

Status: complete.

Added `experimental/opt_ffn_native_diagnosis/`:

- Per-layer diagnosis of the OPT FFN high-level MLIR fallback
- Explicit reporting of the original ONNX-MLIR LayerNorm dtype blocker
- Read-only extraction of topology-proven `fc1 -> activation -> fc2` FFN-core ONNX artifacts
- MLIR coverage discovery preference for `mlp_native_core` evidence artifacts
- Native dependence reruns, per-layer Markdown/JSON diagnosis, tests, and demo

The native proof criterion remains unchanged: the local MLIR-linked tool must emit preserved and reduced dependence facts sufficient for `FFN_INTERMEDIATE_CHAIN`. This is static evidence diagnosis only. It does not execute pruning, mutate model weights, or evaluate accuracy.
