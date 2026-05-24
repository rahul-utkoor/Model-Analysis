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
