# Usage Guide

This repository is organized around a staged analysis pipeline. Each stage writes durable artifacts under `reports/` so later stages can run without reloading or re-exporting models.

## Guided Demo Track

For a learning-oriented path, start with:

```bash
bash demo_scripts/run_demo_01_setup_check.sh
```

Then follow [the demo track](../demos/README.md) or run the full single-model research pipeline:

```bash
PYTHON=python MODEL=bert-base-uncased bash demo_scripts/run_full_analysis_pipeline.sh
```

The demo track is organized around the main research artifacts: structural inventory, frontend-independent Tensor IR, Structural Region Tree, Region-Aware Dimension IR, Region Pruning Semantics, Op Semantics, region-aware legality analysis, dependency graph, correspondence evidence, join-aware subgraph evidence, bounded DAG-region evidence, Netron visualization fragments, pruning maps, Dimension IR, and legality analysis. Executable pruning commands are documented as optional experimental backend demos.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Supported Models

Configured models live in `configs/models.yaml`.

Current model names:

```text
bert-base-uncased
distilbert-base-uncased
gpt2
opt-125m
vit-base-patch16-224
```

Most commands accept either the configured name or the Hugging Face ID where applicable. For example, `opt-125m` and `facebook/opt-125m` both resolve to the OPT entry.

## Model Download

Download one model:

```bash
python scripts/download_models.py --model bert-base-uncased
```

Download all configured models:

```bash
python scripts/download_models.py --model all
```

Force a redownload:

```bash
python scripts/download_models.py --model bert-base-uncased --force
```

Use a custom Hugging Face cache directory:

```bash
python scripts/download_models.py --model bert-base-uncased --cache-dir /path/to/cache
```

Downloaded model files are written under `data/models/hf/` and are ignored by git.

## ONNX Export

Export one model:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased
```

Export all downloaded models:

```bash
python scripts/export_to_onnx.py --model all
```

Use a different opset:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased --opset 17
```

ONNX exports are written under `data/models/onnx/` and are ignored by git.

### Static-Shape ONNX for Netron

The dynamic export above remains the analysis input. A separate static export can make Netron display concrete tensor dimensions:

```bash
./conda-env/bin/python scripts/export_static_shape_onnx.py \
  --model bert-base-uncased \
  --seq-len max \
  --batch-size 1 \
  --opset 17 \
  --device cpu
```

The result is written to `data/models/onnx_static/bert-base-uncased/model.static.onnx` with adjacent metadata. To attempt all local models while retaining a report for any failure:

```bash
./conda-env/bin/python scripts/export_static_shape_onnx.py \
  --model all \
  --seq-len 128 \
  --batch-size 1 \
  --opset 17 \
  --device cpu \
  --continue-on-error
```

This path is for visualization only and does not replace `data/models/onnx/`.

## Frontend-Independent Tensor Graph IR

ONNX graph summaries are the currently implemented frontend input. Import them into Tensor IR before future structural-region analysis:

```bash
python scripts/build_tensor_ir.py --model bert-base-uncased --verbose
```

Generate and compare all currently available model IRs:

```bash
python scripts/build_tensor_ir.py --model all --verbose
python scripts/compare_tensor_ir.py --models all
```

Generated outputs:

```text
reports/tensor_ir/<model>.json
reports/tensor_ir/<model>.md
reports/tensor_ir_dumps/<model>.tir
reports/tensor_ir_stats/<model>.json
reports/tensor_ir_stats/<model>.md
```

Tensor IR exposes canonical operations, tensor values, producer/consumer connectivity, forks, joins, semantic roles, and region hints. ONNX remains useful as a frontend and for Netron inspection; it is not the core structural-analysis abstraction.

## Semantic Fusion for Activation and Feed-Forward Regions

Run semantic idiom recovery on Tensor IR before rebuilding semantic regions:

```bash
python scripts/analyze_semantic_fusion.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 20
```

Generated fusion reports are written to `reports/semantic_fusion/` and `reports/fused_region_patterns/`. The Structural Region Tree enables semantic fusion by default; pass `--disable-semantic-fusion` to `build_structural_region_tree.py` only when comparing against the conservative unfused baseline. GELU fusion lifts decomposed activation operations into `ActivationRegion` and `FeedForwardRegion` candidates without rewriting Tensor IR, ONNX, or model weights.

## Stepwise Dataflow Control-Tree Trace

Build a step-by-step construction trace for the Structural Region Tree:

```bash
python scripts/build_control_tree_trace.py --model bert-base-uncased --format all --max-dot-steps 20 --verbose
python tools/export_control_tree_trace_mindnode.py --model bert-base-uncased
```

Inspect:

```text
reports/control_tree_steps/<model>.md
reports/control_tree_step_dumps/<model>.ctrace
reports/control_tree_step_graphs/<model>/step_000.dot
reports/mindnode_outlines/<model>.control_tree_steps.mindnode.txt
```

The trace starts from primitive TensorOps, records each semantic-region collapse, and snapshots the current abstract graph. It is explanatory structural analysis only and does not replace the final Structural Region Tree.

## Lightweight Control-Tree Trace Viewer

Run the trace API server after building a trace:

```bash
python scripts/build_control_tree_trace.py \
  --model bert-base-uncased \
  --format all \
  --max-dot-steps 20 \
  --verbose

python tools/control_tree_trace_api_server.py \
  --model bert-base-uncased \
  --port 8766
```

Open:

```text
http://127.0.0.1:8766/
```

The final Structural Region Tree shows what was found; the control-tree trace shows how structures were collapsed step by step. The viewer fetches only local step neighborhoods and is visualization only.

## Ordered Dataflow Control-Tree Browser

Run the ordered hierarchy browser:

```bash
python tools/ordered_control_tree_api_server.py \
  --model bert-base-uncased \
  --port 8767
```

Open:

```text
http://127.0.0.1:8767/
```

This viewer shows the final Structural Region Tree as an expandable dataflow control tree. It preserves Tensor IR / ONNX-like source order, expands one node at a time, and lets a selected abstract region reveal ordered primitive leaves. It is different from the step trace viewer: the trace viewer shows how reductions happened step by step, while this browser shows the final hierarchy in model order.

## Abstract Node Expansion Report

Use this report when you want a printable explanation of what each learner-facing abstract node expands into. It records both direct hierarchy and source evidence:

- `immediate_expansion`: direct abstract or primitive children in the learner hierarchy
- `recursive_primitive_leaves`: source ONNX/TensorIR operations under the node

Generate the main learner report:

```bash
./conda-env/bin/python tools/export_abstract_node_expansion_report.py \
  --model bert-base-uncased \
  --view main \
  --max-leaf-names 30
```

Generate the grouped shape/mask report:

```bash
./conda-env/bin/python tools/export_abstract_node_expansion_report.py \
  --model bert-base-uncased \
  --view shape \
  --max-leaf-names 30
```

The main view hides auxiliary shape/mask flow. The shape view groups those operations into `ShapeMotifRegion` records by default. Use `--include-single-op-shape-regions` and `--include-root-leaves` for debugging raw one-op shape regions and root/section/motif primitive leaves.

## Region Pruning Semantics

Build a conservative region-level semantics report:

```bash
./conda-env/bin/python scripts/build_region_pruning_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Explain specific opportunities or blockers:

```bash
./conda-env/bin/python scripts/explain_region_pruning_semantics.py \
  --model bert-base-uncased \
  --contains "Feed Forward" \
  --limit 5

./conda-env/bin/python scripts/explain_region_pruning_semantics.py \
  --model bert-base-uncased \
  --blocked-only \
  --limit 10
```

Compare available reports:

```bash
./conda-env/bin/python scripts/compare_region_pruning_semantics.py --models all
```

Generated outputs:

```text
reports/region_pruning_semantics/<model>.json
reports/region_pruning_semantics_dumps/<model>.rpsem
reports/region_pruning_semantics_explanations/<model>.md
reports/region_pruning_semantics_compare/summary.md
```

This layer explains pruning information flow through semantic regions. It records prunable dimensions, propagated dimensions, protected dimensions, required repairs, and blockers such as residual hidden equality, LayerNorm hidden width, and unproven attention head-axis mapping.

By default, the Markdown report summarizes raw `AxisTransformRegion`, `ForkRegion`, and `JoinRegion` auxiliary flow instead of listing every one-op record. Pass `--include-auxiliary-details` to `build_region_pruning_semantics.py` only when debugging shape/axis plumbing.

Each region record includes both `source_region_type` from the Structural Region Tree and `semantic_category` from the pruning semantics layer. This is important for attention internals: score/context MatMuls can be structurally shaped like `LinearProjectionRegion` records while semantically acting as attention contractions.
Mask-broadcast Axis/Fork/Join helper regions use auxiliary attention-mask categories, while `attention_mask_add` is reserved for the true score-bias Add node.

## Op Semantics

Build primitive Tensor IR op semantics:

```bash
./conda-env/bin/python scripts/build_op_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Explain selected op classes:

```bash
./conda-env/bin/python scripts/explain_op_semantics.py \
  --model bert-base-uncased \
  --semantic-kind attention_score_matmul \
  --limit 5

./conda-env/bin/python scripts/explain_op_semantics.py \
  --model bert-base-uncased \
  --category parameterized_projection \
  --limit 10
```

Generated outputs:

```text
reports/op_semantics/<model>.json
reports/op_semantics_dumps/<model>.opsem
reports/op_semantics_explanations/<model>.md
reports/op_semantics_compare/summary.md
```

Op Semantics annotates primitive Tensor IR operations with local pruning-relevant behavior. It distinguishes learned projection MatMuls from attention score/context contractions, bias adds from residual adds, GELU elementwise pieces from axis/metadata flow, and unknown ops that need future classifier work. This is static reporting only.

## Structural Region Tree over Tensor IR

Build the first compiler-style semantic-region hierarchy from a persisted Tensor IR:

```bash
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
```

Build and compare trees across all available model IRs:

```bash
python scripts/build_structural_region_tree.py --model all --verbose
python scripts/compare_structural_region_trees.py --models all
```

Generated outputs:

```text
reports/structural_region_trees/<model>.json
reports/structural_region_trees/<model>.md
reports/structural_region_dumps/<model>.srtree
reports/structural_region_interfaces/<model>.json
reports/structural_region_interfaces/<model>.md
reports/structural_region_patterns/<model>.json
reports/structural_region_patterns/<model>.md
```

Leaves remain primitive TensorOps. Internal nodes conservatively summarize recognized tensor-computation regions and expose preliminary pruning/propagation roles; this step is static analysis only.

## Region-Aware Dimension IR

Lower structural-region interfaces into region-scoped dimension variables and constraints:

```bash
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
```

Build and compare region Dimension IR reports across all available region trees:

```bash
python scripts/build_region_dimension_ir.py --model all --verbose
python scripts/compare_region_dimension_ir.py --models all
```

Generated outputs:

```text
reports/region_dimension_ir/<model>.json
reports/region_dimension_ir/<model>.md
reports/region_pruning_ir_dumps/<model>.rdim
reports/region_constraint_equations/<model>.json
reports/region_constraint_equations/<model>.md
reports/region_dimension_equivalence/<model>.json
reports/region_dimension_equivalence/<model>.md
```

RegionDimensionIR derives symbolic axes and equations from semantic regions. It intentionally retains unresolved attention and reshape-axis mappings and blocking residual/normalization constraints. It does not modify models.

## Region-Aware Pruning Propagation Analysis

Inspect available region dimensions and protected constraints:

```bash
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 10
python scripts/explain_region_blocked_dimensions.py --model bert-base-uncased
```

Analyze a selected region dimension:

```bash
python scripts/check_region_pruning_legality.py \
  --model bert-base-uncased \
  --dimension-var <region_dimension_var_id> \
  --count 4 \
  --verbose
```

Generated outputs:

```text
reports/region_legality_checks/<model>__<request>.json
reports/region_legality_checks/<model>__<request>.md
reports/region_propagation_slices/<model>__<request>__forward.json
reports/region_propagation_slices/<model>__<request>__backward.json
reports/region_repair_sets/<model>__<request>.json
reports/region_repair_sets/<model>__<request>.md
reports/region_blocked_analysis/<model>__dimension_list.json
reports/region_blocked_analysis/<model>__blocked_dimensions.md
```

This query layer operates on semantic region dimensions, infers constraint traversal conservatively, and reports repair obligations without applying them. It does not modify models.

## Region Tree Browsing and Outline Export

Build a catalog of unique abstract structures from a Structural Region Tree and Region-Aware Dimension IR:

```bash
python abstract_structure_collector.py \
  --model bert-base-uncased \
  --write
```

Start the focused API-backed browser:

```bash
python region_structure_api_server.py \
  --model bert-base-uncased \
  --port 8765
```

Open `http://127.0.0.1:8765/`. The browser fetches catalog entries, selected instances, focused region details, direct children, and dimensions through GET requests instead of loading the complete tree JSON.

Export a compact MindNode outline:

```bash
./conda-env/bin/python tools/export_region_tree_mindnode.py \
  --model bert-base-uncased \
  --label-mode semantic \
  --include-counts \
  --max-depth 3
```

Generated outputs are under `reports/abstract_structures/`, `reports/mindnode_outlines/`, and `viewer_data/`; those generated artifacts are ignored by git except `.gitkeep` placeholders.

## Quick Inspection

Print a lightweight local model summary and write a Markdown summary:

```bash
python scripts/inspect_model.py --model bert-base-uncased
```

## Structural Inventory

Generate PyTorch structural inventory, optional ONNX graph summary, and pruning hints:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Require the ONNX export to be present:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
```

Control structural and ONNX report formats:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased --format json
python scripts/generate_structural_inventory.py --model bert-base-uncased --format md
python scripts/generate_structural_inventory.py --model bert-base-uncased --format both
```

Generated outputs:

```text
reports/structural_inventory/<model>.json
reports/structural_inventory/<model>.md
reports/onnx_graphs/<model>.json
reports/onnx_graphs/<model>.md
reports/pruning_hints/<model>.md
```

## Dependency Graph Construction

Build the pruning-dependency graph from structural inventory reports:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Require the ONNX graph summary:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx
```

Build from PyTorch inventory only:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --torch-only
```

Print graph statistics:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --verbose
```

Generated outputs:

```text
reports/dependency_graphs/<model>.json
reports/dependency_graphs/<model>.md
reports/dependency_summaries/<model>.json
reports/dependency_summaries/<model>.md
```

## Recommended Single-Model Flow

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx --verbose
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/export_demo_subgraphs.py --model bert-base-uncased --max-per-category 3 --verbose
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
```

## Recommended All-Model Flow

```bash
python scripts/download_models.py --model all
python scripts/export_to_onnx.py --model all
python scripts/generate_structural_inventory.py --model all --require-onnx
python scripts/build_dependency_graph.py --model all --require-onnx --verbose
python scripts/build_correspondence.py --model all --require-dependency-graph --verbose
python scripts/analyze_subgraphs.py --model all --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_dag_regions.py --model all --max-branch-depth 4 --verbose
python scripts/generate_candidate_actions.py --model all --simulate --limit 5
```

## PyTorch-to-ONNX Correspondence and Shape Evidence

Build static correspondence and shape evidence for one model:

```bash
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
```

Build reports for all models:

```bash
python scripts/build_correspondence.py --model all --require-dependency-graph --verbose
```

Generated outputs:

```text
reports/correspondence/<model>.json
reports/correspondence/<model>.md
reports/shape_evidence/<model>.json
reports/shape_evidence/<model>.md
reports/validated_dependency_graphs/<model>.json
reports/validated_dependency_graphs/<model>.md
```

Use evidence during manual pruning action simulation:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --use-evidence \
  --verbose
```

Correspondence is heuristic and conservative. Shape evidence is static and derived from ONNX graph metadata. This still does not prune weights, rewrite PyTorch modules, or rewrite ONNX.

## k-Node and Join-Aware Subgraph Analysis

Analyze local directed ONNX paths and branch-merge subgraphs:

```bash
python scripts/analyze_subgraphs.py \
  --model bert-base-uncased \
  --max-nodes 5 \
  --branch-depth 2 \
  --post-join-depth 2 \
  --verbose
```

Analyze all models with existing ONNX summaries and compare their patterns:

```bash
python scripts/analyze_subgraphs.py --model all --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/compare_subgraphs.py --models all
```

Generated outputs:

```text
reports/subgraphs/<model>.json
reports/subgraphs/<model>.md
reports/subgraph_patterns/<model>.json
reports/subgraph_patterns/<model>.md
reports/subgraph_pruning_analysis/<model>.json
reports/subgraph_pruning_analysis/<model>.md
reports/subgraph_dimension_evidence/<model>.json
reports/subgraph_dimension_evidence/<model>.md
reports/join_subgraphs/<model>.json
reports/join_subgraphs/<model>.md
reports/residual_subgraphs/<model>.json
reports/residual_subgraphs/<model>.md
```

A `MatMul` followed by an initializer-backed `Add` is reported as a bias addition, not a residual merge. Joins combining two dataflow branches, especially before `LayerNormalization`, are reported separately as residual-like evidence. This pass produces reports only.

## DAG Motif and Multi-Join Region Analysis

Analyze bounded fanout and reconvergence patterns:

```bash
python scripts/analyze_dag_regions.py \
  --model bert-base-uncased \
  --max-branch-depth 4 \
  --verbose
```

Analyze and compare existing ONNX summaries for all configured models:

```bash
python scripts/analyze_dag_regions.py --model all --max-branch-depth 4 --verbose
python scripts/compare_dag_regions.py --models all
```

Generated outputs:

```text
reports/dag_regions/<model>.json
reports/dag_regions/<model>.md
reports/dag_region_patterns/<model>.json
reports/dag_region_patterns/<model>.md
reports/dag_region_pruning_evidence/<model>.json
reports/dag_region_pruning_evidence/<model>.md
```

Path analysis captures sequential neighborhoods; join analysis captures an individual merge. DAG-region analysis captures forks, diamonds, and `join_fork_join` motifs such as `A,B -> C -> D,E -> F`, where multi-branch propagation and reconvergence constraints must be considered together. This is report-only structural analysis.

## Netron ONNX Subgraph Export

Export a curated small bundle of visualization fragments:

```bash
python scripts/export_demo_subgraphs.py --model bert-base-uncased --max-per-category 3 --verbose
```

Export an exact recorded subgraph:

```bash
python scripts/export_subgraph_onnx.py \
  --model bert-base-uncased \
  --subgraph-id path_3_000012 \
  --verbose
```

Filter exported fragments by analysis evidence:

```bash
python scripts/export_subgraph_onnx.py --model bert-base-uncased --kind join --pattern-contains "Join(Add)" --max-exports 10
python scripts/export_subgraph_onnx.py --model bert-base-uncased --kind dag_region --max-exports 20
python scripts/export_subgraph_onnx.py --model bert-base-uncased --kind path --pattern-contains "Softmax" --max-exports 10
```

Generated outputs:

```text
artifacts/subgraph_onnx/<model>/<kind>/<subgraph-id>.onnx
artifacts/subgraph_onnx/<model>/demo/*.onnx
reports/subgraph_exports/<model>.json
reports/subgraph_exports/<model>.md
reports/subgraph_exports/<model>__demo.json
reports/subgraph_exports/<model>__demo.md
reports/netron_subgraph_index/<model>.md
reports/netron_subgraph_index/<model>__demo.md
```

Each generated Netron index lists the original model at `data/models/onnx/<model>/model.onnx` first as the full-graph comparison baseline, followed by commands for its extracted fragments. Each extracted ONNX graph preserves its selected nodes, required initializers, artificial boundary inputs/outputs, available value information, opsets, and provenance metadata. It is intended for Netron structural inspection; it is not a semantically complete model fragment and it does not alter the original ONNX model.

## Reversible PyTorch Linear Pruning

Milestone 6 adds the first executable transform, restricted to PyTorch `nn.Linear` `out_features` and `in_features` pruning. The output is a new local artifact; the source model is not modified.

Dry run:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.attention.self.query \
  --dim out_features \
  --indices 0,1,2,3 \
  --only-target \
  --dry-run \
  --verbose
```

Actual Linear-only execution:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.attention.self.query \
  --dim out_features \
  --indices 0,1,2,3 \
  --only-target \
  --allow-ambiguous \
  --verbose
```

Execute from an existing plan JSON:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --plan-json reports/pruning_plans/<plan>.json \
  --only-target \
  --allow-ambiguous \
  --verbose
```

Generated outputs:

```text
artifacts/pruned_models/<model>/<execution-id>/
reports/pruning_execution/<model>__<execution-id>.json
reports/pruning_execution/<model>__<execution-id>.md
reports/pruning_diffs/<model>__<execution-id>.json
reports/pruning_diffs/<model>__<execution-id>.md
reports/rollback_manifests/<model>__<execution-id>.json
reports/rollback_manifests/<model>__<execution-id>.md
```

Caveats:

- Only `nn.Linear` surgery is supported.
- ONNX is not rewritten.
- End-to-end transformer correctness is not proven.
- Ambiguous plans require `--allow-ambiguous`.
- Use `--only-target` for the safest first experiments.

## Paired Linear Repair and Forward Smoke Tests

Milestone 7 adds the first structural consistency repair pattern: MLP expansion/projection paired Linear pruning. An expansion layer `out_features` prune can induce a matching projection layer `in_features` prune when the pruning plan explicitly contains `mlp_hidden_coupling`.

Repair-plan-only flow:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense \
  --dim out_features \
  --indices 0,1,2,3 \
  --repair-pairs \
  --write-repair-plan-only \
  --allow-ambiguous \
  --verbose
```

Dry-run with repair detection:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense \
  --dim out_features \
  --indices 0,1,2,3 \
  --repair-pairs \
  --dry-run \
  --allow-ambiguous \
  --verbose
```

Execution with before/after smoke tests:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense \
  --dim out_features \
  --indices 0,1,2,3 \
  --repair-pairs \
  --allow-ambiguous \
  --smoke-test-before \
  --smoke-test-after \
  --verbose
```

Standalone forward smoke test:

```bash
python scripts/run_forward_smoke_test.py --model bert-base-uncased --device cpu --verbose
```

Generated outputs:

```text
reports/repair_plans/<model>__<execution-id>.json
reports/repair_plans/<model>__<execution-id>.md
reports/repair_transactions/<model>__<execution-id>.json
reports/repair_transactions/<model>__<execution-id>.md
reports/forward_smoke_tests/<model>__<execution-id>__before.json
reports/forward_smoke_tests/<model>__<execution-id>__after.json
```

MLP paired repair is the first supported consistency repair. Attention-head pruning is still not executable by default. Residual and LayerNorm repairs remain manual review. Forward smoke tests only check executable shape consistency, not model quality.

## BERT MLP Block-Level Pruning

Milestone 8 adds the first architecture-specific executable pruning path. It is narrower and safer than arbitrary dependency-graph execution because it only reduces the BERT MLP intermediate dimension:

```text
bert.encoder.layer.<L>.intermediate.dense out_features
bert.encoder.layer.<L>.output.dense in_features
```

List detected targets:

```bash
python scripts/list_bert_mlp_targets.py --model bert-base-uncased
```

Dry run:

```bash
python scripts/prune_bert_mlp_block.py \
  --model bert-base-uncased \
  --layer 0 \
  --indices 0,1,2,3 \
  --dry-run \
  --smoke-test-before \
  --verbose
```

Actual execution:

```bash
python scripts/prune_bert_mlp_block.py \
  --model bert-base-uncased \
  --layer 0 \
  --indices 0,1,2,3 \
  --smoke-test-before \
  --smoke-test-after \
  --verbose
```

Generated outputs:

```text
artifacts/pruned_models/<model>/bert_mlp_layer_<L>_<execution-id>/
reports/block_pruning/<model>__layer_<L>__<execution-id>.json
reports/block_pruning/<model>__layer_<L>__<execution-id>.md
reports/block_pruning_diffs/<model>__layer_<L>__<execution-id>.json
reports/block_pruning_diffs/<model>__layer_<L>__<execution-id>.md
reports/block_validation/<model>__layer_<L>__<execution-id>__before.json
reports/block_validation/<model>__layer_<L>__<execution-id>__after.json
reports/rollback_manifests/<model>__bert_mlp_layer_<L>__<execution-id>.json
```

This path should preserve model hidden size, so residual and LayerNorm dimensions remain unchanged. It does not prune attention heads, rewrite ONNX, or prove accuracy preservation. Downstream evaluation and fine-tuning are still required for quality. Single-layer BERT MLP pruning creates non-uniform intermediate sizes, so standard Hugging Face reload paths may need custom metadata support in a later milestone.

The executable pruning commands are experimental validation backends. They are useful for testing whether a structural hypothesis can be executed, but they are not the main research artifact.

## Compiler-Style Pruning Opportunity Maps

The main research path is compiler-style pruning opportunity analysis. It treats the model as a structural IR and emits:

- pruning dimensions
- propagation constraints
- pruning opportunities
- independent and coupled opportunity regions
- blocked regions
- structural risks

Full one-model analysis flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/export_demo_subgraphs.py --model bert-base-uncased --max-per-category 3 --verbose
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
```

Build maps for all configured models:

```bash
python scripts/build_pruning_map.py --model all --verbose
```

Compare existing maps across models:

```bash
python scripts/compare_pruning_maps.py --models all
```

Generated outputs:

```text
reports/model_pruning_maps/<model>.json
reports/model_pruning_maps/<model>.md
reports/pruning_opportunities/<model>.json
reports/pruning_opportunities/<model>.md
reports/propagation_constraints/<model>.json
reports/propagation_constraints/<model>.md
reports/structural_risk_maps/<model>.json
reports/structural_risk_maps/<model>.md
reports/model_pruning_maps/comparison.json
reports/model_pruning_maps/comparison.md
```

The goal is to reason about legal pruning spaces and pruning-information propagation before transforming weights. The pruning map is static evidence, not a correctness proof and not an executable pruning transform.

## Dimension Variable IR

Dimension variables are the compiler-style representation of prunable model dimensions. Index variables represent symbolic pruning selections. Constraint equations encode propagation rules such as MLP same-index coupling, QKV consistency, residual hidden equality, tied parameters, reshape preservation, and unknown mappings. Equivalence classes group dimensions that must be kept equal or pruned with the same symbolic index set.

The `.pir` dump is a research textual IR inspired by MLIR. It is deterministic and human-readable, but it is not executable MLIR and does not require MLIR tooling.

Build a Dimension IR for one model:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
```

Build Dimension IRs for all configured models:

```bash
python scripts/build_pruning_map.py --model all --verbose
python scripts/build_dimension_ir.py --model all --verbose
```

Compare existing Dimension IR reports:

```bash
python scripts/compare_dimension_irs.py --models all
```

Generated outputs:

```text
reports/dimension_ir/<model>.json
reports/dimension_ir/<model>.md
reports/pruning_ir_dumps/<model>.pir
reports/constraint_equations/<model>.json
reports/constraint_equations/<model>.md
reports/dimension_equivalence/<model>.json
reports/dimension_equivalence/<model>.md
reports/dimension_ir/comparison.json
reports/dimension_ir/comparison.md
```

Tensor IR, Structural Region Tree, Region-Aware Dimension IR, region-aware legality analysis, pruning maps, and Dimension IR are the primary research artifacts. Executable pruning remains experimental backend support only.

## Dimension-IR Legality Analysis

Milestone 11 adds a static compiler-analysis layer over Dimension IR. It checks symbolic pruning legality without touching model weights, extracts forward and backward propagation slices, reports minimal structural repair sets, and explains blocked regions.

Example flow:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
python scripts/list_pruning_dimensions.py --model bert-base-uncased --contains intermediate.dense
python scripts/check_pruning_legality.py \
  --model bert-base-uncased \
  --dimension-var <dimension_var_id> \
  --count 4 \
  --verbose
python scripts/explain_blocked_regions.py --model bert-base-uncased
```

Generated outputs:

```text
reports/legality_checks/<model>__<request>.json
reports/legality_checks/<model>__<request>.md
reports/propagation_slices/<model>__<request>__forward.json
reports/propagation_slices/<model>__<request>__backward.json
reports/repair_sets/<model>__<request>.json
reports/repair_sets/<model>__<request>.md
reports/ir_analysis/<model>__blocked_regions.json
reports/ir_analysis/<model>__dimension_list.json
```

Legality analysis is static and conservative. It does not modify models, execute pruning, rewrite ONNX, or evaluate accuracy. Tensor IR, Structural Region Tree, Region-Aware Dimension IR, region-aware legality analysis, pruning maps, and Dimension IR remain the primary research artifacts; executable pruning remains experimental backend support only.

## Pruning Action Simulation

Milestone 4 adds a dry-run pruning planner. It does not modify weights, PyTorch modules, or ONNX graphs. It simulates how a proposed pruning action propagates through the dependency graph and writes diagnostics.

Generate candidate actions:

```bash
python scripts/generate_candidate_actions.py --model bert-base-uncased
```

Generate and simulate the first five candidates:

```bash
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
```

Simulate a manual pruning action:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --use-evidence \
  --verbose
```

Use an action JSON file:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --action-json path/to/action.json
```

Ambiguous plans usually indicate missing shape mapping, residual coupling, embedding tying uncertainty, or PyTorch-to-ONNX correspondence gaps. To allow ambiguous plans in automation:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --allow-ambiguous
```

Generated outputs:

```text
reports/pruning_plans/<model>__<action>.json
reports/pruning_plans/<model>__<action>.md
reports/propagation_traces/<model>__<action>.json
reports/pruning_action_checks/<model>__<action>.json
reports/pruning_action_checks/<model>__candidate_actions.json
reports/pruning_action_checks/<model>__candidate_actions.md
```

## Validation

```bash
python -m compileall src scripts tests
.venv/bin/pytest -q
```

If the virtual environment does not exist but dependencies are installed globally:

```bash
pytest -q
```
