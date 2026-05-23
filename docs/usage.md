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

The demo track is organized around the main research artifacts: structural inventory, dependency graph, correspondence evidence, join-aware subgraph evidence, pruning maps, Dimension IR, and legality analysis. Executable pruning commands are documented as optional experimental backend demos.

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

Dimension IR and pruning maps are the primary research artifacts. Executable pruning remains experimental backend support only.

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

Legality analysis is static and conservative. It does not modify models, execute pruning, rewrite ONNX, or evaluate accuracy. Pruning maps and Dimension IR remain the primary research artifacts; executable pruning remains experimental backend support only.

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
