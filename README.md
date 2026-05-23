# Model Analysis

Model Analysis is a research scaffold for structural analysis of neural networks, with an emphasis on pruning opportunities, dependency tracking, and forward/backward propagation of pruning information across model graphs.

The first milestone is infrastructure: a clean repository structure, reproducible setup, model download scripts, ONNX export scripts, and basic inspection summaries.

## Guided Demo Track

The best way to understand the repository is the guided demo track:

- [Model-Analysis Demo Track](demos/README.md)
- [Full Research Pipeline](demos/full_research_pipeline.md)
- `demo_scripts/run_full_analysis_pipeline.sh`

The demo path explains each milestone, the command to run, the report to inspect, and the compiler/pruning concept demonstrated. The main research artifacts are pruning maps, Dimension IR, legality reports, local/join-aware subgraph evidence, and bounded DAG-region evidence. Executable pruning support is optional and experimental backend work.

## Initial Supported Models

| Name | Hugging Face ID | Task |
| --- | --- | --- |
| `bert-base-uncased` | `bert-base-uncased` | masked-lm |
| `distilbert-base-uncased` | `distilbert-base-uncased` | masked-lm |
| `gpt2` | `gpt2` | causal-lm |
| `opt-125m` | `facebook/opt-125m` | causal-lm |
| `vit-base-patch16-224` | `google/vit-base-patch16-224` | image-classification |

## Repository Layout

```text
configs/                  Model registry configuration
scripts/                  CLI utilities for downloads, ONNX export, and inspection
demo_scripts/             Guided demo shell wrappers
demos/                    Milestone-by-milestone learning walkthroughs
src/model_analysis/       Reusable Python package code
data/models/hf/           Local Hugging Face model snapshots (ignored by git)
data/models/onnx/         Exported ONNX models (ignored by git)
reports/model_summaries/  Generated Markdown summaries (ignored by git)
reports/structural_inventory/  Generated PyTorch inventory reports (ignored by git)
reports/onnx_graphs/      Generated ONNX graph reports (ignored by git)
reports/pruning_hints/    Generated pruning hint reports (ignored by git)
reports/dependency_graphs/  Generated dependency graph reports (ignored by git)
reports/dependency_summaries/  Generated dependency analyzer summaries (ignored by git)
reports/correspondence/  Generated PyTorch-to-ONNX correspondence reports (ignored by git)
reports/shape_evidence/  Generated static shape evidence reports (ignored by git)
reports/validated_dependency_graphs/  Dependency graph validation reports (ignored by git)
artifacts/pruned_models/  Generated pruned model checkpoints (ignored by git)
reports/pruning_execution/  Generated pruning execution reports (ignored by git)
reports/pruning_diffs/  Generated pruning structural diffs (ignored by git)
reports/rollback_manifests/  Generated rollback manifests (ignored by git)
reports/repair_plans/  Generated paired Linear repair plans (ignored by git)
reports/repair_transactions/  Generated paired repair transaction logs (ignored by git)
reports/forward_smoke_tests/  Generated forward smoke validation reports (ignored by git)
reports/block_pruning/  Generated BERT MLP block pruning reports (ignored by git)
reports/block_validation/  Generated block-level forward smoke reports (ignored by git)
reports/block_pruning_diffs/  Generated block-level structural diffs (ignored by git)
reports/model_pruning_maps/  Compiler-style pruning opportunity maps (ignored by git)
reports/pruning_opportunities/  Focused opportunity reports (ignored by git)
reports/propagation_constraints/  Focused constraint reports (ignored by git)
reports/structural_risk_maps/  Focused structural risk reports (ignored by git)
reports/dimension_ir/  Symbolic Dimension IR reports (ignored by git)
reports/constraint_equations/  Focused symbolic constraint equations (ignored by git)
reports/dimension_equivalence/  Dimension equivalence class reports (ignored by git)
reports/pruning_ir_dumps/  MLIR-like pruning IR text dumps (ignored by git)
reports/legality_checks/  Static Dimension-IR legality reports (ignored by git)
reports/propagation_slices/  Forward/backward propagation slice reports (ignored by git)
reports/repair_sets/  Minimal structural repair-set reports (ignored by git)
reports/ir_analysis/  Dimension IR helper analysis reports (ignored by git)
reports/subgraphs/  k-node ONNX path and join-aware report bundles (ignored by git)
reports/subgraph_patterns/  Aggregated local pattern reports (ignored by git)
reports/subgraph_pruning_analysis/  Subgraph pruning evidence reports (ignored by git)
reports/subgraph_dimension_evidence/  Candidate constraint evidence reports (ignored by git)
reports/join_subgraphs/  Branch-merge subgraph reports (ignored by git)
reports/residual_subgraphs/  Residual-like join candidate reports (ignored by git)
reports/dag_regions/  Fork, diamond, and join-fork-join region reports (ignored by git)
reports/dag_region_patterns/  Aggregated DAG motif reports (ignored by git)
reports/dag_region_pruning_evidence/  Multi-branch constraint evidence reports (ignored by git)
docs/                     Design notes, milestone notes, and detailed usage
tests/                    Lightweight pytest coverage
```

## Documentation

Detailed project documentation lives in:

- [Usage Guide](docs/usage.md)
- [Design Notes](docs/design.md)
- [Milestones](docs/milestones.md)
- [Demo Track](docs/demo_track.md)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Download all configured models:

```bash
python scripts/download_models.py --model all
```

Export one model to ONNX:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased
```

Inspect one local model:

```bash
python scripts/inspect_model.py --model bert-base-uncased
```

## Structural Inventory

Generate structural inventory reports for one downloaded model:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Suggested first flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Generated outputs:

```text
reports/structural_inventory/<model>.json  PyTorch module, parameter, layer, and pruning-group inventory
reports/structural_inventory/<model>.md    Human-readable PyTorch structural inventory
reports/onnx_graphs/<model>.json           ONNX graph node, initializer, IO, and pruning-relevant node inventory
reports/onnx_graphs/<model>.md             Human-readable ONNX graph summary
reports/pruning_hints/<model>.md           Conservative structural pruning hints and dependency caveats
```

Use `--require-onnx` when an ONNX report must exist, and `--format json|md|both` to control generated structural and ONNX report formats.

## Dependency Graph Construction

Build a conservative pruning-dependency graph from existing structural inventory reports:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Full single-model flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Generated dependency outputs:

```text
reports/dependency_graphs/<model>.json      Dependency graph IR with prunable units and dependency edges
reports/dependency_graphs/<model>.md        Human-readable dependency graph report
reports/dependency_summaries/<model>.json   Analyzer summary for targets, paths, constraints, and review items
reports/dependency_summaries/<model>.md     Human-readable dependency summary
```

The dependency graph is a conservative static pruning-dependency IR. It is not an executable pruning transform yet and does not prove that a pruning decision is safe.

## Pruning Action Simulation

Generate and simulate small candidate pruning actions:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
```

Simulate one manual action:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --verbose
```

This does not prune weights, rewrite PyTorch modules, or rewrite ONNX. It only simulates dependency propagation and emits candidate plans, propagation traces, and validation diagnostics. Ambiguous results are expected for complex transformer structures until later milestones add stronger PyTorch/ONNX correspondence and executable pruning transforms.

## PyTorch-to-ONNX Correspondence and Shape Evidence

Build static correspondence and shape evidence after structural inventory and dependency graph reports exist:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
```

Use evidence during pruning simulation:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --use-evidence \
  --verbose
```

Correspondence is heuristic and conservative. Shape evidence is static ONNX metadata. This still does not perform pruning.

## k-Node and Join-Aware Subgraph Analysis

Analyze consecutive ONNX paths of one through five nodes and separately capture branch-merge regions such as residual-style `Add` joins:

```bash
python scripts/analyze_subgraphs.py \
  --model bert-base-uncased \
  --max-nodes 5 \
  --branch-depth 2 \
  --post-join-depth 2 \
  --verbose
```

Compare existing subgraph reports across models:

```bash
python scripts/compare_subgraphs.py --models all
```

This report pass distinguishes bias additions from residual candidates, preserving join semantics that ordinary directed paths cannot represent. It produces local pruning and propagation evidence for future refinement of pruning maps and Dimension IR; it does not modify models.

## DAG Motif and Multi-Join Region Analysis

Detect bounded fork, diamond, and join-fork-join regions that cannot be represented as one linear path or one merge-centered subgraph:

```bash
python scripts/analyze_dag_regions.py \
  --model bert-base-uncased \
  --max-branch-depth 4 \
  --verbose
```

Compare existing DAG-region reports across models:

```bash
python scripts/compare_dag_regions.py --models all
```

For example, `A,B -> C -> D,E -> F` is represented as a `join_fork_join` region: `C` is both merge and fanout, and `F` is the reconvergence join. This pass records multi-branch propagation evidence only and does not modify models.

## Reversible PyTorch Linear Pruning

Dry-run a Linear-only pruning action:

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

Execute the same Linear-only structural surgery into a new artifact directory:

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

This creates a new checkpoint under `artifacts/pruned_models/`. The original model directory remains untouched. This is not full transformer-valid pruning yet; it is a controlled prototype for module-level `nn.Linear` structural surgery.

## Paired Linear Repair and Forward Smoke Tests

Write a repair plan for an MLP expansion/projection pair:

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

Dry-run paired repair detection:

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

Run a standalone forward smoke test:

```bash
python scripts/run_forward_smoke_test.py --model bert-base-uncased --device cpu --verbose
```

MLP paired repair is the first supported consistency repair. Attention-head pruning, residual repair, and LayerNorm repair remain manual-review items. Forward smoke tests only check executable shape consistency; they do not prove model quality.

## BERT MLP Block-Level Pruning

List executable BERT MLP targets:

```bash
python scripts/list_bert_mlp_targets.py --model bert-base-uncased
```

Dry-run layer 0 intermediate pruning:

```bash
python scripts/prune_bert_mlp_block.py \
  --model bert-base-uncased \
  --layer 0 \
  --indices 0,1,2,3 \
  --dry-run \
  --smoke-test-before \
  --verbose
```

Execute the same architecture-specific pruning path:

```bash
python scripts/prune_bert_mlp_block.py \
  --model bert-base-uncased \
  --layer 0 \
  --indices 0,1,2,3 \
  --smoke-test-before \
  --smoke-test-after \
  --verbose
```

This is an experimental execution backend. It only reduces the BERT MLP intermediate dimension by pruning `intermediate.dense` `out_features` and `output.dense` `in_features` with the same indices. It preserves hidden size, does not prune attention heads, does not rewrite ONNX, and still requires downstream evaluation or fine-tuning for quality. Single-layer BERT MLP pruning creates non-uniform intermediate sizes, so standard Hugging Face reload paths may need custom metadata support in a later milestone.

## Compiler-Style Pruning Opportunity Maps

The primary research path of this repository is compiler-style structural analysis: identify pruning dimensions, propagation constraints, coupled regions, blocked regions, and structural risks before transforming weights.

Full one-model analysis flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
```

Build and compare maps for all configured models:

```bash
python scripts/build_pruning_map.py --model all --verbose
python scripts/compare_pruning_maps.py --models all
```

Executable pruning modules are experimental validation backends. The main artifact is the model pruning map: a static IR for legal pruning spaces and propagation constraints.

## Dimension Variable IR

Dimension variables are the compiler-style representation of prunable model dimensions. Index variables represent symbolic pruning selections, constraint equations encode propagation rules, and equivalence classes capture dimensions that must be pruned consistently. The `.pir` dump is a research textual IR inspired by MLIR; it does not require MLIR tooling.

Build one model’s Dimension IR:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
```

Build and compare Dimension IRs for all configured models:

```bash
python scripts/build_pruning_map.py --model all --verbose
python scripts/build_dimension_ir.py --model all --verbose
python scripts/compare_dimension_irs.py --models all
```

Dimension IR and pruning maps are the main research artifacts. Executable pruning remains experimental backend support only.

## Dimension-IR Legality Analysis

Legality analysis checks symbolic pruning requests against the Dimension IR without touching weights. It reports required same-index propagation, forward/backward slices, blocking constraints, unresolved mappings, and minimal structural repair sets.

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

This performs static legality analysis only. It does not modify models, execute pruning, rewrite ONNX, or evaluate accuracy.

## First Push

```bash
git add .
git commit -m "Initial model analysis project scaffold"
git push -u origin main
```
