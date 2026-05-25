# Demo Track

The demo track is a research walkthrough for Model Analysis. It presents the repository as a compiler-style analysis pipeline rather than a collection of scripts.

## Motivation

The central research question is how to reason about pruning before modifying model weights. Instead of treating pruning as a local weight-ranking problem, the repository treats pruning as a structural transformation with legality constraints:

- frontend graph records are imported into Tensor IR as program operations
- dimensions are symbolic variables
- pruning index sets are transformation operands
- dependency edges and Dimension IR equations define propagation rules
- blocked regions identify transformations that are not legal with current evidence

## Artifact Ladder

Mainline artifacts:

```text
Model checkpoint
  -> ONNX frontend graph
  -> Structural inventory
  -> Tensor Graph IR
  -> Semantic Fusion
  -> Structural Region Tree
  -> Stepwise Control-Tree Construction Trace
  -> Region-Aware Dimension IR
  -> Region Pruning Semantics
  -> Op Semantics
  -> Pruning Opportunity Ranking
  -> Region-Aware Legality Analysis
  -> Dependency graph
  -> Correspondence and shape evidence
  -> k-node and join-aware subgraph evidence
  -> DAG motif and multi-join region evidence
  -> Netron visualization fragments
  -> Pruning opportunity map
  -> Dimension IR
  -> Legality check
```

Optional backend artifacts:

```text
Executable backend dry run
  -> Paired repair
  -> BERT MLP pruning prototype
```

## Milestone Table

| Milestone | Demo | Main artifact | Concept |
| --- | --- | --- | --- |
| 1 | Project setup | Model registry and local artifacts | Input program setup |
| 2 | Structural inventory | PyTorch and ONNX summaries | Front-end parsing |
| 3 | Dependency graph | Prunable units and edges | Use-def-like graph |
| 4 | Action simulation | Pruning plans and traces | Dry-run transformation planning |
| 5 | Correspondence | Module-node and shape evidence | Source-to-lowered-IR evidence |
| 6 | Linear backend | Dry-run execution reports | Experimental lowering backend |
| 7 | Paired repair | Repair plans | Structural repair set |
| 8 | BERT MLP backend | Block target reports | Architecture-specific lowering |
| 9 | Pruning map | Model pruning map | Global pruning legality space |
| 10 | Dimension IR | `.pir` textual dump | Symbolic compiler IR |
| 11 | Legality analysis | Legality and slice reports | Static legality oracle |
| 13 | Subgraph analysis | Path and join-centered evidence | Local pattern and join analysis |
| 14 | DAG region analysis | Fork/diamond/join-fork-join evidence | Multi-branch dataflow analysis |
| 15 | Netron subgraph export | Extracted ONNX fragments and index | Visual IR inspection |
| 16 | Tensor IR | Frontend-independent tensor graph and `.tir` dump | Frontend lowering into analysis IR |
| 17 | Structural Region Tree | Semantic region hierarchy and interfaces | Compiler-style region analysis |
| 18 | Region-Aware Dimension IR | `.rdim` region constraint dump | Semantic-region-derived symbolic dimensions |
| 19 | Region-Aware Legality Analysis | Region slices and repair obligations | Static semantic-region legality oracle |
| 20 | Semantic Fusion | GELU/feed-forward fusion reports | Idiom recovery from decomposed Tensor IR |
| 21 | Stepwise Control-Tree Trace | Collapse trace, DOT snapshots, MindNode outline | Explaining structural-analysis reductions |
| 25 | Region Pruning Semantics | `.rpsem` and explanation reports | Region-level pruning flow, repairs, and blockers |
| 25.4 | Op Semantics | `.opsem` and explanation reports | Primitive TensorOp pruning behavior |
| 26 | Pruning Opportunity Ranking | `.rank` and explanation reports | Prioritized safe/constrained/blocked candidates |

## Mainline vs Backend

The main research path is:

```text
1 -> 2 -> 16 -> 20 -> 17 -> 21 -> 18 -> 25 -> 25.4 -> 26 -> 19 -> 3 -> 5 -> 13 -> 14 -> 15 -> 9 -> 10 -> 11
```

Milestones 6, 7, and 8 are experimental execution backends. They are useful for validating structural ideas, but they are not the primary contribution. New analysis work should operate on Tensor IR, Structural Region Tree, Region-Aware Dimension IR, and region-aware legality analysis before expanding executable pruning.

## Recommended Presentation Sequence

1. Start with `demos/README.md` to frame pruning as compiler analysis.
2. Run `demo_scripts/run_demo_01_setup_check.sh`.
3. If model files are not present, run the download and ONNX export commands manually.
4. Run the structural inventory, Tensor IR, semantic fusion, Structural Region Tree, control-tree trace, Region-Aware Dimension IR, region pruning semantics, Op Semantics, pruning opportunity ranking, pruning plan synthesis, pruning plan validation, layer subgraph validation pack, region-aware legality, dependency graph, correspondence, subgraph analysis, DAG-region analysis, Netron export, pruning map, Dimension IR, and legality demos.
5. Open the reports in the artifact ladder order.
6. Mention backend demos only as optional lowering experiments.

## Expected Outputs

For a `bert-base-uncased` walkthrough, the key reports are:

- `reports/structural_inventory/bert-base-uncased.md`
- `reports/onnx_graphs/bert-base-uncased.md`
- `reports/tensor_ir/bert-base-uncased.md`
- `reports/tensor_ir_dumps/bert-base-uncased.tir`
- `reports/semantic_fusion/bert-base-uncased.md`
- `reports/fused_region_patterns/bert-base-uncased.md`
- `reports/structural_region_trees/bert-base-uncased.md`
- `reports/structural_region_dumps/bert-base-uncased.srtree`
- `reports/control_tree_steps/bert-base-uncased.md`
- `reports/control_tree_step_dumps/bert-base-uncased.ctrace`
- `reports/region_dimension_ir/bert-base-uncased.md`
- `reports/region_pruning_ir_dumps/bert-base-uncased.rdim`
- `reports/region_pruning_semantics_explanations/bert-base-uncased.md`
- `reports/region_pruning_semantics_dumps/bert-base-uncased.rpsem`
- `reports/op_semantics_explanations/bert-base-uncased.md`
- `reports/op_semantics_dumps/bert-base-uncased.opsem`
- `reports/pruning_opportunity_explanations/bert-base-uncased.md`
- `reports/pruning_opportunity_ranking_dumps/bert-base-uncased.rank`
- `reports/pruning_plan_explanations/bert-base-uncased.md`
- `reports/pruning_plan_dumps/bert-base-uncased.plan`
- `reports/pruning_plan_validation_explanations/bert-base-uncased.md`
- `reports/pruning_plan_validation_dumps/bert-base-uncased.pvalid`
- `reports/layer_subgraph_validation/bert-base-uncased/layer_0/index.md`
- `artifacts/layer_subgraphs/bert-base-uncased/layer_0/`
- `reports/region_blocked_analysis/bert-base-uncased__blocked_dimensions.md`
- `reports/region_legality_checks/`
- `reports/dependency_graphs/bert-base-uncased.md`
- `reports/correspondence/bert-base-uncased.md`
- `reports/shape_evidence/bert-base-uncased.md`
- `reports/join_subgraphs/bert-base-uncased.md`
- `reports/subgraph_pruning_analysis/bert-base-uncased.md`
- `reports/dag_regions/bert-base-uncased.md`
- `reports/dag_region_pruning_evidence/bert-base-uncased.md`
- `reports/netron_subgraph_index/bert-base-uncased__demo.md`
- `reports/model_pruning_maps/bert-base-uncased.md`
- `reports/dimension_ir/bert-base-uncased.md`
- `reports/pruning_ir_dumps/bert-base-uncased.pir`
- `reports/ir_analysis/bert-base-uncased__dimension_list.md`
- `reports/legality_checks/`

## Running the Full Demo

```bash
PYTHON=python MODEL=bert-base-uncased bash demo_scripts/run_full_analysis_pipeline.sh
```

This command can download a model and export ONNX. It does not execute pruning or modify model weights.
