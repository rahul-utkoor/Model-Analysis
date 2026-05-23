# Demo Track

The demo track is a research walkthrough for Model Analysis. It presents the repository as a compiler-style analysis pipeline rather than a collection of scripts.

## Motivation

The central research question is how to reason about pruning before modifying model weights. Instead of treating pruning as a local weight-ranking problem, the repository treats pruning as a structural transformation with legality constraints:

- model modules and ONNX nodes are program operations
- dimensions are symbolic variables
- pruning index sets are transformation operands
- dependency edges and Dimension IR equations define propagation rules
- blocked regions identify transformations that are not legal with current evidence

## Artifact Ladder

Mainline artifacts:

```text
Model checkpoint
  -> ONNX graph
  -> Structural inventory
  -> Dependency graph
  -> Correspondence and shape evidence
  -> k-node and join-aware subgraph evidence
  -> DAG motif and multi-join region evidence
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

## Mainline vs Backend

The main research path is:

```text
1 -> 2 -> 3 -> 5 -> 13 -> 14 -> 9 -> 10 -> 11
```

Milestones 6, 7, and 8 are experimental execution backends. They are useful for validating structural ideas, but they are not the primary contribution. New analysis work should usually extend pruning maps, Dimension IR, legality checking, or evidence precision before expanding executable pruning.

## Recommended Presentation Sequence

1. Start with `demos/README.md` to frame pruning as compiler analysis.
2. Run `demo_scripts/run_demo_01_setup_check.sh`.
3. If model files are not present, run the download and ONNX export commands manually.
4. Run the structural inventory, dependency graph, correspondence, subgraph analysis, DAG-region analysis, pruning map, Dimension IR, and legality demos.
5. Open the reports in the artifact ladder order.
6. Mention backend demos only as optional lowering experiments.

## Expected Outputs

For a `bert-base-uncased` walkthrough, the key reports are:

- `reports/structural_inventory/bert-base-uncased.md`
- `reports/onnx_graphs/bert-base-uncased.md`
- `reports/dependency_graphs/bert-base-uncased.md`
- `reports/correspondence/bert-base-uncased.md`
- `reports/shape_evidence/bert-base-uncased.md`
- `reports/join_subgraphs/bert-base-uncased.md`
- `reports/subgraph_pruning_analysis/bert-base-uncased.md`
- `reports/dag_regions/bert-base-uncased.md`
- `reports/dag_region_pruning_evidence/bert-base-uncased.md`
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
