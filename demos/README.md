# Model-Analysis Demo Track

The demo track is the recommended way to learn this repository. It walks from a downloaded model to compiler-style pruning analysis artifacts without requiring users to read every implementation module first.

## The central idea

Neural-network pruning can be treated like a compiler transformation problem:

- frontend graph records are imported into Tensor IR as the input program
- dimensions are symbolic variables
- pruning choices are transformations over those variables
- propagation constraints define legality
- blocked regions are failed or unresolved legality checks
- executable pruning is a backend/lowering problem, not the main research artifact

ONNX is currently one implemented frontend. The repository is therefore organized around frontend-independent analysis artifacts: Tensor IR, Structural Region Tree, Region-Aware Dimension IR, pruning maps, Dimension IR, and legality reports, with existing ONNX-specific evidence passes retained as frontend utilities.

## Recommended demo path

Main research path:

1. [Milestone 1: Project Setup](milestone_01_project_setup.md)
2. [Milestone 2: Structural Inventory](milestone_02_structural_inventory.md)
3. [Milestone 16: Frontend-Independent Tensor Graph IR](milestone_16_tensor_ir.md)
4. [Milestone 17: Structural Region Tree](milestone_17_structural_region_tree.md)
5. [Milestone 18: Region-Aware Dimension IR](milestone_18_region_dimension_ir.md)
6. [Milestone 3: Dependency Graph](milestone_03_dependency_graph.md)
7. [Milestone 5: Correspondence and Shape Evidence](milestone_05_correspondence_shape_evidence.md)
8. [Milestone 13: k-Node and Join-Aware Subgraph Analysis](milestone_13_subgraph_analysis.md)
9. [Milestone 14: DAG Motif and Multi-Join Region Analysis](milestone_14_dag_region_analysis.md)
10. [Milestone 15: Netron ONNX Subgraph Export](milestone_15_netron_subgraph_export.md)
11. [Milestone 9: Pruning Opportunity Map](milestone_09_pruning_opportunity_map.md)
12. [Milestone 10: Dimension IR](milestone_10_dimension_ir.md)
13. [Milestone 11: Legality Analysis](milestone_11_legality_analysis.md)

Optional backend path:

- [Milestone 6: Linear Pruning Backend](milestone_06_linear_pruning_backend.md)
- [Milestone 7: Paired Linear Repair](milestone_07_paired_linear_repair.md)
- [Milestone 8: BERT MLP Block Pruning](milestone_08_bert_mlp_block_pruning.md)

Milestone 4 is useful as a bridge between dependency graphs and later legality checks: [Pruning Action Simulation](milestone_04_pruning_action_simulation.md).

## Minimal demo

Use `bert-base-uncased` only:

```bash
PYTHON=python bash demo_scripts/run_full_analysis_pipeline.sh
```

This may download a model and export ONNX if those artifacts are missing.

## Full comparison demo

After all five models are downloaded and exported, run the all-model analysis commands from [Full Research Pipeline](full_research_pipeline.md). The configured models are:

- `bert-base-uncased`
- `distilbert-base-uncased`
- `gpt2`
- `facebook/opt-125m`
- `google/vit-base-patch16-224`

Use [Glossary](glossary.md) while reading reports.
