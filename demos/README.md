# Model-Analysis Demo Track

The demo track is the recommended way to learn this repository. It walks from a downloaded model to compiler-style pruning analysis artifacts without requiring users to read every implementation module first.

## The central idea

Neural-network pruning can be treated like a compiler transformation problem:

- model modules and ONNX graph nodes are the input program
- dimensions are symbolic variables
- pruning choices are transformations over those variables
- propagation constraints define legality
- blocked regions are failed or unresolved legality checks
- executable pruning is a backend/lowering problem, not the main research artifact

The repository is therefore organized around analysis artifacts: structural inventories, dependency graphs, pruning maps, Dimension IR, and legality reports.

## Recommended demo path

Main research path:

1. [Milestone 1: Project Setup](milestone_01_project_setup.md)
2. [Milestone 2: Structural Inventory](milestone_02_structural_inventory.md)
3. [Milestone 3: Dependency Graph](milestone_03_dependency_graph.md)
4. [Milestone 5: Correspondence and Shape Evidence](milestone_05_correspondence_shape_evidence.md)
5. [Milestone 13: k-Node and Join-Aware Subgraph Analysis](milestone_13_subgraph_analysis.md)
6. [Milestone 9: Pruning Opportunity Map](milestone_09_pruning_opportunity_map.md)
7. [Milestone 10: Dimension IR](milestone_10_dimension_ir.md)
8. [Milestone 11: Legality Analysis](milestone_11_legality_analysis.md)

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
