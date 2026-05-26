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

ONNX is currently one implemented frontend. The repository is therefore organized around frontend-independent analysis artifacts: Tensor IR, Structural Region Tree, Region-Aware Dimension IR, Region Pruning Semantics, Op Semantics, Pruning Opportunity Ranking, symbolic Pruning Plans, Pruning Plan Validation, Layer Subgraph Validation Packs, region-aware legality reports, pruning maps, and the earlier Dimension IR path, with existing ONNX-specific evidence passes retained as frontend utilities.

## Recommended demo path

Main research path:

1. [Milestone 1: Project Setup](milestone_01_project_setup.md)
2. [Milestone 2: Structural Inventory](milestone_02_structural_inventory.md)
3. [Milestone 16: Frontend-Independent Tensor Graph IR](milestone_16_tensor_ir.md)
4. [Milestone 17: Structural Region Tree](milestone_17_structural_region_tree.md)
5. [Milestone 20: Semantic Fusion for Feed-Forward Regions](milestone_20_semantic_fusion.md)
6. [Milestone 21: Stepwise Control-Tree Trace](milestone_21_control_tree_trace.md)
7. [Milestone 18: Region-Aware Dimension IR](milestone_18_region_dimension_ir.md)
8. [Milestone 19: Region-Aware Legality Analysis](milestone_19_region_legality_analysis.md)
9. [Milestone 25: Region Pruning Semantics](milestone_25_region_pruning_semantics.md)
10. [Milestone 25.4: Op Semantics](milestone_25_4_op_semantics.md)
11. [Milestone 26: Pruning Opportunity Ranking](milestone_26_pruning_opportunity_ranking.md)
12. [Milestone 27: Pruning Plan Synthesis](milestone_27_pruning_plan_synthesis.md)
13. [Milestone 28: Pruning Plan Validation](milestone_28_pruning_plan_validation.md)
14. [Milestone 29: Layer Subgraph Validation Pack](milestone_29_layer_subgraph_validation_pack.md)
15. [Milestone 30: Full-Model Analysis Reports](milestone_30_full_model_analysis_reports.md)
16. [Milestone 31: Cross-Model Static Coverage](milestone_31_cross_model_static_coverage.md)
17. [Milestone 32: Rule-Gap Diagnosis and FFN Generalization](milestone_32_rule_gap_diagnosis_and_ffn_generalization.md)
18. [Milestone 33: Generic MLP Region Fusion](milestone_33_generic_mlp_region_fusion.md)
19. [Milestone 34: Generic Transformer Block Grouping](milestone_34_generic_block_grouping.md)
20. [Milestone 35: Interactive Static Analysis Explorer](milestone_35_interactive_analysis_explorer.md)
21. [Milestone 36: Pruning Analysis Web UI](milestone_36_analysis_web_ui.md)
22. [Milestone 3: Dependency Graph](milestone_03_dependency_graph.md)
23. [Milestone 5: Correspondence and Shape Evidence](milestone_05_correspondence_shape_evidence.md)
24. [Milestone 13: k-Node and Join-Aware Subgraph Analysis](milestone_13_subgraph_analysis.md)
25. [Milestone 14: DAG Motif and Multi-Join Region Analysis](milestone_14_dag_region_analysis.md)
26. [Milestone 15: Netron ONNX Subgraph Export](milestone_15_netron_subgraph_export.md)
27. [Milestone 9: Pruning Opportunity Map](milestone_09_pruning_opportunity_map.md)
28. [Milestone 10: Dimension IR](milestone_10_dimension_ir.md)
29. [Milestone 11: Legality Analysis](milestone_11_legality_analysis.md)

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
