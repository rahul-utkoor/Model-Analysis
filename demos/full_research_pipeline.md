# Full Research Pipeline

This is the coherent mainline demo for the repository. It intentionally skips executable pruning backends by default because the primary artifacts are analysis reports.

## Single-model mainline

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_tensor_ir.py --model bert-base-uncased --verbose
python scripts/analyze_semantic_fusion.py --model bert-base-uncased --verbose
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
python scripts/build_control_tree_trace.py --model bert-base-uncased --format all --max-dot-steps 20 --verbose
python tools/export_control_tree_trace_mindnode.py --model bert-base-uncased
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
python scripts/build_region_pruning_semantics.py --model bert-base-uncased --verbose
python scripts/build_op_semantics.py --model bert-base-uncased --verbose
python scripts/rank_pruning_opportunities.py --model bert-base-uncased --verbose
python scripts/synthesize_pruning_plans.py --model bert-base-uncased --verbose
python scripts/validate_pruning_plans.py --model bert-base-uncased --verbose
python scripts/build_layer_subgraph_validation_pack.py --model bert-base-uncased --layer 0 --export-onnx --render-svg --verbose
python scripts/explain_region_pruning_semantics.py --model bert-base-uncased --contains "Feed Forward" --limit 5
python scripts/list_region_dimensions.py --model bert-base-uncased --contains intermediate --limit 10
python scripts/explain_region_blocked_dimensions.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx --verbose
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
python scripts/analyze_subgraphs.py --model bert-base-uncased --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose
python scripts/analyze_dag_regions.py --model bert-base-uncased --max-branch-depth 4 --verbose
python scripts/export_demo_subgraphs.py --model bert-base-uncased --max-per-category 3 --verbose
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
python scripts/list_pruning_dimensions.py --model bert-base-uncased --contains intermediate.dense --limit 10
python scripts/check_pruning_legality.py \
  --model bert-base-uncased \
  --dimension-var <dimension_var_id> \
  --count 4 \
  --verbose
python scripts/explain_blocked_regions.py --model bert-base-uncased
```

You can run the same mainline with:

```bash
bash demo_scripts/run_full_analysis_pipeline.sh
```

## Artifact ladder

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
  -> Symbolic Pruning Plans
  -> Pruning Plan Validation
  -> Layer Subgraph Validation Pack
  -> Region-Aware Legality Analysis
  -> Dependency graph
  -> Correspondence and shape evidence
  -> k-node and join-aware subgraph evidence
  -> DAG motif and multi-join region evidence
  -> Netron visualization artifacts
  -> Pruning opportunity map
  -> Dimension IR
  -> Legality check
```

ONNX currently supplies the frontend graph and Netron visualization. Tensor IR is the frontend-independent analysis substrate; the Structural Region Tree organizes that IR into compiler-inspired semantic regions; Region-Aware Dimension IR lowers region interfaces into symbolic dimensions and equations; Region Pruning Semantics explains region-level roles and blockers; Op Semantics annotates primitive TensorOps with local pruning behavior; Pruning Opportunity Ranking prioritizes safe/constrained/blocked candidates; Pruning Plan Synthesis turns top safe FFN opportunities into symbolic index-set plans; Pruning Plan Validation checks those plans for static consistency; Layer Subgraph Validation Packs project the full-model analysis onto individual learner nodes for inspection; Full-Model Analysis Reports aggregate every layer and compare generated model reports; Static Coverage Study records complete/partial/skipped support across configured models; region legality analysis answers symbolic requests over those dimensions. The Netron index lists the original ONNX model first, so extracted structural regions can be opened alongside their full-graph source context.

Optional backend ladder:

```text
Executable backend dry run
  -> Paired repair
  -> BERT MLP pruning prototype
```

## What to inspect

Read these reports in order:

1. `reports/structural_inventory/bert-base-uncased.md`
2. `reports/tensor_ir/bert-base-uncased.md`
3. `reports/tensor_ir_dumps/bert-base-uncased.tir`
4. `reports/semantic_fusion/bert-base-uncased.md`
5. `reports/structural_region_trees/bert-base-uncased.md`
6. `reports/structural_region_dumps/bert-base-uncased.srtree`
7. `reports/control_tree_steps/bert-base-uncased.md`
8. `reports/control_tree_step_dumps/bert-base-uncased.ctrace`
9. `reports/region_dimension_ir/bert-base-uncased.md`
10. `reports/region_pruning_ir_dumps/bert-base-uncased.rdim`
11. `reports/region_pruning_semantics_explanations/bert-base-uncased.md`
12. `reports/region_pruning_semantics_dumps/bert-base-uncased.rpsem`
13. `reports/op_semantics_explanations/bert-base-uncased.md`
14. `reports/op_semantics_dumps/bert-base-uncased.opsem`
15. `reports/pruning_opportunity_explanations/bert-base-uncased.md`
16. `reports/pruning_opportunity_ranking_dumps/bert-base-uncased.rank`
17. `reports/pruning_plan_explanations/bert-base-uncased.md`
18. `reports/pruning_plan_dumps/bert-base-uncased.plan`
19. `reports/pruning_plan_validation_explanations/bert-base-uncased.md`
20. `reports/pruning_plan_validation_dumps/bert-base-uncased.pvalid`
21. `reports/layer_subgraph_validation/bert-base-uncased/layer_0/index.md`
22. `artifacts/layer_subgraphs/bert-base-uncased/layer_0/`
23. `reports/region_blocked_analysis/bert-base-uncased__blocked_dimensions.md`
24. `reports/region_legality_checks/`
25. `reports/dependency_graphs/bert-base-uncased.md`
26. `reports/correspondence/bert-base-uncased.md`
27. `reports/join_subgraphs/bert-base-uncased.md`
28. `reports/subgraph_pruning_analysis/bert-base-uncased.md`
29. `reports/dag_regions/bert-base-uncased.md`
30. `reports/dag_region_pruning_evidence/bert-base-uncased.md`
31. `reports/netron_subgraph_index/bert-base-uncased__demo.md`
32. `reports/model_pruning_maps/bert-base-uncased.md`
33. `reports/dimension_ir/bert-base-uncased.md`
34. `reports/pruning_ir_dumps/bert-base-uncased.pir`
35. `reports/legality_checks/`

## Mainline vs backend

Milestones 6-8 are experimental lowering/backend demos. They are useful for validating structural hypotheses, but they are not required for the main analysis path.

## Expected interpretation

The pipeline should make pruning look less like "remove small weights" and more like a legality problem over dimensions, constraints, equivalence classes, and blocked regions.
