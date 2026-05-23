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
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
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
  -> Region-Aware Dimension IR
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

ONNX currently supplies the frontend graph and Netron visualization. Tensor IR is the frontend-independent analysis substrate; the Structural Region Tree organizes that IR into compiler-inspired semantic regions; Region-Aware Dimension IR lowers region interfaces into symbolic dimensions and equations; region legality analysis answers symbolic requests over those dimensions. The Netron index lists the original ONNX model first, so extracted structural regions can be opened alongside their full-graph source context.

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
7. `reports/region_dimension_ir/bert-base-uncased.md`
8. `reports/region_pruning_ir_dumps/bert-base-uncased.rdim`
9. `reports/region_blocked_analysis/bert-base-uncased__blocked_dimensions.md`
10. `reports/region_legality_checks/`
11. `reports/dependency_graphs/bert-base-uncased.md`
12. `reports/correspondence/bert-base-uncased.md`
13. `reports/join_subgraphs/bert-base-uncased.md`
14. `reports/subgraph_pruning_analysis/bert-base-uncased.md`
15. `reports/dag_regions/bert-base-uncased.md`
16. `reports/dag_region_pruning_evidence/bert-base-uncased.md`
17. `reports/netron_subgraph_index/bert-base-uncased__demo.md`
18. `reports/model_pruning_maps/bert-base-uncased.md`
19. `reports/dimension_ir/bert-base-uncased.md`
20. `reports/pruning_ir_dumps/bert-base-uncased.pir`
21. `reports/legality_checks/`

## Mainline vs backend

Milestones 6-8 are experimental lowering/backend demos. They are useful for validating structural hypotheses, but they are not required for the main analysis path.

## Expected interpretation

The pipeline should make pruning look less like "remove small weights" and more like a legality problem over dimensions, constraints, equivalence classes, and blocked regions.
