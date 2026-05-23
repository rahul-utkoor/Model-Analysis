# Milestone 17: Structural Region Tree over Tensor IR

## What you learn

You learn how primitive Tensor IR operations are organized into a compiler-inspired hierarchy of semantic tensor-computation regions, including projections, axis transforms, residual merges, attention skeletons, forks, and joins.

## Why this milestone exists

A flat dataflow graph exposes connectivity but does not organize repeated structural meaning. Compiler structural analysis builds region hierarchies to make later data-flow reasoning tractable. For neural networks, this tree provides a region-oriented basis for future pruning propagation.

## Prerequisites

- Tensor IR from Milestone 16 under `reports/tensor_ir/<model>.json`

## Commands

```bash
bash demo_scripts/run_demo_17_structural_region_tree.sh
```

Equivalent direct command:

```bash
python scripts/build_structural_region_tree.py --model bert-base-uncased --verbose
```

Compare all constructed trees:

```bash
python scripts/compare_structural_region_trees.py --models all
```

## Main artifacts produced

- `reports/structural_region_trees/bert-base-uncased.md`
- `reports/structural_region_dumps/bert-base-uncased.srtree`
- `reports/structural_region_interfaces/bert-base-uncased.md`
- `reports/structural_region_patterns/bert-base-uncased.md`

## What to inspect

Open the `.srtree` dump. Locate `ModelRegion`, then inspect nested `LinearProjectionRegion`, `ResidualMergeRegion`, `AttentionSkeletonRegion`, and `PrimitiveRegion` entries. In the interface report, check which regions are labeled `directly_prunable`, `propagation_only`, `constraint_carrier`, or `blocked`.

## Expected interpretation

Primitive TensorOps remain leaves. Internal nodes summarize recognized structural computation without altering the underlying graph. Residual merges block local hidden-width changes; projection regions expose candidate dimensions; axis transforms and forks carry propagation requirements.

## Compiler analogy

The Structural Region Tree is the neural tensor-dataflow analogue of a compiler control tree: it recognizes bounded structural regions and provides a hierarchy for efficient later data-flow and legality analysis.

## What this milestone does not prove

The first pass is conservative and does not recover every decomposed feed-forward or attention detail. It does not execute pruning, modify weights, rewrite ONNX, or evaluate accuracy.

## Connection to next milestone

Future work should connect region interfaces to Dimension IR, allowing propagation constraints and legality queries to operate over hierarchical regions rather than flat operation neighborhoods.
