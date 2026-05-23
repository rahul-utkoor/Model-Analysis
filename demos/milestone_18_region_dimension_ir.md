# Milestone 18: Region-Aware Dimension IR

## What you learn

You learn how semantic regions introduce symbolic dimensions and constraints. A projection exposes feature dimensions, a feed-forward region exposes a coupled intermediate dimension, and residual, normalization, attention, or transform regions preserve constraints that block or qualify pruning reasoning.

## Why this milestone exists

The earlier Dimension IR was derived from low-level pruning dependency units. The Structural Region Tree provides semantic computation regions and interfaces. RegionDimensionIR translates those interfaces into region-scoped dimensions, equations, equivalence classes, blocked dimensions, and unresolved constraints.

## Prerequisites

- Structural Region Tree from Milestone 17 under `reports/structural_region_trees/<model>.json`

## Commands

```bash
bash demo_scripts/run_demo_18_region_dimension_ir.sh
```

Equivalent direct command:

```bash
python scripts/build_region_dimension_ir.py --model bert-base-uncased --verbose
```

Compare available region-scoped IR reports:

```bash
python scripts/compare_region_dimension_ir.py --models all
```

## Main artifacts produced

- `reports/region_dimension_ir/bert-base-uncased.md`
- `reports/region_pruning_ir_dumps/bert-base-uncased.rdim`
- `reports/region_constraint_equations/bert-base-uncased.md`
- `reports/region_dimension_equivalence/bert-base-uncased.md`

## What to inspect

Open the `.rdim` dump and locate dimensions belonging to `LinearProjectionRegion`, `FeedForwardRegion`, `ResidualMergeRegion`, and `AttentionSkeletonRegion`. In the Markdown report, compare prunable, protected, propagated, and blocked dimensions. Check `mlp_intermediate_same_indices`, `residual_hidden_equality`, and `axis_transform_mapping` equations.

## Expected interpretation

Feed-forward regions may expose same-index intermediate constraints that are structurally promising. Residual joins protect hidden width. Axis transforms and attention skeletons are retained as unresolved or blocking constraints because a semantic axis mapping has not yet been proven.

## Compiler analogy

This is a region-scoped symbolic data-flow IR: semantic regions supply variables and transfer constraints, and equivalence classes record which symbolic dimensions must move together.

## What this milestone does not prove

It does not decide that a pruning request is executable, recover every tensor-axis mapping, modify weights, rewrite ONNX, or assess model accuracy.

## Connection to next milestone

Region-aware legality analysis can consume this IR to explain propagation slices and blocked decisions in terms of semantic regions rather than only flat graph units.
