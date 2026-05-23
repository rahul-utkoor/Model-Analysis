# Milestone 19: Region-Aware Pruning Propagation Analysis

## What you learn

You learn how a symbolic pruning request is analyzed against dimensions owned by semantic regions rather than raw graph nodes. The analyzer reports propagation obligations, protected dimensions, unresolved mappings, blockers, and a minimal structural repair set.

## Why this milestone exists

Region-Aware Dimension IR identifies dimensions and constraints, but a research user still needs a query layer: "what happens if I attempt to prune this region dimension?" This milestone is the static legality oracle for semantic-region dimensions.

## Prerequisites

- Region-Aware Dimension IR from Milestone 18 under `reports/region_dimension_ir/<model>.json`

## Commands

```bash
bash demo_scripts/run_demo_19_region_legality_analysis.sh
```

After selecting a prunable variable from the dimension listing:

```bash
python scripts/check_region_pruning_legality.py \
  --model bert-base-uncased \
  --dimension-var <region_dimension_var_id> \
  --count 4 \
  --verbose
```

## Main artifacts produced

- `reports/region_blocked_analysis/bert-base-uncased__dimension_list.md`
- `reports/region_blocked_analysis/bert-base-uncased__blocked_dimensions.md`
- `reports/region_legality_checks/<request>.md`
- `reports/region_propagation_slices/<request>__forward.md`
- `reports/region_propagation_slices/<request>__backward.md`
- `reports/region_repair_sets/<request>.md`

## What to inspect

Start with the dimension listing to find region-owned variables. Open the blocked-dimensions report to see why residual, normalization, attention, or transform regions remain protected or unresolved. For a query, inspect status, constraint satisfaction, forward/backward slices, and minimal repair obligations.

## Expected interpretation

A feed-forward intermediate variable may require the same index set at its paired consumer. A residual hidden dimension is rejected because branch widths must agree. An axis-transform or attention mapping may remain ambiguous because symbolic axis evidence is incomplete.

## Compiler analogy

This is data-flow and legality analysis over a semantic region IR: constraints define transfer obligations, slices expose affected regions, and blockers explain why a transformation cannot be lowered safely.

## What this milestone does not prove

It does not execute repairs, modify model weights, rewrite ONNX, prove semantic equivalence, or evaluate accuracy.

## Connection to next milestone

Future work can connect region-scoped variables to low-level variables and tensor-axis evidence, allowing a legality explanation to cite both semantic regions and concrete frontend graph structure.
