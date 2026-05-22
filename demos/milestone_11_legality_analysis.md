# Milestone 11: Legality Analysis

## What you learn

You learn how the Dimension IR answers symbolic pruning legality questions without touching model weights.

## Why this milestone exists

Given a proposed dimension and pruning amount, the analysis should explain whether the request is legal, legal with repairs, ambiguous, or rejected.

## Prerequisites

- Dimension IR from Milestone 10

## Commands

```bash
bash demo_scripts/run_demo_11_legality_analysis.sh
```

Manual flow:

```bash
python scripts/list_pruning_dimensions.py --model bert-base-uncased --contains intermediate.dense --limit 10
python scripts/check_pruning_legality.py \
  --model bert-base-uncased \
  --dimension-var <dimension_var_id> \
  --count 4 \
  --verbose
python scripts/explain_blocked_regions.py --model bert-base-uncased
```

## Main artifacts produced

- `reports/ir_analysis/bert-base-uncased__dimension_list.md`
- `reports/legality_checks/`
- `reports/propagation_slices/`
- `reports/repair_sets/`
- `reports/ir_analysis/bert-base-uncased__blocked_regions.md`

## What to inspect

Inspect status, requested dimension, equivalent dimensions, constraint satisfaction, forward slice, backward slice, minimal repair set, blocking reasons, and unresolved items.

## Expected interpretation

MLP intermediate dimensions may require same-index repairs. Hidden-size or residual-coupled dimensions should be blocked or ambiguous.

## Compiler analogy

This is a legality oracle and dataflow analysis pass over symbolic IR.

## What this milestone does not prove

It does not modify models, evaluate accuracy, or make ambiguous mappings safe.

## Connection to next milestone

Future work should improve Dimension IR precision with tensor-axis semantics and symbolic propagation equations.

