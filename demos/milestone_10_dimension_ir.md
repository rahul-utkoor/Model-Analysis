# Milestone 10: Dimension IR

## What you learn

You learn how pruning dimensions become symbolic variables with index variables, constraint equations, equivalence classes, blocked dimensions, and unresolved constraints.

## Why this milestone exists

Descriptive pruning maps are useful, but compiler-style legality analysis needs a more explicit symbolic IR.

## Prerequisites

- Model pruning map from Milestone 9

## Commands

```bash
bash demo_scripts/run_demo_10_dimension_ir.sh
```

Equivalent direct command:

```bash
python scripts/build_dimension_ir.py --model bert-base-uncased --verbose
```

## Main artifacts produced

- `reports/dimension_ir/bert-base-uncased.md`
- `reports/pruning_ir_dumps/bert-base-uncased.pir`
- `reports/constraint_equations/bert-base-uncased.md`
- `reports/dimension_equivalence/bert-base-uncased.md`

## What to inspect

Open the `.pir` dump and inspect `pruning.dim`, `pruning.index`, `pruning.constraint`, and `pruning.eq_class` entries.

## Expected interpretation

Dimensions that must be pruned together should form constraints and equivalence classes. Unknown or blocking constraints remain explicit.

## Compiler analogy

This is an IR construction pass: dimensions become SSA-like symbolic values and constraints become equations over those values.

## What this milestone does not prove

It does not solve every constraint or transform the model.

## Connection to next milestone

Milestone 11 runs legality checks and slice extraction over the Dimension IR.

