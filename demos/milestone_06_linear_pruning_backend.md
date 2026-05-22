# Milestone 6: Linear Pruning Backend

## What you learn

You learn how the repository can dry-run a very narrow executable backend for `nn.Linear` row or column surgery.

## Why this milestone exists

The project is mainly about analysis, but a backend probe helps validate whether a structural hypothesis can eventually lower to model surgery.

## Prerequisites

- Local downloaded model
- Dependency graph and pruning plan inputs if executing from a plan

## Commands

Dry-run only:

```bash
python scripts/execute_pruning_plan.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.attention.self.query \
  --dim out_features \
  --indices 0,1,2,3 \
  --only-target \
  --dry-run \
  --verbose
```

## Main artifacts produced

- `reports/pruning_execution/`
- `reports/pruning_diffs/`
- `reports/rollback_manifests/`

## What to inspect

Inspect the execution report status, skipped records for dry-run mode, structural diff, and rollback manifest.

## Expected interpretation

The backend validates Linear shape surgery mechanics. It is not the central research artifact.

## Compiler analogy

This is an experimental lowering backend: a narrow path from analysis IR to a concrete artifact.

## What this milestone does not prove

It does not prove transformer-wide correctness, accuracy preservation, or legality of arbitrary dependency graph pruning.

## Connection to next milestone

Milestone 7 demonstrates that coupled dimensions need paired repair rather than isolated Linear edits.

