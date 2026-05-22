# Milestone 9: Pruning Opportunity Map

## What you learn

You learn how the repository summarizes model-level pruning opportunities, coupled dimensions, blocked regions, and structural risks.

## Why this milestone exists

The pruning map is the main research artifact. It describes the legal pruning space before any backend tries to transform weights.

## Prerequisites

- Dependency graph report
- Optional validated dependency graph from Milestone 5

## Commands

```bash
bash demo_scripts/run_demo_09_pruning_map.sh
```

Equivalent direct command:

```bash
python scripts/build_pruning_map.py --model bert-base-uncased --verbose
```

## Main artifacts produced

- `reports/model_pruning_maps/bert-base-uncased.md`
- `reports/pruning_opportunities/bert-base-uncased.md`
- `reports/propagation_constraints/bert-base-uncased.md`
- `reports/structural_risk_maps/bert-base-uncased.md`

## What to inspect

Inspect pruning dimensions, propagation constraints, opportunity risk levels, executability labels, independent opportunities, coupled opportunities, and blocked opportunities.

## Expected interpretation

MLP intermediate pruning should look structurally more promising than attention-head pruning. Residual hidden-size pruning should appear risky or blocked.

## Compiler analogy

This is the global legality space: an analysis pass identifies candidate transformations and the constraints they must satisfy.

## What this milestone does not prove

It does not select neurons by quality, execute pruning, or prove accuracy preservation.

## Connection to next milestone

Milestone 10 lowers pruning maps into explicit symbolic Dimension IR.

