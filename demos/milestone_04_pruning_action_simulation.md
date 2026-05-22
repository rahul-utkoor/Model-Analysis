# Milestone 4: Pruning Action Simulation

## What you learn

You learn how a proposed pruning action is dry-run across the dependency graph and turned into a pruning plan with propagation traces.

## Why this milestone exists

Before modifying weights, the repository needs a way to ask "what would this pruning action affect?"

## Prerequisites

- Dependency graph report from Milestone 3

## Commands

```bash
bash demo_scripts/run_demo_04_pruning_action_simulation.sh
```

Manual direct example:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense \
  --dim out_features \
  --indices 0,1,2,3 \
  --allow-ambiguous \
  --verbose
```

## Main artifacts produced

- `reports/pruning_action_checks/bert-base-uncased__candidate_actions.json`
- `reports/pruning_plans/`
- `reports/propagation_traces/`

## What to inspect

Open a generated pruning plan and inspect status, requested action, affected units, propagation trace, constraints, conflicts, and manual-review items.

## Expected interpretation

Simple local actions may be valid locally. Transformer actions are often ambiguous because residual, attention, reshape, and tying evidence is incomplete.

## Compiler analogy

This is dry-run transformation planning: the compiler predicts which IR values must change if a transformation is applied.

## What this milestone does not prove

It does not modify model weights, repair modules, or prove end-to-end correctness.

## Connection to next milestone

Milestone 5 adds PyTorch-to-ONNX correspondence and static shape evidence to reduce ambiguity.

