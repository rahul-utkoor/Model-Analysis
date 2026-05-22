# Milestone 3: Dependency Graph

## What you learn

You learn how prunable units and propagation edges are represented as a conservative graph.

## Why this milestone exists

Pruning one layer often affects another. A dependency graph records these relationships before any pruning action is attempted.

## Prerequisites

- Structural inventory reports from Milestone 2
- ONNX graph summary when using `--require-onnx`

## Commands

```bash
bash demo_scripts/run_demo_03_dependency_graph.sh
```

Equivalent direct command:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx --verbose
```

## Main artifacts produced

- `reports/dependency_graphs/bert-base-uncased.md`
- `reports/dependency_summaries/bert-base-uncased.md`

## What to inspect

Inspect prunable units, dependency edges, coupled groups, ambiguous units, high-value pruning targets, and manual-review items.

## Expected interpretation

MLP expansion/projection relationships and Q/K/V coupling should appear as structural dependencies. Residuals, normalization, and unknown mappings should remain conservative.

## Compiler analogy

This is a use-def-like dependency graph: a transformation on one dimension can force updates to users or constraints from definitions.

## What this milestone does not prove

It does not prove that a proposed pruning action is legal or executable.

## Connection to next milestone

Milestone 4 uses the dependency graph to simulate pruning actions without touching weights.

