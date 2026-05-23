# Milestone 14: DAG Motif and Multi-Join Region Analysis

## What you learn

You learn how the repository recognizes bounded multi-branch ONNX regions: forks, reconvergent diamonds, and join-fork-join structures.

## Why this milestone exists

Path subgraphs represent sequential operations. Join-centered subgraphs represent one merge. Neither fully captures:

```text
A -> C
B -> C
C -> D
C -> E
D -> F
E -> F
```

In this motif, `C` is a join and a fork, while `F` is a reconvergent join. Pruning information may need to satisfy constraints across every branch and both merge points.

## Prerequisites

- ONNX graph summary from Milestone 2
- Milestone 13 is recommended context for path and single-join reports

## Commands

```bash
bash demo_scripts/run_demo_14_dag_region_analysis.sh
```

Equivalent direct command:

```bash
python scripts/analyze_dag_regions.py \
  --model bert-base-uncased \
  --max-branch-depth 4 \
  --verbose
```

## Main artifacts produced

- `reports/dag_regions/bert-base-uncased.md`
- `reports/dag_region_patterns/bert-base-uncased.md`
- `reports/dag_region_pruning_evidence/bert-base-uncased.md`

## What to inspect

Open the DAG region report and inspect `region_kind`, `fork_nodes`, `join_nodes`, `branch_paths`, and `suggested_constraints`. Look specifically for `join_fork_join` patterns and evidence such as `fanout_same_indices`, `branch_output_compatibility`, or `residual_equal_shape`.

## Expected interpretation

A fork means a producer dimension may constrain multiple consumers. A diamond means branches must be compatible when they reconverge. A join-fork-join region means a dimension participates in constraints before branching, throughout each branch, and again at reconvergence.

## Compiler analogy

This is bounded DAG motif matching and dataflow-region analysis. It augments local instruction patterns with multi-branch regions where transformation legality must account for fanout and reconvergence.

## What this milestone does not prove

It does not alter pruning maps or Dimension IR automatically, execute pruning, modify weights, or prove semantic correctness.

## Connection to next milestone

Future work can lower DAG-region evidence into stronger pruning-map constraints and Dimension IR equations for fanout and reconvergent branch legality.

