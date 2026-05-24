# Milestone 21: Stepwise Control-Tree Trace

## What you learn

You learn how a compiler-style Structural Region Tree can be explained as a sequence of graph reductions: primitive TensorOps are initialized, semantic patterns are recognized, matched nodes are collapsed into abstract regions, and the current abstract graph is updated after each collapse.

## Why this milestone exists

The final region tree is compact but hides the construction process. This milestone makes the construction visible for teaching, debugging, and research walkthroughs.

## Prerequisites

- Tensor IR from Milestone 16 under `reports/tensor_ir/<model>.json`
- Recommended: Structural Region Tree from Milestone 17
- Recommended: Semantic fusion report from Milestone 20

## Commands

```bash
bash demo_scripts/run_demo_21_control_tree_trace.sh
```

Equivalent commands:

```bash
python scripts/build_control_tree_trace.py --model bert-base-uncased --format all --max-dot-steps 20 --verbose
python tools/export_control_tree_trace_mindnode.py --model bert-base-uncased
```

## Main artifacts produced

- `reports/control_tree_steps/bert-base-uncased.md`
- `reports/control_tree_steps/bert-base-uncased.json`
- `reports/control_tree_step_dumps/bert-base-uncased.ctrace`
- `reports/control_tree_step_graphs/bert-base-uncased/step_000.dot`
- `reports/control_tree_step_summaries/bert-base-uncased.md`
- `reports/mindnode_outlines/bert-base-uncased.control_tree_steps.mindnode.txt`

## What to inspect

Open the Markdown trace and read the step table. Then inspect the `.ctrace` dump for a compact textual version. DOT files under `reports/control_tree_step_graphs/` show snapshots of the active abstract graph after selected steps.

## Expected interpretation

Early steps contain primitive TensorOps. Later steps show collapses such as `LinearProjectionRegion`, `ActivationRegion`, `FeedForwardRegion`, and `ResidualMergeRegion`. Skip steps indicate candidates already represented by an earlier abstract region.

## Compiler analogy

This is the neural dataflow analogue of interval or structural control-flow analysis: identify a region, collapse it into an abstract node, update the graph, and repeat until a hierarchy emerges.

## What this milestone does not prove

It does not execute pruning, modify model weights, rewrite ONNX, evaluate accuracy, or replace the final Structural Region Tree as the authoritative hierarchy.

## Connection to next milestone

The trace makes region construction explainable. Future work can attach richer propagation diagnostics to each collapse and show how region-aware dimensions are introduced at each abstraction level.
