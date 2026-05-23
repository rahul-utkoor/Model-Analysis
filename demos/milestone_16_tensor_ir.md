# Milestone 16: Frontend-Independent Tensor Graph IR

## What you learn

You learn how an ONNX graph summary is imported into a canonical tensor-dataflow intermediate representation containing tensor values, operations, producer/consumer links, forks, joins, semantic roles, and structural-region hints.

## Why this milestone exists

ONNX is an available frontend, not the research abstraction. Structural decomposition and pruning-propagation reasoning need a stable graph substrate that can later also be populated from PyTorch FX, MLIR, Relay, or other graph formats.

## Prerequisites

- An ONNX graph summary produced by Milestone 2

## Commands

```bash
bash demo_scripts/run_demo_16_tensor_ir.sh
```

Equivalent direct command:

```bash
python scripts/build_tensor_ir.py --model bert-base-uncased --verbose
```

Compare available Tensor IR graphs:

```bash
python scripts/compare_tensor_ir.py --models all
```

## Main artifacts produced

- `reports/tensor_ir/bert-base-uncased.md`
- `reports/tensor_ir/bert-base-uncased.json`
- `reports/tensor_ir_dumps/bert-base-uncased.tir`
- `reports/tensor_ir_stats/bert-base-uncased.md`

## What to inspect

Open the `.tir` dump and locate canonical `tensor.op` entries such as `linear`, `elementwise_join`, `shape_op`, and `softmax`. In the Markdown report, inspect operations marked as forks or joins and compare their `region_hint` values.

## Expected interpretation

The graph identifies dataflow facts without committing later analysis to ONNX naming or syntax. A join is visible as a tensor operation with multiple incoming activation values; a fork identifies a value whose pruning information may need to flow to multiple consumers.

## Compiler analogy

This is analogous to lowering a frontend format into an analysis IR before building structural regions or a control-tree-like hierarchy. Here the graph is tensor dataflow rather than control flow.

## What this milestone does not prove

It does not build the Structural Region Tree yet, infer pruning legality by itself, modify model weights, rewrite ONNX, or evaluate quality.

## Connection to next milestone

Milestone 17 constructs a Structural Region Tree over Tensor IR, collapsing recognized projections, joins, forks, axis transforms, and residual merge structure into a hierarchy used by future pruning-propagation analysis.
