# Milestone 5: Correspondence and Shape Evidence

## What you learn

You learn how PyTorch modules are heuristically linked to ONNX nodes and initializers, and how ONNX tensor-shape metadata supports dependency validation.

## Why this milestone exists

Dependency graphs built from names alone are weak. Correspondence and shape evidence connect high-level modules to lower-level graph evidence.

## Prerequisites

- Structural inventory and ONNX graph reports
- Dependency graph report

## Commands

```bash
bash demo_scripts/run_demo_05_correspondence_shape_evidence.sh
```

Equivalent direct command:

```bash
python scripts/build_correspondence.py --model bert-base-uncased --require-dependency-graph --verbose
```

## Main artifacts produced

- `reports/correspondence/bert-base-uncased.md`
- `reports/shape_evidence/bert-base-uncased.md`
- `reports/validated_dependency_graphs/bert-base-uncased.md`

## What to inspect

Inspect module-to-node matches, parameter evidence, tensor shapes, validated units, shape-supported edges, and manual-review items.

## Expected interpretation

Some Linear modules may correspond to MatMul/Gemm nodes with initializer evidence. Some ONNX nodes remain unmapped and should stay conservative.

## Compiler analogy

This is source-to-lowered-IR debug information: it links high-level structures to lower-level graph operations.

## What this milestone does not prove

Correspondence is heuristic and static. It does not prove semantic equivalence or make pruning executable.

## Connection to next milestone

Milestones 6-8 use this analysis direction as context for optional experimental backend probes. The main research path resumes at Milestone 9.

