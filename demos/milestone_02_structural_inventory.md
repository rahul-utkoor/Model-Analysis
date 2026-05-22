# Milestone 2: Structural Inventory

## What you learn

You learn how the repository inventories a model's PyTorch modules and ONNX graph nodes, including Linear layers, embeddings, normalization layers, attention-like names, and pruning hints.

## Why this milestone exists

Before pruning can be reasoned about, the system needs a front-end pass that describes the model structure in a durable report.

## Prerequisites

- Local Hugging Face model under `data/models/hf/bert-base-uncased/`
- ONNX export under `data/models/onnx/bert-base-uncased/model.onnx` when using `--require-onnx`

## Commands

```bash
bash demo_scripts/run_demo_02_structural_inventory.sh
```

Equivalent direct command:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
```

## Main artifacts produced

- `reports/structural_inventory/bert-base-uncased.md`
- `reports/onnx_graphs/bert-base-uncased.md`
- `reports/pruning_hints/bert-base-uncased.md`

## What to inspect

In the structural inventory, inspect parameter counts, Linear layers, embedding layers, normalization layers, attention-like modules, and MLP-like modules. In the ONNX graph report, inspect op type counts, initializers, and MatMul/Gemm nodes.

## Expected interpretation

The reports show where structural pruning might be possible and where propagation may be needed, but they do not yet encode a full dependency graph.

## Compiler analogy

This is the front-end parsing stage: model structure is converted into a structured inventory that later passes can consume.

## What this milestone does not prove

It does not establish legality of pruning or decide which dimensions can be removed safely.

## Connection to next milestone

Milestone 3 builds a dependency graph over the structural inventory.

