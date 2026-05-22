# Milestone 1: Project Setup

## What you learn

You learn how the repository defines its input model set, where local model artifacts live, and how a Hugging Face model becomes an exported ONNX graph.

## Why this milestone exists

Compiler-style analysis needs an input program. Here the input program is a model checkpoint plus, when available, an ONNX graph representation.

## Prerequisites

- Python environment installed from `requirements.txt`
- Editable package install with `pip install -e .`
- Network access if downloading the model for the first time

## Commands

Check the setup without downloading:

```bash
bash demo_scripts/run_demo_01_setup_check.sh
```

Download and export one model manually:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
```

## Main artifacts produced

- `configs/models.yaml`
- `data/models/hf/bert-base-uncased/`
- `data/models/onnx/bert-base-uncased/model.onnx`

## What to inspect

Open `configs/models.yaml` and inspect each model entry: `name`, `hf_id`, `task`, `local_dir`, `onnx_dir`, and default input shape.

## Expected interpretation

The model registry is the analysis entry point. Downloaded checkpoints and ONNX exports are ignored by git because they are reproducible artifacts.

## Compiler analogy

This milestone establishes the source program and an alternate graph-level representation, similar to loading source code and lowering it into an intermediate representation.

## What this milestone does not prove

It does not analyze pruning opportunities, prove ONNX equivalence, or modify model weights.

## Connection to next milestone

Milestone 2 parses the PyTorch and ONNX structures into structural inventories.

