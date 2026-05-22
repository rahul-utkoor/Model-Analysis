# Usage Guide

This repository is organized around a staged analysis pipeline. Each stage writes durable artifacts under `reports/` so later stages can run without reloading or re-exporting models.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Supported Models

Configured models live in `configs/models.yaml`.

Current model names:

```text
bert-base-uncased
distilbert-base-uncased
gpt2
opt-125m
vit-base-patch16-224
```

Most commands accept either the configured name or the Hugging Face ID where applicable. For example, `opt-125m` and `facebook/opt-125m` both resolve to the OPT entry.

## Model Download

Download one model:

```bash
python scripts/download_models.py --model bert-base-uncased
```

Download all configured models:

```bash
python scripts/download_models.py --model all
```

Force a redownload:

```bash
python scripts/download_models.py --model bert-base-uncased --force
```

Use a custom Hugging Face cache directory:

```bash
python scripts/download_models.py --model bert-base-uncased --cache-dir /path/to/cache
```

Downloaded model files are written under `data/models/hf/` and are ignored by git.

## ONNX Export

Export one model:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased
```

Export all downloaded models:

```bash
python scripts/export_to_onnx.py --model all
```

Use a different opset:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased --opset 17
```

ONNX exports are written under `data/models/onnx/` and are ignored by git.

## Quick Inspection

Print a lightweight local model summary and write a Markdown summary:

```bash
python scripts/inspect_model.py --model bert-base-uncased
```

## Structural Inventory

Generate PyTorch structural inventory, optional ONNX graph summary, and pruning hints:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Require the ONNX export to be present:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
```

Control structural and ONNX report formats:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased --format json
python scripts/generate_structural_inventory.py --model bert-base-uncased --format md
python scripts/generate_structural_inventory.py --model bert-base-uncased --format both
```

Generated outputs:

```text
reports/structural_inventory/<model>.json
reports/structural_inventory/<model>.md
reports/onnx_graphs/<model>.json
reports/onnx_graphs/<model>.md
reports/pruning_hints/<model>.md
```

## Dependency Graph Construction

Build the pruning-dependency graph from structural inventory reports:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Require the ONNX graph summary:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx
```

Build from PyTorch inventory only:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --torch-only
```

Print graph statistics:

```bash
python scripts/build_dependency_graph.py --model bert-base-uncased --verbose
```

Generated outputs:

```text
reports/dependency_graphs/<model>.json
reports/dependency_graphs/<model>.md
reports/dependency_summaries/<model>.json
reports/dependency_summaries/<model>.md
```

## Recommended Single-Model Flow

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased --require-onnx
python scripts/build_dependency_graph.py --model bert-base-uncased --require-onnx --verbose
```

## Recommended All-Model Flow

```bash
python scripts/download_models.py --model all
python scripts/export_to_onnx.py --model all
python scripts/generate_structural_inventory.py --model all --require-onnx
python scripts/build_dependency_graph.py --model all --require-onnx --verbose
```

## Validation

```bash
python -m compileall src scripts tests
.venv/bin/pytest -q
```

If the virtual environment does not exist but dependencies are installed globally:

```bash
pytest -q
```
