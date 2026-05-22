# Model Analysis

Model Analysis is a research scaffold for structural analysis of neural networks, with an emphasis on pruning opportunities, dependency tracking, and forward/backward propagation of pruning information across model graphs.

The first milestone is infrastructure: a clean repository structure, reproducible setup, model download scripts, ONNX export scripts, and basic inspection summaries.

## Initial Supported Models

| Name | Hugging Face ID | Task |
| --- | --- | --- |
| `bert-base-uncased` | `bert-base-uncased` | masked-lm |
| `distilbert-base-uncased` | `distilbert-base-uncased` | masked-lm |
| `gpt2` | `gpt2` | causal-lm |
| `opt-125m` | `facebook/opt-125m` | causal-lm |
| `vit-base-patch16-224` | `google/vit-base-patch16-224` | image-classification |

## Repository Layout

```text
configs/                  Model registry configuration
scripts/                  CLI utilities for downloads, ONNX export, and inspection
src/model_analysis/       Reusable Python package code
data/models/hf/           Local Hugging Face model snapshots (ignored by git)
data/models/onnx/         Exported ONNX models (ignored by git)
reports/model_summaries/  Generated Markdown summaries (ignored by git)
tests/                    Lightweight pytest coverage
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Download all configured models:

```bash
python scripts/download_models.py --model all
```

Export one model to ONNX:

```bash
python scripts/export_to_onnx.py --model bert-base-uncased
```

Inspect one local model:

```bash
python scripts/inspect_model.py --model bert-base-uncased
```

## First Push

```bash
git add .
git commit -m "Initial model analysis project scaffold"
git push -u origin main
```
