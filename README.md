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
reports/structural_inventory/  Generated PyTorch inventory reports (ignored by git)
reports/onnx_graphs/      Generated ONNX graph reports (ignored by git)
reports/pruning_hints/    Generated pruning hint reports (ignored by git)
reports/dependency_graphs/  Generated dependency graph reports (ignored by git)
reports/dependency_summaries/  Generated dependency analyzer summaries (ignored by git)
docs/                     Design notes, milestone notes, and detailed usage
tests/                    Lightweight pytest coverage
```

## Documentation

Detailed project documentation lives in:

- [Usage Guide](docs/usage.md)
- [Design Notes](docs/design.md)
- [Milestones](docs/milestones.md)

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

## Structural Inventory

Generate structural inventory reports for one downloaded model:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Suggested first flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
```

Generated outputs:

```text
reports/structural_inventory/<model>.json  PyTorch module, parameter, layer, and pruning-group inventory
reports/structural_inventory/<model>.md    Human-readable PyTorch structural inventory
reports/onnx_graphs/<model>.json           ONNX graph node, initializer, IO, and pruning-relevant node inventory
reports/onnx_graphs/<model>.md             Human-readable ONNX graph summary
reports/pruning_hints/<model>.md           Conservative structural pruning hints and dependency caveats
```

Use `--require-onnx` when an ONNX report must exist, and `--format json|md|both` to control generated structural and ONNX report formats.

## Dependency Graph Construction

Build a conservative pruning-dependency graph from existing structural inventory reports:

```bash
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Full single-model flow:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
```

Generated dependency outputs:

```text
reports/dependency_graphs/<model>.json      Dependency graph IR with prunable units and dependency edges
reports/dependency_graphs/<model>.md        Human-readable dependency graph report
reports/dependency_summaries/<model>.json   Analyzer summary for targets, paths, constraints, and review items
reports/dependency_summaries/<model>.md     Human-readable dependency summary
```

The dependency graph is a conservative static pruning-dependency IR. It is not an executable pruning transform yet and does not prove that a pruning decision is safe.

## Pruning Action Simulation

Generate and simulate small candidate pruning actions:

```bash
python scripts/download_models.py --model bert-base-uncased
python scripts/export_to_onnx.py --model bert-base-uncased
python scripts/generate_structural_inventory.py --model bert-base-uncased
python scripts/build_dependency_graph.py --model bert-base-uncased
python scripts/generate_candidate_actions.py --model bert-base-uncased --simulate --limit 5
```

Simulate one manual action:

```bash
python scripts/simulate_pruning_action.py \
  --model bert-base-uncased \
  --target-unit <unit_id> \
  --dim out_features \
  --indices 0,1,2,3 \
  --verbose
```

This does not prune weights, rewrite PyTorch modules, or rewrite ONNX. It only simulates dependency propagation and emits candidate plans, propagation traces, and validation diagnostics. Ambiguous results are expected for complex transformer structures until later milestones add stronger PyTorch/ONNX correspondence and executable pruning transforms.

## First Push

```bash
git add .
git commit -m "Initial model analysis project scaffold"
git push -u origin main
```
