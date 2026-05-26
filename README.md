# Model Analysis

Model Analysis is a research infrastructure project for **static structural analysis of neural-network models** with a focus on pruning legality, dependency propagation, and learner-facing explanations.

The current system analyzes models without changing them. It builds compiler-style intermediate representations, recovers pruning-relevant regions, ranks pruning opportunities, synthesizes symbolic pruning plans, validates those plans, and exposes the evidence through reports, a terminal explorer, and a React web UI.

> **Scope:** this repository currently performs static analysis, reporting, and visualization. It does **not** execute pruning, choose concrete pruning indices, modify weights, rewrite full ONNX models, download models implicitly, or evaluate accuracy unless an explicitly experimental command is invoked.

---

## Current Status

The current pipeline is complete for the configured model set.

| Model | Layers / Blocks | Subgraphs | Valid MLP/FFN plans |
| --- | ---: | ---: | ---: |
| `bert-base-uncased` | 12 | 204 | 12 |
| `distilbert-base-uncased` | 6 | 36 | 6 |
| `facebook/opt-125m` | 12 | 108 | 12 |
| `google/vit-base-patch16-224` | 12 | 72 | 12 |
| `gpt2` | 12 | 72 | 12 |

The main recovered pruning pattern is:

```text
expansion projection:  hidden_dim -> intermediate_dim
index-preserving activation
contraction projection: intermediate_dim -> hidden_dim
```

This pattern is recognized across BERT-style FFN blocks, OPT/GPT-2 decoder MLP blocks, DistilBERT FFN blocks, and ViT MLP blocks.

---

## What the Pipeline Produces

For each supported model, the pipeline can generate:

- op-semantics reports over primitive Tensor/ONNX operations,
- structural region trees and pruning semantics,
- pruning opportunity rankings,
- symbolic MLP/FFN pruning plans,
- plan validation reports,
- layer/block subgraph atlases,
- ONNX/SVG/DOT subgraph evidence artifacts,
- cross-model static coverage reports,
- rule-gap diagnosis reports,
- terminal and browser interfaces for exploration.

The analysis flow is:

```text
Model / ONNX
  -> Tensor IR
  -> Op Semantics
  -> Structural Region Tree
  -> Region Pruning Semantics
  -> Opportunity Ranking
  -> Symbolic Pruning Plan
  -> Plan Validation
  -> Layer / Block Subgraph Atlas
  -> CLI and Web Exploration
```

---

## Repository Layout

```text
configs/                         Model registry
scripts/                         Pipeline CLIs
tools/                           Explorers, viewers, and local API servers
src/model_analysis/              Reusable analysis modules
ui/pruning-analysis-explorer/    React + Vite web UI
docs/                            Usage and design notes
demos/                           Milestone walkthroughs
demo_scripts/                    Reproducible demo commands

data/models/                     Local model and ONNX artifacts, ignored by git
reports/                         Generated analysis reports, ignored by git
artifacts/                       Generated subgraphs and evidence artifacts, ignored by git
```

Important generated directories include:

```text
reports/model_analysis_reports/
reports/layer_subgraph_validation/
reports/op_semantics/
reports/region_pruning_semantics/
reports/pruning_opportunity_rankings/
reports/pruning_plans/
reports/pruning_plan_validation/
reports/static_coverage_study/
reports/rule_gap_diagnosis/
artifacts/model_analysis_subgraphs/
artifacts/layer_subgraphs/
```

---

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you are using the repository-local conda environment used in the examples, replace `python` with:

```bash
./conda-env/bin/python
```

For the web UI, install frontend dependencies:

```bash
cd ui/pruning-analysis-explorer
npm install
npm run build
cd ../..
```

---

## Supported Models

Configured models are listed in `configs/models.yaml`.

Common model names:

```text
bert-base-uncased
distilbert-base-uncased
gpt2
facebook/opt-125m
google/vit-base-patch16-224
```

Some commands also accept aliases such as `opt-125m` and `vit-base-patch16-224`, depending on registry normalization.

---

## Quick Start: Web UI

The easiest way to inspect the analysis is the browser UI.

Build the UI:

```bash
cd ui/pruning-analysis-explorer
npm install
npm run build
cd ../..
```

Start the local API/UI server:

```bash
./conda-env/bin/python tools/analysis_ui_api_server.py \
  --host 127.0.0.1 \
  --port 8777 \
  --verbose
```

Open:

```text
http://127.0.0.1:8777/
```

Suggested walkthrough:

```text
Dashboard
  -> bert-base-uncased
  -> Layer 0
  -> Feed Forward
  -> Plan
  -> Validation
  -> Artifacts

Dashboard
  -> facebook/opt-125m
  -> Layer 0
  -> MLP Block
  -> Plan
  -> Validation
```

See [`README_analysis_ui.md`](README_analysis_ui.md) for the full web UI guide.

---

## Quick Start: Terminal Explorer

The terminal explorer provides a guided console over the same reports.

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py
```

Direct launch into BERT Layer 0:

```bash
./conda-env/bin/python tools/interactive_analysis_explorer.py \
  --model bert-base-uncased \
  --layer 0 \
  --no-open
```

Typical commands inside the explorer:

```text
summary
layers
layer 0
nodes
subgraph Feed Forward
plan
validation
onnx
back
quit
```

See [`README_interactive_analysis_explorer.md`](README_interactive_analysis_explorer.md) for the command reference.

---

## Rebuild the Static Analysis Pipeline

The full demo pipeline is available through:

```bash
PYTHON=./conda-env/bin/python MODEL=bert-base-uncased \
  bash demo_scripts/run_full_analysis_pipeline.sh
```

To rebuild reports for all configured models:

```bash
./conda-env/bin/python scripts/build_static_pipeline_for_all_models.py \
  --models all \
  --build-missing-analysis \
  --build-layer-packs \
  --verbose
```

Regenerate cross-model coverage:

```bash
./conda-env/bin/python scripts/report_static_pipeline_coverage.py \
  --models all \
  --verbose
```

Open the coverage report:

```bash
open reports/static_coverage_study/index.md
```

---

## Core Analysis Commands

### Op Semantics

```bash
./conda-env/bin/python scripts/build_op_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Output:

```text
reports/op_semantics/<model>.json
reports/op_semantics_dumps/<model>.opsem
reports/op_semantics_explanations/<model>.md
```

### Region Pruning Semantics

```bash
./conda-env/bin/python scripts/build_region_pruning_semantics.py \
  --model bert-base-uncased \
  --verbose
```

Output:

```text
reports/region_pruning_semantics/<model>.json
reports/region_pruning_semantics_dumps/<model>.rpsem
reports/region_pruning_semantics_explanations/<model>.md
```

### Opportunity Ranking

```bash
./conda-env/bin/python scripts/rank_pruning_opportunities.py \
  --model bert-base-uncased \
  --verbose
```

Output:

```text
reports/pruning_opportunity_rankings/<model>.json
reports/pruning_opportunity_ranking_dumps/<model>.rank
reports/pruning_opportunity_explanations/<model>.md
```

### Symbolic Plan Synthesis

```bash
./conda-env/bin/python scripts/synthesize_pruning_plans.py \
  --model bert-base-uncased \
  --verbose
```

Output:

```text
reports/pruning_plans/<model>.json
reports/pruning_plan_dumps/<model>.plan
reports/pruning_plan_explanations/<model>.md
```

### Plan Validation

```bash
./conda-env/bin/python scripts/validate_pruning_plans.py \
  --model bert-base-uncased \
  --verbose
```

Output:

```text
reports/pruning_plan_validation/<model>.json
reports/pruning_plan_validation_dumps/<model>.pvalid
reports/pruning_plan_validation_explanations/<model>.md
```

### Layer / Block Subgraph Atlas

```bash
./conda-env/bin/python scripts/build_layer_subgraph_validation_pack.py \
  --model bert-base-uncased \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose
```

Output:

```text
reports/layer_subgraph_validation/<model>/layer_<N>/
artifacts/model_analysis_subgraphs/<model>/layers/layer_<N>/
```

### Full Model Report

```bash
./conda-env/bin/python scripts/build_full_model_analysis_report.py \
  --model bert-base-uncased \
  --layers all \
  --export-onnx-subgraphs \
  --render-svg \
  --verbose
```

Output:

```text
reports/model_analysis_reports/<model>/index.md
reports/model_analysis_reports/<model>/layers/layer_<N>/index.md
artifacts/model_analysis_subgraphs/<model>/
```

---

## What Counts as a Valid MLP/FFN Plan?

A valid symbolic intermediate-dimension pruning plan includes:

```text
1. prune expansion projection output
2. prune expansion bias, when present
3. propagate the same index set through the activation
4. prune contraction projection input
5. preserve contraction projection output hidden_dim
6. preserve residual and LayerNorm hidden dimensions
```

The plan is validated before it is reported as valid. Validation is symbolic and static; it does not mutate model weights.

---

## Cross-Model Generalization

The generic MLP/FFN fusion recognizes equivalent blocks despite different naming conventions.

| Family | Expansion | Activation | Contraction |
| --- | --- | --- | --- |
| BERT | `intermediate.dense` | GELU | `output.dense` |
| DistilBERT | `ffn.lin1` | GELU | `ffn.lin2` |
| OPT | `fc1` | activation | `fc2` |
| ViT | `mlp.fc1` | GELU | `mlp.fc2` |
| GPT-2 | `mlp.c_fc` | GELU | `mlp.c_proj` |

Attention score/context MatMuls remain blocked as direct pruning targets because they are contractions, not learned projection weights. Attention projection pruning is treated conservatively unless head-axis mapping is proven.

---

## Generated Report Safety

Generated reports and artifacts are intentionally ignored by git. They can be regenerated from the local model/ONNX artifacts.

This repository distinguishes:

```text
analysis evidence     -> reports/
visual evidence       -> artifacts/
source implementation -> src/, scripts/, tools/, ui/
```

Do not commit large generated reports, ONNX files, SVGs, or local model checkpoints unless explicitly needed for a small reproducible fixture.

---

## Tests

Run Python tests:

```bash
python -m compileall src scripts tests tools *.py
.venv/bin/pytest -q
```

Run with the conda environment:

```bash
./conda-env/bin/python -m compileall src scripts tests tools *.py
./conda-env/bin/pytest -q
```

Build the web UI:

```bash
cd ui/pruning-analysis-explorer
npm run build
cd ../..
```

---

## Documentation

Useful entry points:

- [`docs/usage.md`](docs/usage.md): command-oriented usage guide
- [`docs/design.md`](docs/design.md): design and pipeline notes
- [`docs/milestones.md`](docs/milestones.md): milestone history
- [`demos/README.md`](demos/README.md): guided demo track
- [`demos/full_research_pipeline.md`](demos/full_research_pipeline.md): complete research walkthrough
- [`README_analysis_ui.md`](README_analysis_ui.md): browser UI guide
- [`README_interactive_analysis_explorer.md`](README_interactive_analysis_explorer.md): terminal explorer guide

---

## Suggested Demo Narrative

For a compact demonstration:

1. Open the web UI.
2. Show that all five configured models are complete.
3. Select BERT Layer 0 and inspect `Feed Forward`.
4. Show its symbolic plan and validation checks.
5. Switch to OPT Layer 0 and inspect the `MLP Block`.
6. Explain that the same static pruning legality pattern applies across different model families.
7. Open the ONNX/SVG artifact to show the concrete subgraph evidence.

---

## Project Direction

Near-term research directions:

- classify remaining unknown op semantics by criticality,
- prove whether unknown ops are outside pruning-critical paths,
- extend attention head-axis mapping proofs,
- export validated symbolic plans into an executable pruning backend,
- connect static plan validation with runtime/accuracy evaluation.

The current codebase is intentionally conservative: it prefers to report constrained or blocked opportunities rather than overclaim pruning safety.
