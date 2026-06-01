#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/build_attention_value_path_subgraphs.py \
  --model facebook/opt-125m \
  --layer 0 \
  --export-onnx \
  --render-svg \
  --verbose

"$PYTHON" scripts/explain_attention_value_path_subgraph.py \
  --model facebook/opt-125m \
  --layer 0 \
  --status seedable

"$PYTHON" -m experimental.mlir_evidence_coverage.cli \
  --models facebook__opt-125m \
  --layers layer0 \
  --patterns ATTENTION_VALUE_PATH \
  --output-dir reports/mlir_evidence_coverage_opt_value_path \
  --format both \
  --verbose
