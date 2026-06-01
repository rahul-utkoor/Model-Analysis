#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/build_attention_value_path_subgraphs.py \
  --model gpt2 \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose

"$PYTHON" scripts/build_attention_value_path_subgraphs.py \
  --model google/vit-base-patch16-224 \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose

"$PYTHON" -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers all \
  --output-dir reports/all_model_plan_proof \
  --format both \
  --build-missing-value-paths \
  --verbose
