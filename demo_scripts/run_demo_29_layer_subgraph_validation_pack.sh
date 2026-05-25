#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
LAYER="${LAYER:-0}"

"${PYTHON}" scripts/build_layer_subgraph_validation_pack.py \
  --model "${MODEL}" \
  --layer "${LAYER}" \
  --export-onnx \
  --render-svg \
  --verbose

"${PYTHON}" scripts/explain_layer_subgraph_validation.py \
  --model "${MODEL}" \
  --layer "${LAYER}" \
  --contains "Feed Forward"

"${PYTHON}" scripts/explain_layer_subgraph_validation.py \
  --model "${MODEL}" \
  --layer "${LAYER}" \
  --class safe

SAFE_MODEL="${MODEL//\//__}"
echo "Index: reports/layer_subgraph_validation/${SAFE_MODEL}/layer_${LAYER}/index.md"
echo "Reports: reports/layer_subgraph_validation/${SAFE_MODEL}/layer_${LAYER}/<node>/"
echo "Artifacts: artifacts/layer_subgraphs/${SAFE_MODEL}/layer_${LAYER}/<node>/"
