#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 15 Demo: Netron ONNX Subgraph Export ==="
echo "Model: ${MODEL}"
echo
echo "Exporting a small curated set of visualization-only ONNX fragments."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/export_demo_subgraphs.py --model ${MODEL} --max-per-category 3 --verbose"
"${PYTHON_BIN}" scripts/export_demo_subgraphs.py \
  --model "${MODEL}" \
  --max-per-category 3 \
  --verbose

echo
echo "Netron index:"
echo "  reports/netron_subgraph_index/${SAFE_MODEL}__demo.md"
echo
echo "Original full model baseline:"
echo "  netron data/models/onnx/${SAFE_MODEL}/model.onnx"
echo
echo "Example Netron command:"
echo "  netron artifacts/subgraph_onnx/${SAFE_MODEL}/demo/<subgraph>.onnx"
echo
echo "These files are visualization artifacts only; the source ONNX model is unchanged."
