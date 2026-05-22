#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 1 Demo: Project Setup Check ==="
echo "Model: ${MODEL}"
echo
echo "Checking repository inputs without downloading models."
test -f configs/models.yaml
test -f scripts/download_models.py
test -f scripts/export_to_onnx.py

echo
echo "Configured models:"
"${PYTHON_BIN}" -c "from model_analysis.registry import list_models; print('\n'.join(m['name'] for m in list_models()))"

echo
echo "Commands to create the first model artifacts:"
echo "  ${PYTHON_BIN} scripts/download_models.py --model ${MODEL}"
echo "  ${PYTHON_BIN} scripts/export_to_onnx.py --model ${MODEL}"
echo
echo "Artifacts after running those commands:"
echo "  data/models/hf/${MODEL}/"
echo "  data/models/onnx/${MODEL}/model.onnx"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_02_structural_inventory.sh"

