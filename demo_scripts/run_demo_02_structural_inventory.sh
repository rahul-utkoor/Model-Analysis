#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 2 Demo: Structural Inventory ==="
echo "Model: ${MODEL}"
echo
echo "Running structural inventory with required ONNX evidence."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/generate_structural_inventory.py --model ${MODEL} --require-onnx"
"${PYTHON_BIN}" scripts/generate_structural_inventory.py --model "${MODEL}" --require-onnx

echo
echo "Artifacts:"
echo "  reports/structural_inventory/${MODEL}.md"
echo "  reports/onnx_graphs/${MODEL}.md"
echo "  reports/pruning_hints/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_03_dependency_graph.sh"

