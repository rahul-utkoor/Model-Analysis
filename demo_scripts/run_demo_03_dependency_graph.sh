#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 3 Demo: Dependency Graph ==="
echo "Model: ${MODEL}"
echo
echo "Building the conservative pruning dependency graph."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_dependency_graph.py --model ${MODEL} --require-onnx --verbose"
"${PYTHON_BIN}" scripts/build_dependency_graph.py --model "${MODEL}" --require-onnx --verbose

echo
echo "Artifacts:"
echo "  reports/dependency_graphs/${MODEL}.md"
echo "  reports/dependency_summaries/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_04_pruning_action_simulation.sh"

