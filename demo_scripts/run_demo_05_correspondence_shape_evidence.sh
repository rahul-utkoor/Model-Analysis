#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 5 Demo: Correspondence and Shape Evidence ==="
echo "Model: ${MODEL}"
echo
echo "Building PyTorch-to-ONNX correspondence and static shape evidence."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_correspondence.py --model ${MODEL} --require-dependency-graph --verbose"
"${PYTHON_BIN}" scripts/build_correspondence.py --model "${MODEL}" --require-dependency-graph --verbose

echo
echo "Artifacts:"
echo "  reports/correspondence/${MODEL}.md"
echo "  reports/shape_evidence/${MODEL}.md"
echo "  reports/validated_dependency_graphs/${MODEL}.md"
echo
echo "Next main research demo:"
echo "  bash demo_scripts/run_demo_13_subgraph_analysis.sh"
