#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 16 Demo: Frontend-Independent Tensor Graph IR ==="
echo "Model: ${MODEL}"
echo
echo "Importing the saved ONNX frontend summary into Tensor IR."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_tensor_ir.py --model ${MODEL} --verbose"
"${PYTHON_BIN}" scripts/build_tensor_ir.py --model "${MODEL}" --verbose

echo
echo "Artifacts:"
echo "  reports/tensor_ir/${SAFE_MODEL}.md"
echo "  reports/tensor_ir_dumps/${SAFE_MODEL}.tir"
echo "  reports/tensor_ir_stats/${SAFE_MODEL}.md"
echo
echo "Tensor IR is frontend-independent; ONNX is the currently implemented importer."
echo
echo "Next region-hierarchy demo:"
echo "  bash demo_scripts/run_demo_17_structural_region_tree.sh"
