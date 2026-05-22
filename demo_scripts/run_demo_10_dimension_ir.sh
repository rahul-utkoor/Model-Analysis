#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 10 Demo: Dimension IR ==="
echo "Model: ${MODEL}"
echo
echo "Building symbolic Dimension IR and MLIR-like text dump."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_dimension_ir.py --model ${MODEL} --verbose"
"${PYTHON_BIN}" scripts/build_dimension_ir.py --model "${MODEL}" --verbose

echo
echo "Artifacts:"
echo "  reports/dimension_ir/${MODEL}.md"
echo "  reports/pruning_ir_dumps/${MODEL}.pir"
echo "  reports/constraint_equations/${MODEL}.md"
echo "  reports/dimension_equivalence/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_11_legality_analysis.sh"

