#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 9 Demo: Pruning Opportunity Map ==="
echo "Model: ${MODEL}"
echo
echo "Building the model-level pruning opportunity map."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_pruning_map.py --model ${MODEL} --verbose"
"${PYTHON_BIN}" scripts/build_pruning_map.py --model "${MODEL}" --verbose

echo
echo "Artifacts:"
echo "  reports/model_pruning_maps/${MODEL}.md"
echo "  reports/pruning_opportunities/${MODEL}.md"
echo "  reports/propagation_constraints/${MODEL}.md"
echo "  reports/structural_risk_maps/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_10_dimension_ir.sh"

