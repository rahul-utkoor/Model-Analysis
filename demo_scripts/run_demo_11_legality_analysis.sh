#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
CONTAINS="${CONTAINS:-intermediate.dense}"
LIMIT="${LIMIT:-10}"

echo "=== Milestone 11 Demo: Dimension-IR Legality Analysis ==="
echo "Model: ${MODEL}"
echo "Dimension filter: ${CONTAINS}"
echo
echo "Listing candidate Dimension IR variables."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/list_pruning_dimensions.py --model ${MODEL} --contains ${CONTAINS} --limit ${LIMIT}"
"${PYTHON_BIN}" scripts/list_pruning_dimensions.py --model "${MODEL}" --contains "${CONTAINS}" --limit "${LIMIT}"

echo
echo "Explaining blocked regions."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/explain_blocked_regions.py --model ${MODEL}"
"${PYTHON_BIN}" scripts/explain_blocked_regions.py --model "${MODEL}"

echo
echo "To run a legality check, choose a var_id from the dimension list and run:"
echo "  ${PYTHON_BIN} scripts/check_pruning_legality.py --model ${MODEL} --dimension-var <dimension_var_id> --count 4 --verbose"
echo
echo "Artifacts:"
echo "  reports/ir_analysis/${MODEL}__dimension_list.md"
echo "  reports/ir_analysis/${MODEL}__blocked_regions.md"
echo "  reports/legality_checks/"
echo "  reports/propagation_slices/"
echo "  reports/repair_sets/"

