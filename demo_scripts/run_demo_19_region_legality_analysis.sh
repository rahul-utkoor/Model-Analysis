#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 19 Demo: Region-Aware Pruning Propagation Analysis ==="
echo "Model: ${MODEL}"
echo
echo "Listing region dimensions whose names contain 'intermediate'."
"${PYTHON_BIN}" scripts/list_region_dimensions.py \
  --model "${MODEL}" \
  --contains intermediate \
  --limit 10

echo
echo "Explaining blocked and unresolved semantic-region dimensions."
"${PYTHON_BIN}" scripts/explain_region_blocked_dimensions.py \
  --model "${MODEL}"

echo
echo "Artifacts:"
echo "  reports/region_blocked_analysis/${SAFE_MODEL}__dimension_list.md"
echo "  reports/region_blocked_analysis/${SAFE_MODEL}__blocked_dimensions.md"
echo
echo "After selecting a prunable region dimension, run:"
echo "  ${PYTHON_BIN} scripts/check_region_pruning_legality.py --model ${MODEL} --dimension-var <region_dimension_var_id> --count 4 --verbose"
echo
echo "This is static region-aware legality analysis only; it does not modify models."
