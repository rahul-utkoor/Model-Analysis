#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

"${PYTHON}" scripts/validate_pruning_plans.py \
  --model "${MODEL}" \
  --verbose

"${PYTHON}" scripts/explain_pruning_plan_validation.py \
  --model "${MODEL}" \
  --status valid \
  --limit 20

"${PYTHON}" scripts/explain_pruning_plan_validation.py \
  --model "${MODEL}" \
  --contains "Layer 0 Feed Forward"

"${PYTHON}" scripts/explain_pruning_plan_validation.py \
  --model "${MODEL}" \
  --failed-only

SAFE_MODEL="${MODEL//\//__}"
echo "Validation JSON: reports/pruning_plan_validation/${SAFE_MODEL}.json"
echo "Text dump: reports/pruning_plan_validation_dumps/${SAFE_MODEL}.pvalid"
echo "Markdown: reports/pruning_plan_validation_explanations/${SAFE_MODEL}.md"
