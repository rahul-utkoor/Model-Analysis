#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

"${PYTHON}" scripts/synthesize_pruning_plans.py \
  --model "${MODEL}" \
  --verbose

"${PYTHON}" scripts/explain_pruning_plan.py \
  --model "${MODEL}" \
  --status ready_symbolic \
  --limit 20

"${PYTHON}" scripts/explain_pruning_plan.py \
  --model "${MODEL}" \
  --contains "Layer 0 Feed Forward"

SAFE_MODEL="${MODEL//\//__}"
echo "Plan JSON: reports/pruning_plans/${SAFE_MODEL}.json"
echo "Text dump: reports/pruning_plan_dumps/${SAFE_MODEL}.plan"
echo "Markdown: reports/pruning_plan_explanations/${SAFE_MODEL}.md"
