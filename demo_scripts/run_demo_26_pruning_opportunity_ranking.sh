#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

"${PYTHON}" scripts/rank_pruning_opportunities.py \
  --model "${MODEL}" \
  --verbose

"${PYTHON}" scripts/explain_pruning_opportunity.py \
  --model "${MODEL}" \
  --class safe \
  --limit 20

"${PYTHON}" scripts/explain_pruning_opportunity.py \
  --model "${MODEL}" \
  --class constrained \
  --limit 20

"${PYTHON}" scripts/explain_pruning_opportunity.py \
  --model "${MODEL}" \
  --contains "Attention Score MatMul"

SAFE_MODEL="${MODEL//\//__}"
echo "Ranking report: reports/pruning_opportunity_rankings/${SAFE_MODEL}.json"
echo "Text dump: reports/pruning_opportunity_ranking_dumps/${SAFE_MODEL}.rank"
echo "Markdown: reports/pruning_opportunity_explanations/${SAFE_MODEL}.md"

