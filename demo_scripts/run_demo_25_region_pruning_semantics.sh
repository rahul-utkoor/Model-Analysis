#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

"${PYTHON}" scripts/build_region_pruning_semantics.py \
  --model "${MODEL}" \
  --verbose

"${PYTHON}" scripts/explain_region_pruning_semantics.py \
  --model "${MODEL}" \
  --contains "Feed Forward" \
  --limit 5

"${PYTHON}" scripts/explain_region_pruning_semantics.py \
  --model "${MODEL}" \
  --blocked-only \
  --limit 10

SAFE_MODEL="${MODEL//\//__}"
echo "Region pruning semantics:"
echo "  reports/region_pruning_semantics/${SAFE_MODEL}.json"
echo "  reports/region_pruning_semantics_dumps/${SAFE_MODEL}.rpsem"
echo "  reports/region_pruning_semantics_explanations/${SAFE_MODEL}.md"
