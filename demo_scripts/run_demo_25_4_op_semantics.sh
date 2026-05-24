#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

"${PYTHON}" scripts/build_op_semantics.py \
  --model "${MODEL}" \
  --verbose

"${PYTHON}" scripts/explain_op_semantics.py \
  --model "${MODEL}" \
  --semantic-kind attention_score_matmul \
  --limit 5

"${PYTHON}" scripts/explain_op_semantics.py \
  --model "${MODEL}" \
  --category parameterized_projection \
  --limit 10

SAFE_MODEL="${MODEL//\//__}"
echo "Op semantics report: reports/op_semantics/${SAFE_MODEL}.json"
echo "Text dump: reports/op_semantics_dumps/${SAFE_MODEL}.opsem"
echo "Markdown: reports/op_semantics_explanations/${SAFE_MODEL}.md"

