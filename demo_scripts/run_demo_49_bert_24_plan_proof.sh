#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/build_attention_value_path_subgraphs.py \
  --model bert-base-uncased \
  --layers all \
  --export-onnx \
  --render-svg \
  --verbose

"$PYTHON" -m experimental.mlir_evidence_coverage.cli \
  --models bert-base-uncased \
  --layers all \
  --patterns FFN_MLP_INTERMEDIATE,ATTENTION_VALUE_PATH \
  --output-dir reports/mlir_evidence_coverage_bert_24_plan \
  --format both \
  --verbose

"$PYTHON" -m experimental.bert_24_plan_proof.cli \
  --output-dir reports/bert_24_plan_proof \
  --verbose
