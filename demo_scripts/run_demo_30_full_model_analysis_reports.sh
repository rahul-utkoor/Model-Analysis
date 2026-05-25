#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-./conda-env/bin/python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 30: Full-Model Analysis Reports ==="
echo "Model: ${MODEL}"
echo
echo "This demo summarizes existing static analysis artifacts."
echo "It does not execute pruning, modify models, download models, or evaluate accuracy."
echo

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${PYTHON_BIN}" scripts/build_full_model_analysis_report.py \
  --model "${MODEL}" \
  --layers all \
  --export-onnx-subgraphs \
  --render-svg \
  --verbose

run_cmd "${PYTHON_BIN}" scripts/compare_model_analysis_reports.py --models all --verbose

echo
echo "Inspect:"
echo "  reports/model_analysis_reports/${MODEL}/index.md"
echo "  reports/model_analysis_reports/${MODEL}/layers/layer_0/index.md"
echo "  reports/model_analysis_reports/cross_model/index.md"
echo "  artifacts/model_analysis_subgraphs/${MODEL}/"
