#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-./conda-env/bin/python}"

echo "=== Milestone 31: Cross-Model Static Coverage ==="
echo
echo "This demo audits existing local static artifacts."
echo "It does not download models, execute pruning, rewrite full ONNX models, or evaluate accuracy."
echo

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${PYTHON_BIN}" scripts/build_static_pipeline_for_all_models.py \
  --models all \
  --build-missing-analysis \
  --build-layer-packs \
  --verbose

run_cmd "${PYTHON_BIN}" scripts/report_static_pipeline_coverage.py --models all --verbose
run_cmd "${PYTHON_BIN}" scripts/explain_static_pipeline_status.py --model bert-base-uncased

echo
echo "Inspect:"
echo "  reports/static_pipeline_status/bert-base-uncased.md"
echo "  reports/static_coverage_study/index.md"
