#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

echo "=== Milestone 32: Rule-Gap Diagnosis and FFN Generalization ==="
echo "This demo reads existing static artifacts and writes diagnosis reports."
echo "It does not modify models, execute pruning, download models, or rewrite ONNX."

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${PYTHON_BIN}" scripts/diagnose_rule_gaps.py --models all --verbose
run_cmd "${PYTHON_BIN}" scripts/explain_rule_gap.py --model facebook/opt-125m
run_cmd "${PYTHON_BIN}" scripts/compare_rule_gaps.py --models all
run_cmd "${PYTHON_BIN}" scripts/report_static_pipeline_coverage.py --models all --verbose

echo
echo "Main artifacts:"
echo "  reports/rule_gap_diagnosis/"
echo "  reports/rule_gap_diagnosis_compare/index.md"
echo "  reports/static_coverage_study/index.md"
