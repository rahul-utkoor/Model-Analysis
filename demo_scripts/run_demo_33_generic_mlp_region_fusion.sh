#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-distilbert-base-uncased}"

echo "=== Milestone 33: Generic MLP Region Fusion ==="
echo "Model: ${MODEL}"
echo "This demo reads existing static artifacts and writes analysis reports."
echo "It does not modify models, execute pruning, download models, or rewrite ONNX."

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_cmd "${PYTHON_BIN}" scripts/build_region_pruning_semantics.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/rank_pruning_opportunities.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/synthesize_pruning_plans.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/validate_pruning_plans.py --model "${MODEL}" --verbose
run_cmd "${PYTHON_BIN}" scripts/diagnose_rule_gaps.py --models all --verbose

echo
echo "Main artifacts:"
echo "  reports/region_pruning_semantics/${MODEL}.json"
echo "  reports/pruning_opportunity_rankings/${MODEL}.json"
echo "  reports/pruning_plans/${MODEL}.json"
echo "  reports/pruning_plan_validation/${MODEL}.json"
