#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
LIMIT="${LIMIT:-5}"

echo "=== Milestone 4 Demo: Pruning Action Simulation ==="
echo "Model: ${MODEL}"
echo "Candidate limit: ${LIMIT}"
echo
echo "Generating and simulating small candidate pruning actions."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/generate_candidate_actions.py --model ${MODEL} --simulate --limit ${LIMIT}"
"${PYTHON_BIN}" scripts/generate_candidate_actions.py --model "${MODEL}" --simulate --limit "${LIMIT}"

echo
echo "Optional manual action example:"
echo "  ${PYTHON_BIN} scripts/simulate_pruning_action.py --model ${MODEL} --target-unit torch:linear:bert.encoder.layer.0.intermediate.dense --dim out_features --indices 0,1,2,3 --allow-ambiguous --verbose"
echo
echo "Artifacts:"
echo "  reports/pruning_action_checks/${MODEL}__candidate_actions.md"
echo "  reports/pruning_plans/"
echo "  reports/propagation_traces/"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_05_correspondence_shape_evidence.sh"

