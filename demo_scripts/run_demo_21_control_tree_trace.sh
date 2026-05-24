#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 21 Demo: Stepwise Control-Tree Trace ==="
echo "Model: ${MODEL}"

echo
echo "Building the control-tree construction trace..."
"${PYTHON_BIN}" scripts/build_control_tree_trace.py \
  --model "${MODEL}" \
  --format all \
  --max-dot-steps 20 \
  --verbose

echo
echo "Exporting the trace as a MindNode outline..."
"${PYTHON_BIN}" tools/export_control_tree_trace_mindnode.py \
  --model "${MODEL}"

echo
echo "Artifacts:"
echo "  reports/control_tree_steps/${SAFE_MODEL}.md"
echo "  reports/control_tree_step_dumps/${SAFE_MODEL}.ctrace"
echo "  reports/control_tree_step_graphs/${SAFE_MODEL}/"
echo "  reports/mindnode_outlines/${SAFE_MODEL}.control_tree_steps.mindnode.txt"
echo
echo "Next demo: run region-aware Dimension IR and legality analysis."
