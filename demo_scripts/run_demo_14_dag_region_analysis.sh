#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 14 Demo: DAG Motif and Multi-Join Region Analysis ==="
echo "Model: ${MODEL}"
echo
echo "Analyzing bounded fork, diamond, and join-fork-join regions from the saved ONNX graph summary."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/analyze_dag_regions.py --model ${MODEL} --max-branch-depth 4 --verbose"
"${PYTHON_BIN}" scripts/analyze_dag_regions.py \
  --model "${MODEL}" \
  --max-branch-depth 4 \
  --verbose

echo
echo "Artifacts:"
echo "  reports/dag_regions/${MODEL}.md"
echo "  reports/dag_region_patterns/${MODEL}.md"
echo "  reports/dag_region_pruning_evidence/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_15_netron_subgraph_export.sh"
