#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"

echo "=== Milestone 13 Demo: k-Node and Join-Aware Subgraph Analysis ==="
echo "Model: ${MODEL}"
echo
echo "Analyzing directed paths and branch-merge regions from the saved ONNX graph summary."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/analyze_subgraphs.py --model ${MODEL} --max-nodes 5 --branch-depth 2 --post-join-depth 2 --verbose"
"${PYTHON_BIN}" scripts/analyze_subgraphs.py \
  --model "${MODEL}" \
  --max-nodes 5 \
  --branch-depth 2 \
  --post-join-depth 2 \
  --verbose

echo
echo "Artifacts:"
echo "  reports/subgraphs/${MODEL}.md"
echo "  reports/subgraph_patterns/${MODEL}.md"
echo "  reports/subgraph_pruning_analysis/${MODEL}.md"
echo "  reports/subgraph_dimension_evidence/${MODEL}.md"
echo "  reports/join_subgraphs/${MODEL}.md"
echo "  reports/residual_subgraphs/${MODEL}.md"
echo
echo "Next demo:"
echo "  bash demo_scripts/run_demo_09_pruning_map.sh"

