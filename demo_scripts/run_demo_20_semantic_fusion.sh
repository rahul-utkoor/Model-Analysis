#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 20 Demo: Semantic Fusion for Feed-Forward Regions ==="
echo "Model: ${MODEL}"

"${PYTHON_BIN}" scripts/analyze_semantic_fusion.py --model "${MODEL}" --verbose
"${PYTHON_BIN}" scripts/build_structural_region_tree.py --model "${MODEL}" --verbose
"${PYTHON_BIN}" scripts/build_region_dimension_ir.py --model "${MODEL}" --verbose
"${PYTHON_BIN}" scripts/list_region_dimensions.py --model "${MODEL}" --contains intermediate --limit 20

echo
echo "Artifacts:"
echo "  reports/semantic_fusion/${SAFE_MODEL}.md"
echo "  reports/fused_region_patterns/${SAFE_MODEL}.md"
echo "  reports/structural_region_trees/${SAFE_MODEL}.md"
echo "  reports/region_dimension_ir/${SAFE_MODEL}.md"
echo
echo "Semantic fusion recovers high-level activation/feed-forward regions without modifying models."
