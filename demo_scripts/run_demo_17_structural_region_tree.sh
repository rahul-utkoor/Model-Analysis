#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 17 Demo: Structural Region Tree over Tensor IR ==="
echo "Model: ${MODEL}"
echo
echo "Building semantic structural regions from the persisted Tensor IR report."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_structural_region_tree.py --model ${MODEL} --verbose"
"${PYTHON_BIN}" scripts/build_structural_region_tree.py --model "${MODEL}" --verbose

echo
echo "Artifacts:"
echo "  reports/structural_region_trees/${SAFE_MODEL}.md"
echo "  reports/structural_region_dumps/${SAFE_MODEL}.srtree"
echo "  reports/structural_region_interfaces/${SAFE_MODEL}.md"
echo "  reports/structural_region_patterns/${SAFE_MODEL}.md"
echo
echo "This is compiler-style structural analysis only; it does not modify models."
echo "Next main research demo:"
echo "  bash demo_scripts/run_demo_18_region_dimension_ir.sh"
