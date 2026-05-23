#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODEL="${MODEL:-bert-base-uncased}"
SAFE_MODEL="${MODEL//\//__}"

echo "=== Milestone 18 Demo: Region-Aware Dimension IR ==="
echo "Model: ${MODEL}"
echo
echo "Deriving region-scoped symbolic dimensions and constraints from the Structural Region Tree."
echo "Command:"
echo "  ${PYTHON_BIN} scripts/build_region_dimension_ir.py --model ${MODEL} --verbose"
"${PYTHON_BIN}" scripts/build_region_dimension_ir.py --model "${MODEL}" --verbose

echo
echo "Artifacts:"
echo "  reports/region_dimension_ir/${SAFE_MODEL}.md"
echo "  reports/region_pruning_ir_dumps/${SAFE_MODEL}.rdim"
echo "  reports/region_constraint_equations/${SAFE_MODEL}.md"
echo "  reports/region_dimension_equivalence/${SAFE_MODEL}.md"
echo
echo "This is static region-aware dimension analysis only; it does not modify models."
echo "Next main research demo:"
echo "  bash demo_scripts/run_demo_19_region_legality_analysis.sh"
