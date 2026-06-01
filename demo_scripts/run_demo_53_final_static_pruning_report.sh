#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/build_final_static_pruning_report.py \
  --output-dir reports/final \
  --verbose
