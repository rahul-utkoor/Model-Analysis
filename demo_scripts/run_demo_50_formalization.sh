#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/build_static_pruning_formalization.py \
  --output-dir reports/formalization \
  --verbose
