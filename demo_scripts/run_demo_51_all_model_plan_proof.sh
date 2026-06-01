#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" -m experimental.all_model_plan_proof.cli \
  --models all \
  --layers all \
  --output-dir reports/all_model_plan_proof \
  --format both \
  --build-missing-value-paths \
  --verbose
