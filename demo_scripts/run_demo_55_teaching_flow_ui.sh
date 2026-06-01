#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f reports/final/static_pruning_propagation_final_summary.json ]]; then
  ./conda-env/bin/python scripts/build_final_static_pruning_report.py \
    --output-dir reports/final \
    --verbose
fi

(
  cd ui/pruning-analysis-explorer
  npm install
  npm run build
)

printf '\nStart the read-only teaching UI with:\n'
printf './conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777 --verbose\n'
