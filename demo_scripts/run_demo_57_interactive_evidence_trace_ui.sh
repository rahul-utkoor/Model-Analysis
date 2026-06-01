#!/usr/bin/env bash
set -euo pipefail

(
  cd ui/pruning-analysis-explorer
  npm install
  npm run build
)

printf '\nStart the read-only Evidence Trace UI with:\n'
printf './conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777 --verbose\n'
printf '\nRecommended walkthrough:\n'
printf 'Pipeline Flow -> Evidence Trace -> FFN -> Attention Value -> QK Blocker -> Models -> Reports\n'
