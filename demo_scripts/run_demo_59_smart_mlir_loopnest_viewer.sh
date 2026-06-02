#!/usr/bin/env bash
set -euo pipefail

(
  cd ui/pruning-analysis-explorer
  npm install
  npm run build
)

printf '\nStart the read-only smart MLIR viewer with:\n'
printf './conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777 --verbose\n'
printf '\nRecommended walkthrough:\n'
printf 'Evidence Trace -> FFN -> Real Artifact -> MLIR -> lowered affine -> focused loop/access regions -> Full file\n'
