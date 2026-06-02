#!/usr/bin/env bash
set -euo pipefail

./conda-env/bin/python scripts/build_ui_mlir_artifact_index.py \
  --output reports/ui_artifact_index/index.json \
  --verbose

(
  cd ui/pruning-analysis-explorer
  npm install
  npm run build
)

printf '\nStart the read-only artifact viewer UI with:\n'
printf './conda-env/bin/python tools/analysis_ui_api_server.py --host 127.0.0.1 --port 8777 --verbose\n'
printf '\nRecommended walkthrough:\n'
printf 'Evidence Trace -> FFN -> Real Artifact -> ONNX Graph -> MLIR -> Models -> BERT layer 0 FFN -> Artifact Bundle -> Reports\n'
