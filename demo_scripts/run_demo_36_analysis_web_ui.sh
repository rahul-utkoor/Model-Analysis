#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./conda-env/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8777}"

echo "Building React frontend..."
(
  cd ui/pruning-analysis-explorer
  npm install
  npm run build
)

echo
echo "Starting pruning analysis web UI."
echo "Open: http://${HOST}:${PORT}/"
echo

"$PYTHON" tools/analysis_ui_api_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --verbose
