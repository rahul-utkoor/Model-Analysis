#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" tools/interactive_analysis_explorer.py \
  --model bert-base-uncased \
  --layer 0 \
  --no-open \
  --scripted "nodes;subgraph Feed Forward;plan;validation;path;back;back"
