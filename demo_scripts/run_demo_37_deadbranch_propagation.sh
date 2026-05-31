#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./conda-env/bin/python}"
MODEL="${MODEL:-facebook/opt-125m}"

"$PYTHON" scripts/analyze_deadbranch_propagation.py --model "$MODEL" --verbose
"$PYTHON" scripts/explain_deadbranch_propagation.py --model "$MODEL" --contains v_proj --limit 5
"$PYTHON" scripts/explain_deadbranch_propagation.py --model "$MODEL" --blocked-only --limit 10
