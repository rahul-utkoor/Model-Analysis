#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./conda-env/bin/python}"
INPUT="artifacts/attention_value_path_subgraphs/bert-base-uncased/layers/layer_0/bert_layer_0_attention_value_path/subgraph.onnx"
OUTPUT="artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.onnx"
SIDECAR="reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.json"
DOT="artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.dot"
SVG="artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.svg"
MLIR_DIR="reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path_mlir"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing demo input: $INPUT"
  exit 0
fi

"$PYTHON" scripts/annotate_onnx_axis_semantics.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --sidecar-json "$SIDECAR" \
  --dot "$DOT" \
  --svg "$SVG" \
  --mlir-output-dir "$MLIR_DIR" \
  --annotation-mode doc_string \
  --run-native-pass \
  --fallback-doc-string \
  --check-onnx \
  --verbose

"$PYTHON" - <<'PY'
import json
from pathlib import Path

sidecar = Path("reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.json")
if sidecar.is_file():
    payload = json.loads(sidecar.read_text())
    print("semantic_counts:", payload.get("semantic_counts"))
    print("evidence_tier_counts:", payload.get("evidence_tier_counts"))
    print("blocker_counts:", payload.get("blocker_counts"))
PY
