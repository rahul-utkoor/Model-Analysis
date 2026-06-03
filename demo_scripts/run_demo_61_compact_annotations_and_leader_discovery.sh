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
LEADERS="reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.leaders.md"
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
  --leader-report "$LEADERS" \
  --mlir-output-dir "$MLIR_DIR" \
  --annotation-mode doc_string \
  --doc-string-format compact \
  --run-native-pass \
  --fallback-doc-string \
  --check-onnx \
  --verbose

"$PYTHON" - <<'PY'
import json
from pathlib import Path

import onnx

sidecar = Path("reports/onnx_axis_semantics/bert-base-uncased_layer0_attention_value_path.json")
onnx_path = Path("artifacts/annotated_onnx/bert-base-uncased/layer_0/attention_value_path.axis_annotated.onnx")
payload = json.loads(sidecar.read_text())
model = onnx.load(onnx_path)
print("annotated_onnx:", onnx_path)
print("sidecar_json:", sidecar)
print("leader_report:", payload.get("leader_report"))
print("semantic_counts:", payload.get("semantic_counts"))
print("evidence_tier_counts:", payload.get("evidence_tier_counts"))
print("leader_candidate_counts:", payload.get("leader_candidate_counts"))
print("compact_doc_string_example:")
print(model.graph.node[0].doc_string)
PY
