#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./conda-env/bin/python}"

"$PYTHON" scripts/validate_pruning_plans.py --model bert-base-uncased --verbose
"$PYTHON" scripts/validate_pruning_plans.py --model facebook/opt-125m --verbose
"$PYTHON" scripts/validate_pruning_plans.py --model distilbert-base-uncased --verbose
"$PYTHON" scripts/validate_pruning_plans.py --model google/vit-base-patch16-224 --verbose
"$PYTHON" scripts/validate_pruning_plans.py --model gpt2 --verbose

"$PYTHON" scripts/build_layer_subgraph_validation_pack.py --model facebook/opt-125m --layer 0 --export-onnx --render-svg --verbose
"$PYTHON" scripts/build_layer_subgraph_validation_pack.py --model gpt2 --layer 0 --export-onnx --render-svg --verbose
"$PYTHON" scripts/build_layer_subgraph_validation_pack.py --model google/vit-base-patch16-224 --layer 0 --export-onnx --render-svg --verbose
"$PYTHON" scripts/build_layer_subgraph_validation_pack.py --model distilbert-base-uncased --layer 0 --export-onnx --render-svg --verbose

"$PYTHON" scripts/build_static_pipeline_for_all_models.py --models all --build-missing-analysis --build-layer-packs --verbose
"$PYTHON" scripts/report_static_pipeline_coverage.py --models all --verbose
