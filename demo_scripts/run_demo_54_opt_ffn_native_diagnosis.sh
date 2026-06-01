#!/usr/bin/env bash
set -euo pipefail

./conda-env/bin/python -m experimental.opt_ffn_native_diagnosis.cli \
  --model facebook__opt-125m \
  --layers all \
  --output-dir reports/opt_ffn_native_diagnosis \
  --run-native-pass \
  --native-pass-tool experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  --verbose

./conda-env/bin/python -m experimental.mlir_evidence_coverage.cli \
  --models facebook__opt-125m \
  --layers all \
  --patterns FFN_MLP_INTERMEDIATE \
  --output-dir reports/mlir_evidence_coverage_opt_ffn_native_diagnosis \
  --format both \
  --verbose
