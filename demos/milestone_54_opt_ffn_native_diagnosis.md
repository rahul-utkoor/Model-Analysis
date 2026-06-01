# Milestone 54: OPT FFN Native MLIR Evidence Diagnosis

Diagnose and repair the OPT FFN local evidence unit:

```bash
./conda-env/bin/python -m experimental.opt_ffn_native_diagnosis.cli \
  --model facebook__opt-125m \
  --layers all \
  --output-dir reports/opt_ffn_native_diagnosis \
  --run-native-pass \
  --native-pass-tool experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  --verbose
```

Then verify OPT FFN-only coverage:

```bash
./conda-env/bin/python -m experimental.mlir_evidence_coverage.cli \
  --models facebook__opt-125m \
  --layers all \
  --patterns FFN_MLP_INTERMEDIATE \
  --output-dir reports/mlir_evidence_coverage_opt_ffn_native_diagnosis \
  --format both \
  --verbose
```

Inspect `reports/opt_ffn_native_diagnosis/index.md`.

This demo performs static evidence diagnosis only. It does not execute pruning or mutate model weights.
