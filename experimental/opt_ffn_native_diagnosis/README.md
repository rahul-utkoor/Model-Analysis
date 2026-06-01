# OPT FFN Native MLIR Evidence Diagnosis

This read-only experimental layer diagnoses why OPT FFN plans previously used high-level MLIR fallback evidence. The broad OPT MLP artifact includes LayerNorm and residual boundary operations. ONNX-MLIR aborts before affine lowering because the exported LayerNorm parameters are `f16` while the activation input is `f32`.

The repair exports a smaller local evidence unit:

```text
fc1 -> activation -> fc2
```

That core preserves the pruning-relevant intermediate-axis chain while removing unrelated boundary noise. MLIR remains a local evidence generator; DFA remains the propagation analysis.

Run:

```bash
./conda-env/bin/python -m experimental.opt_ffn_native_diagnosis.cli \
  --model facebook__opt-125m \
  --layers all \
  --output-dir reports/opt_ffn_native_diagnosis \
  --run-native-pass \
  --native-pass-tool experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  --verbose
```

This module does not execute pruning or mutate model weights.
