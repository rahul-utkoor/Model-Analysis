# Native MLIR Pruning Axis Dependence Tool

This directory contains an out-of-tree standalone MLIR-linked tool named `pruning-axis-dependence`.

The Python bridge remains the orchestrator. The native tool is optional: it inspects selected ONNX-MLIR artifacts, collects affine/scf loops and affine/memref loads and stores, and emits the JSON contract documented in `../README.md`.

## Build

```bash
bash experimental/mlir_axis_bridge/native/build_native_pass.sh
```

## Run

```bash
experimental/mlir_axis_bridge/native/build/pruning-axis-dependence \
  experimental/mlir_axis_bridge/native/samples/attention_context_affine.mlir \
  --output reports/mlir_axis_bridge/native_attention_context_sample.json
```

## Current Scope

The executable walks real MLIR operations and emits preserved, reduced, and conservative mixed relations. It is intentionally local and minimal. It does not perform transformations, invoke full dependence solvers, or replace the Python bridge.
