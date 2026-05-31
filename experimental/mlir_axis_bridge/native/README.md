# Native MLIR Pruning Axis Dependence Pass Scaffold

This directory scaffolds a future out-of-tree MLIR pass named `pruning-axis-dependence`.

The Python bridge remains the orchestrator. The native pass is optional: it should inspect selected ONNX-MLIR artifacts, walk `func.func`, collect affine/scf loops and affine/memref loads and stores, and emit the JSON contract documented in `../README.md`.

## Intended Build

```bash
cmake -S experimental/mlir_axis_bridge/native -B /tmp/pruning-axis-native \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
cmake --build /tmp/pruning-axis-native
```

## Current Status

The C++ file is a near-compilable pass skeleton. It walks supported operations and documents where affine-map and dependence facts should be emitted. JSON emission and pass-plugin registration remain TODOs. It is not part of automated tests.
