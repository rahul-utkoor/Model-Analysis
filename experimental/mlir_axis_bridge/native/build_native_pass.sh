#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_build="${LLVM_BUILD:-$HOME/Dev/onnx-mlir-work/llvm-project/build}"
mlir_dir="${MLIR_DIR:-$llvm_build/lib/cmake/mlir}"
build_dir="$script_dir/build"

cmake -S "$script_dir" -B "$build_dir" -DMLIR_DIR="$mlir_dir"
cmake --build "$build_dir" --target pruning-axis-dependence
