from __future__ import annotations

from experimental.mlir_axis_bridge.toolchain import check_toolchain


def test_toolchain_missing_is_graceful() -> None:
    status = check_toolchain("/definitely/missing/onnx-mlir", "/definitely/missing/mlir-opt")

    assert not status.onnx_mlir_available
    assert not status.mlir_opt_available
    assert status.warnings
