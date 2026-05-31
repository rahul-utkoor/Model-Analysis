from __future__ import annotations

from experimental.mlir_axis_bridge.mlir_artifacts import detect_dialect_hints
from experimental.mlir_axis_bridge.mlir_parser import parse_mlir_text


def test_parse_affine_load_store() -> None:
    parsed = parse_mlir_text(
        """
        affine.for %j = 0 to 8 {
          %0 = affine.load %X[%b, %s, %j] : memref<1x2x8xf32>
          affine.store %0, %Y[%b, %s, %j] : memref<1x2x8xf32>
        }
        """
    )

    assert [(record.tensor, record.indices, record.access_kind) for record in parsed.accesses] == [
        ("X", ("b", "s", "j"), "read"),
        ("Y", ("b", "s", "j"), "write"),
    ]


def test_parse_scf_memref_load_store() -> None:
    parsed = parse_mlir_text(
        """
        scf.for %j = %c0 to %c8 step %c1 {
          %0 = memref.load %X[%b, %j] : memref<1x8xf32>
          memref.store %0, %Y[%b, %j] : memref<1x8xf32>
        }
        """
    )

    assert [record.access_kind for record in parsed.accesses] == ["read", "write"]
    assert {operation.op_name for operation in parsed.operations} >= {"scf.for", "memref.load", "memref.store"}


def test_detect_dialect_hints() -> None:
    text = 'affine.for %i = 0 to 1 { scf.for %j = %c0 to %c1 step %c1 { %0 = "onnx.MatMul"() : () -> () } }\nlinalg.matmul'

    assert set(detect_dialect_hints(text)) >= {"affine.for", "scf.for", "onnx.", "linalg."}
