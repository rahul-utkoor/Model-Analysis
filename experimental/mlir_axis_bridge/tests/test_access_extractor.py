from __future__ import annotations

from experimental.mlir_axis_bridge.access_extractor import extract_mlir_access_summary
from experimental.mlir_axis_bridge.mlir_artifacts import artifact_from_path


def test_extract_access_summary_records_loops_and_accesses(tmp_path) -> None:
    source = tmp_path / "sample.mlir"
    source.write_text(
        """
        affine.for %j = 0 to 8 {
          %0 = affine.load %X[%b, %j] : memref<1x8xf32>
          affine.store %0, %Y[%b, %j] : memref<1x8xf32>
        }
        """,
        encoding="utf-8",
    )

    summary = extract_mlir_access_summary(artifact_from_path(source, "synthetic"))

    assert summary.loop_kinds == ("affine.for",)
    assert len(summary.access_records) == 2
    assert not summary.warnings
