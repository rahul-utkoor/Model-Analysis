from pathlib import Path

from experimental.mlir_evidence_coverage.coverage_case import CoverageCase, CoveragePatternKind, CoverageVerdict
from experimental.mlir_evidence_coverage.runner import CoverageRunOptions, run_coverage_case


def test_missing_case_does_not_remove_unrelated_output(tmp_path: Path) -> None:
    unrelated = tmp_path / "artifacts/other-case/marker.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep", encoding="utf-8")
    case = CoverageCase("missing", "model", 0, CoveragePatternKind.UNKNOWN, "node", str(tmp_path / "missing.onnx"), "UNKNOWN", "unknown", False)
    result = run_coverage_case(case, CoverageRunOptions(output_root=str(tmp_path / "artifacts")))
    assert result.verdict == CoverageVerdict.MISSING
    assert unrelated.read_text(encoding="utf-8") == "keep"
