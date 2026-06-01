from experimental.opt_ffn_native_diagnosis.diagnosis_model import BlockerKind, OptFfnNativeDiagnosis, OptFfnNativeDiagnosisReport
from experimental.opt_ffn_native_diagnosis.report import render_index_markdown


def test_report_mentions_native_vs_fallback_and_blocker() -> None:
    report = OptFfnNativeDiagnosisReport.create(
        [
            OptFfnNativeDiagnosis(
                "facebook__opt-125m",
                0,
                "layer0.onnx",
                ffn_pattern_detected_by_native=True,
                ffn_pattern_detected_by_fallback=True,
                blocker_kind=BlockerKind.ONNX_MLIR_LOWERING_FAILED,
            )
        ]
    )
    text = render_index_markdown(report)

    assert "Native vs Fallback Evidence" in text
    assert "onnx_mlir_lowering_failed" in text
    assert "fc1 -> activation -> fc2" in text
