from experimental.opt_ffn_native_diagnosis.diagnosis_model import (
    BlockerKind,
    OptFfnNativeDiagnosis,
    OptFfnNativeDiagnosisReport,
)


def test_diagnosis_model_counts() -> None:
    report = OptFfnNativeDiagnosisReport.create(
        [
            OptFfnNativeDiagnosis("facebook__opt-125m", 0, "layer0.onnx", ffn_pattern_detected_by_native=True, blocker_kind=BlockerKind.ONNX_MLIR_LOWERING_FAILED),
            OptFfnNativeDiagnosis("facebook__opt-125m", 1, "layer1.onnx", ffn_pattern_detected_by_fallback=True, blocker_kind=BlockerKind.NO_AFFINE_LOOPS),
            OptFfnNativeDiagnosis("facebook__opt-125m", 2, "layer2.onnx", blocker_kind=BlockerKind.NO_ONNX_ARTIFACT),
        ]
    )

    assert report.total_layers == 3
    assert report.native_proven == 1
    assert report.fallback_only == 1
    assert report.failed == 1
    assert report.blockers_by_kind["onnx_mlir_lowering_failed"] == 1
