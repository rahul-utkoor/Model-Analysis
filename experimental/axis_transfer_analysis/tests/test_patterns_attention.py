from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.examples import attention_value_path_example, layernorm_example, qk_score_example, residual_example
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, recognize_patterns


def _patterns(example):
    return recognize_patterns(example.region, analyze_region(example.region))


def test_attention_value_path_is_recognized() -> None:
    matches = _patterns(attention_value_path_example())

    match = next(pattern for pattern in matches if pattern.pattern_kind == PatternKind.ATTENTION_VALUE_PATH)
    assert match.ops == ("value_projection", "context", "output_projection")
    assert match.status == "propagation_amenable"


def test_qk_score_blocker_is_recognized() -> None:
    matches = _patterns(qk_score_example())

    match = next(pattern for pattern in matches if pattern.pattern_kind == PatternKind.QK_SCORE_BLOCKER)
    assert match.status == "blocked"
    assert "qk_score_contraction_mixes_channels" in match.evidence


def test_residual_protection_is_recognized() -> None:
    assert any(pattern.pattern_kind == PatternKind.RESIDUAL_HIDDEN_PROTECTED for pattern in _patterns(residual_example()))


def test_layernorm_protection_is_recognized() -> None:
    assert any(pattern.pattern_kind == PatternKind.LAYERNORM_HIDDEN_PROTECTED for pattern in _patterns(layernorm_example()))
