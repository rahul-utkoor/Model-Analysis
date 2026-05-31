from experimental.mlir_evidence_coverage.config import model_specs, pattern_specs
from experimental.mlir_evidence_coverage.coverage_case import CoveragePatternKind


def test_default_config_has_five_models() -> None:
    models = model_specs()
    assert len(models) == 5
    assert {model.short_name for model in models} == {"bert", "distilbert", "opt", "gpt2", "vit"}


def test_pattern_filter() -> None:
    patterns = pattern_specs("FFN_MLP_INTERMEDIATE,ATTENTION_QK_SCORE")
    assert [pattern.kind for pattern in patterns] == [
        CoveragePatternKind.FFN_MLP_INTERMEDIATE,
        CoveragePatternKind.ATTENTION_QK_SCORE,
    ]
