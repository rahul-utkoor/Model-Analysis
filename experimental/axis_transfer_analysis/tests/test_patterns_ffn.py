from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.examples import activation_example, ffn_example
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, recognize_patterns


def test_index_preserving_unary_pattern_is_recognized() -> None:
    example = activation_example()
    patterns = recognize_patterns(example.region, analyze_region(example.region))

    assert any(pattern.pattern_kind == PatternKind.INDEX_PRESERVING_UNARY for pattern in patterns)


def test_ffn_intermediate_chain_is_recognized_from_access_relations() -> None:
    example = ffn_example()
    patterns = recognize_patterns(example.region, analyze_region(example.region))

    match = next(pattern for pattern in patterns if pattern.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN)
    assert match.ops == ("expand", "activation", "contract")
    assert match.status == "pruning_amenable"
