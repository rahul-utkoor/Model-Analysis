from experimental.dfa_pruning_propagation.examples import (
    attention_qk_renamed_example,
    attention_value_renamed_example,
    ffn_renamed_example,
)
from experimental.dfa_pruning_propagation.semantics import SemanticPattern, SemanticRole, annotate_graph


def test_semantic_ffn_pattern_detects_renamed_nodes() -> None:
    annotations = annotate_graph(ffn_renamed_example().graph)

    assert annotations.nodes["alpha"].semantic_role == SemanticRole.EXPANSION_PROJECTION
    assert annotations.nodes["beta"].semantic_role == SemanticRole.INDEX_PRESERVING_ACTIVATION
    assert annotations.nodes["gamma"].semantic_role == SemanticRole.CONTRACTION_PROJECTION
    assert any(pattern.pattern == SemanticPattern.FFN_INTERMEDIATE_CHAIN for pattern in annotations.patterns)


def test_semantic_attention_value_pattern_detects_renamed_nodes() -> None:
    annotations = annotate_graph(attention_value_renamed_example().graph)

    assert annotations.nodes["producer_X"].semantic_role == SemanticRole.VALUE_PROJECTION
    assert annotations.nodes["bridge_Y"].semantic_role == SemanticRole.ATTENTION_CONTEXT
    assert annotations.nodes["consumer_Z"].semantic_role == SemanticRole.ATTENTION_OUTPUT_PROJECTION
    assert any(pattern.pattern == SemanticPattern.ATTENTION_VALUE_CHAIN for pattern in annotations.patterns)


def test_semantic_qk_pattern_detects_renamed_score_contraction() -> None:
    annotations = annotate_graph(attention_qk_renamed_example().graph)

    assert annotations.nodes["left_branch"].semantic_role == SemanticRole.QUERY_PROJECTION
    assert annotations.nodes["right_branch"].semantic_role == SemanticRole.KEY_PROJECTION
    assert annotations.nodes["mixing_stage"].semantic_role == SemanticRole.SCORE_CONTRACTION
    assert any(pattern.pattern == SemanticPattern.ATTENTION_QK_SCORE_CHAIN for pattern in annotations.patterns)
