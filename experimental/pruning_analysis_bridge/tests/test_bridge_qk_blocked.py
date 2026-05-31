from experimental.axis_transfer_analysis.pattern_recognition import PatternKind
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.examples import layernorm_from_access_example, qk_blocked_from_access_example, residual_from_access_example


def _by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.dfa_result.state.items() if axis.tensor == tensor)


def test_bridge_qk_from_access_blocks() -> None:
    example = qk_blocked_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert PatternKind.QK_SCORE_BLOCKER in {pattern.pattern_kind for pattern in result.pattern_matches}
    assert _by_tensor(result, "query_from_axis_summary.output").kind == FactKind.BLOCKED
    assert any("qk_score_contraction_mixes_channels" in event.output_fact for event in result.dfa_result.blocked_events)


def test_bridge_residual_from_access_blocks_hidden_pruning() -> None:
    example = residual_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert _by_tensor(result, "protected_from_axis_summary.input").kind == FactKind.BLOCKED
    assert _by_tensor(result, "protected_from_axis_summary.output").kind == FactKind.PROTECTED


def test_bridge_layernorm_from_access_blocks_hidden_pruning() -> None:
    example = layernorm_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert _by_tensor(result, "protected_from_axis_summary.input").kind == FactKind.BLOCKED
    assert _by_tensor(result, "protected_from_axis_summary.output").kind == FactKind.PROTECTED
