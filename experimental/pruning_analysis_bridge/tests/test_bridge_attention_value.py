from experimental.axis_transfer_analysis.pattern_recognition import PatternKind
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.examples import attention_value_from_access_example


def _by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.dfa_result.state.items() if axis.tensor == tensor)


def test_bridge_attention_value_from_access_runs_dfa() -> None:
    example = attention_value_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert PatternKind.ATTENTION_VALUE_PATH in {pattern.pattern_kind for pattern in result.pattern_matches}
    assert _by_tensor(result, "value_producer_from_axis_summary.output").kind == FactKind.DEAD
    assert _by_tensor(result, "output_projection_from_axis_summary.output").kind == FactKind.PROTECTED
