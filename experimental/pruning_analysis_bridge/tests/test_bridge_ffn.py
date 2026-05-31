from __future__ import annotations

from copy import deepcopy

from experimental.axis_transfer_analysis.examples import ffn_example
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.examples import ffn_from_access_example


def _by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.dfa_result.state.items() if axis.tensor == tensor)


def test_bridge_ffn_from_access_recognizes_pattern() -> None:
    example = ffn_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert PatternKind.FFN_INTERMEDIATE_CHAIN in {pattern.pattern_kind for pattern in result.pattern_matches}


def test_bridge_ffn_from_access_runs_dfa() -> None:
    example = ffn_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy)

    assert _by_tensor(result, "producer_from_axis_summary.output").kind == FactKind.DEAD
    assert _by_tensor(result, "consumer_from_axis_summary.output").kind == FactKind.PROTECTED
    assert result.summary["reached_fixed_point"]


def test_bridge_semantics_are_not_name_based() -> None:
    source = deepcopy(ffn_example().region)
    for op, op_id, label in zip(source.ops, ["weird_A", "strange_B", "odd_C"], ["first unknown label", "middle unknown label", "last unknown label"]):
        op.op_id = op_id
        op.label = label
    policy = ffn_from_access_example().seed_policy

    result = run_bridge_analysis(source, policy)

    match = next(pattern for pattern in result.pattern_matches if pattern.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN)
    assert match.ops == ("weird_A", "strange_B", "odd_C")
    assert _by_tensor(result, "producer_from_axis_summary.output").kind == FactKind.DEAD
