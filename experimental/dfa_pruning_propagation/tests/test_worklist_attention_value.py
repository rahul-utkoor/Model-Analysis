from experimental.dfa_pruning_propagation.examples import attention_value_example
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.dfa_pruning_propagation.worklist import analyze


def by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.state.items() if axis.tensor == tensor)


def test_attention_value_deadness_reaches_v_projection_when_mapping_proven() -> None:
    example = attention_value_example(mapping="proven")
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "attention_context.output").kind == FactKind.DEAD
    assert by_tensor(result, "attention_context.value_input").kind == FactKind.DEAD
    assert by_tensor(result, "v_proj.output").kind == FactKind.DEAD
    assert by_tensor(result, "out_proj.output").kind == FactKind.PROTECTED


def test_attention_value_deadness_blocks_when_mapping_unproven() -> None:
    example = attention_value_example(mapping="unproven")
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "attention_context.value_input").kind == FactKind.BLOCKED
    assert by_tensor(result, "v_proj.output").kind == FactKind.BLOCKED
    assert any("value_axis_mapping_unproven" in event.output_fact for event in result.blocked_events)
