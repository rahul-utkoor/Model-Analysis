from experimental.dfa_pruning_propagation.examples import ffn_example, ffn_renamed_example, residual_example
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.dfa_pruning_propagation.report import render_markdown
from experimental.dfa_pruning_propagation.worklist import analyze


def by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.state.items() if axis.tensor == tensor)


def test_ffn_deadness_reaches_fc1_output_and_preserves_fc2_hidden() -> None:
    example = ffn_example()
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "fc1.output").kind == FactKind.DEAD
    assert by_tensor(result, "gelu.input").kind == FactKind.DEAD
    assert by_tensor(result, "gelu.output").kind == FactKind.DEAD
    assert by_tensor(result, "fc2.output").kind == FactKind.PROTECTED
    assert result.summary["reached_fixed_point"]


def test_residual_hidden_pruning_is_blocked_and_layernorm_protected() -> None:
    example = residual_example()
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "linear.output").kind == FactKind.BLOCKED
    assert by_tensor(result, "layernorm.output").kind == FactKind.PROTECTED


def test_ffn_renamed_propagates_by_semantics() -> None:
    example = ffn_renamed_example()
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "alpha.output").kind == FactKind.DEAD
    assert by_tensor(result, "beta.input").kind == FactKind.DEAD
    assert by_tensor(result, "beta.output").kind == FactKind.DEAD
    assert by_tensor(result, "gamma.output").kind == FactKind.PROTECTED


def test_markdown_explains_ffn_backward_propagation() -> None:
    example = ffn_example()
    markdown = render_markdown(example, analyze(example.graph, example.seed_facts))

    assert "consumer input deadness at fc2 propagates backward" in markdown
    assert "| fc1 | Linear | EXPANSION_PROJECTION |" in markdown
    assert "`FFN_INTERMEDIATE_CHAIN`" in markdown
    assert "Static pruning propagation proves how dead axes flow" in markdown
