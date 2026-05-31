from experimental.dfa_pruning_propagation.ir import Axis, Node
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind
from experimental.dfa_pruning_propagation.semantics import SemanticRole
from experimental.dfa_pruning_propagation.transfer import transfer


def test_activation_propagates_deadness_backward() -> None:
    input_axis = Axis("gelu.input", "channel_j", "intermediate_dim")
    output_axis = Axis("gelu.output", "channel_j", "intermediate_dim")
    node = Node("gelu", "GELU", "Gelu", inputs=[input_axis], outputs=[output_axis], semantic_role=SemanticRole.INDEX_PRESERVING_ACTIVATION)

    emissions = transfer(node, {output_axis: Fact(output_axis, FactKind.DEAD, "seed")}, output_axis)

    assert emissions[0].fact == Fact(
        input_axis,
        FactKind.DEAD,
        "index-preserving activation",
        "gelu",
        ("semantic_role=INDEX_PRESERVING_ACTIVATION",),
    )


def test_attention_context_blocks_unproven_mapping() -> None:
    value_axis = Axis("context.value", "channel_j", "value_dim")
    output_axis = Axis("context.output", "channel_j", "value_context_dim")
    node = Node(
        "context",
        "Context",
        "MatMul",
        inputs=[value_axis],
        outputs=[output_axis],
        semantic_role=SemanticRole.ATTENTION_CONTEXT,
        attrs={"value_axis_mapping": "unproven"},
    )

    emissions = transfer(node, {output_axis: Fact(output_axis, FactKind.DEAD, "seed")}, output_axis)

    assert emissions[0].fact.kind == FactKind.BLOCKED
    assert emissions[0].fact.reason == "value_axis_mapping_unproven"


def test_transfer_uses_semantics_not_misleading_name() -> None:
    input_axis = Axis("q_proj_like.input", "channel_j", "intermediate_dim")
    output_axis = Axis("q_proj_like.output", "channel_j", "intermediate_dim")
    node = Node(
        "q_proj",
        "Misleading q_proj-like label",
        "Tanh",
        inputs=[input_axis],
        outputs=[output_axis],
        semantic_role=SemanticRole.INDEX_PRESERVING_ACTIVATION,
    )

    emissions = transfer(node, {output_axis: Fact(output_axis, FactKind.DEAD, "seed")}, output_axis)

    assert emissions[0].fact.axis == input_axis
    assert emissions[0].fact.kind == FactKind.DEAD
