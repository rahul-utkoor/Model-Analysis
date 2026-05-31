"""Small teaching graphs for semantic DFA pruning propagation."""

from __future__ import annotations

from dataclasses import dataclass

from experimental.dfa_pruning_propagation.ir import Axis, Edge, Graph, Node
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind


@dataclass
class Example:
    name: str
    description: str
    graph: Graph
    seed_facts: list[Fact]
    interpretation: str


def _axis(tensor: str, role: str) -> Axis:
    return Axis(tensor=tensor, dim="channel_j", role=role)


def _ffn_example(labels: tuple[str, str, str], *, name: str) -> Example:
    expansion, activation, contraction = labels
    graph = Graph()
    expansion_out = _axis(f"{expansion}.output", "intermediate_dim")
    activation_in = _axis(f"{activation}.input", "intermediate_dim")
    activation_out = _axis(f"{activation}.output", "intermediate_dim")
    contraction_in = _axis(f"{contraction}.input", "intermediate_dim")
    contraction_out = _axis(f"{contraction}.output", "hidden_dim")
    graph.add_node(Node(expansion, expansion, "Linear", outputs=[expansion_out]))
    graph.add_node(Node(activation, activation, "Gelu", inputs=[activation_in], outputs=[activation_out]))
    graph.add_node(Node(contraction, contraction, "Linear", inputs=[contraction_in], outputs=[contraction_out]))
    graph.add_edge(Edge(expansion, expansion_out, activation, activation_in))
    graph.add_edge(Edge(activation, activation_out, contraction, contraction_in))
    return Example(
        name=name,
        description="Backward structural deadness propagation through a semantic FFN expansion, activation, and contraction chain.",
        graph=graph,
        seed_facts=[Fact(contraction_in, FactKind.DEAD, f"seed: {contraction} input channel j is exactly dead", "seed")],
        interpretation=(
            f"The consumer input deadness at {contraction} propagates backward through the index-preserving activation "
            f"and marks {expansion} output dead. The analysis did not rely on names {expansion}/{activation}/{contraction}; "
            "propagation followed inferred semantic roles and axis roles."
        ),
    )


def ffn_example() -> Example:
    return _ffn_example(("fc1", "gelu", "fc2"), name="ffn")


def ffn_renamed_example() -> Example:
    return _ffn_example(("alpha", "beta", "gamma"), name="ffn-renamed")


def _attention_value_example(labels: tuple[str, str, str], *, name: str, mapping: str) -> Example:
    value_projection, context, output_projection = labels
    graph = Graph()
    value_out = _axis(f"{value_projection}.output", "value_dim")
    context_in = _axis(f"{context}.value_input", "value_dim")
    context_out = _axis(f"{context}.output", "value_context_dim")
    output_in = _axis(f"{output_projection}.input", "value_context_dim")
    output_hidden = _axis(f"{output_projection}.output", "hidden_dim")
    graph.add_node(Node(value_projection, value_projection, "Linear", outputs=[value_out]))
    graph.add_node(Node(context, context, "MatMul", inputs=[context_in], outputs=[context_out], attrs={"value_axis_mapping": mapping}))
    graph.add_node(Node(output_projection, output_projection, "Linear", inputs=[output_in], outputs=[output_hidden]))
    graph.add_edge(Edge(value_projection, value_out, context, context_in))
    graph.add_edge(Edge(context, context_out, output_projection, output_in))
    return Example(
        name=name,
        description="Backward deadness propagation through a semantic attention value path.",
        graph=graph,
        seed_facts=[Fact(output_in, FactKind.DEAD, f"seed: {output_projection} input channel j is exactly dead", "seed")],
        interpretation=(
            f"The {output_projection} input deadness propagates backward through the proven value-axis mapping and marks "
            f"{value_projection} output dead. The labels {value_projection}/{context}/{output_projection} are display-only; "
            "the transfer follows inferred attention value-path roles."
        ),
    )


def attention_value_example(*, mapping: str = "proven") -> Example:
    return _attention_value_example(("v_proj", "attention_context", "out_proj"), name="attention-value", mapping=mapping)


def attention_value_renamed_example(*, mapping: str = "proven") -> Example:
    return _attention_value_example(("producer_X", "bridge_Y", "consumer_Z"), name="attention-value-renamed", mapping=mapping)


def _attention_qk_example(labels: tuple[str, str, str], *, name: str) -> Example:
    query, key, score = labels
    graph = Graph()
    query_out = _axis(f"{query}.output", "head_dim")
    key_out = _axis(f"{key}.output", "head_dim")
    score_query = _axis(f"{score}.query_input", "head_dim")
    score_key = _axis(f"{score}.key_input", "head_dim")
    scores = _axis(f"{score}.output", "score_dim")
    graph.add_node(Node(query, query, "Linear", outputs=[query_out], attrs={"attention_projection_role": "query"}))
    graph.add_node(Node(key, key, "Linear", outputs=[key_out], attrs={"attention_projection_role": "key"}))
    graph.add_node(Node(score, score, "MatMul", inputs=[score_query, score_key], outputs=[scores]))
    graph.add_edge(Edge(query, query_out, score, score_query))
    graph.add_edge(Edge(key, key_out, score, score_key))
    return Example(
        name=name,
        description="Blocked simple deadness propagation through a semantic QK^T score contraction.",
        graph=graph,
        seed_facts=[Fact(query_out, FactKind.PRUNED, f"seed: attempt to prune {query} output channel j", "seed")],
        interpretation=(
            "Propagation is blocked because QK^T mixes Q/K dimensions. The score contraction is identified from HEAD "
            "inputs and SCORE output, not from node names."
        ),
    )


def attention_qk_example() -> Example:
    return _attention_qk_example(("q_proj", "k_proj", "score_matmul"), name="attention-qk")


def attention_qk_renamed_example() -> Example:
    return _attention_qk_example(("left_branch", "right_branch", "mixing_stage"), name="attention-qk-renamed")


def residual_example() -> Example:
    graph = Graph()
    linear_out = _axis("linear.output", "hidden_dim")
    residual_in = _axis("residual_add.branch", "hidden_dim")
    residual_out = _axis("residual_add.output", "hidden_dim")
    norm_in = _axis("layernorm.input", "hidden_dim")
    norm_out = _axis("layernorm.output", "hidden_dim")
    graph.add_node(Node("linear", "Hidden projection", "Linear", outputs=[linear_out]))
    graph.add_node(Node("residual_add", "Residual merge", "Add", inputs=[residual_in], outputs=[residual_out]))
    graph.add_node(Node("layernorm", "LayerNorm", "LayerNorm", inputs=[norm_in], outputs=[norm_out]))
    graph.add_edge(Edge("linear", linear_out, "residual_add", residual_in))
    graph.add_edge(Edge("residual_add", residual_out, "layernorm", norm_in))
    return Example(
        name="residual",
        description="Protected residual and LayerNorm hidden dimensions inferred from semantic structure.",
        graph=graph,
        seed_facts=[Fact(linear_out, FactKind.PRUNED, "seed: attempt to prune hidden_dim channel j", "seed")],
        interpretation="Residual and LayerNorm hidden dimensions are protected, so local hidden-width pruning is blocked.",
    )


def get_example(name: str) -> Example:
    examples = {
        "ffn": ffn_example,
        "ffn-renamed": ffn_renamed_example,
        "attention-value": attention_value_example,
        "attention-value-renamed": attention_value_renamed_example,
        "attention-qk": attention_qk_example,
        "attention-qk-renamed": attention_qk_renamed_example,
        "residual": residual_example,
    }
    try:
        return examples[name]()
    except KeyError as exc:
        raise ValueError(f"unknown example: {name}") from exc
