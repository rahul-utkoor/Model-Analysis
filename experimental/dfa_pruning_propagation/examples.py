"""Small teaching graphs for the DFA pruning propagation prototype."""

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


def ffn_example() -> Example:
    graph = Graph()
    fc1_out = _axis("fc1.output", "intermediate_dim")
    gelu_in = _axis("gelu.input", "intermediate_dim")
    gelu_out = _axis("gelu.output", "intermediate_dim")
    fc2_in = _axis("fc2.input", "intermediate_dim")
    fc2_out = _axis("fc2.output", "hidden_dim")
    graph.add_node(Node("fc1", "FC1 expansion projection", "linear_expand", outputs=[fc1_out]))
    graph.add_node(Node("gelu", "Index-preserving GELU", "activation", inputs=[gelu_in], outputs=[gelu_out]))
    graph.add_node(Node("fc2", "FC2 contraction projection", "linear_contract", inputs=[fc2_in], outputs=[fc2_out]))
    graph.add_edge(Edge("fc1", fc1_out, "gelu", gelu_in))
    graph.add_edge(Edge("gelu", gelu_out, "fc2", fc2_in))
    return Example(
        name="ffn",
        description="Backward structural deadness propagation through an FFN expansion, activation, and contraction.",
        graph=graph,
        seed_facts=[Fact(fc2_in, FactKind.DEAD, "seed: fc2 input channel j is exactly dead", "seed")],
        interpretation="The consumer input deadness at fc2 propagates backward through the index-preserving activation and marks fc1 output dead.",
    )


def attention_value_example(*, mapping: str = "proven") -> Example:
    graph = Graph()
    v_out = _axis("v_proj.output", "value_dim")
    context_in = _axis("attention_context.value_input", "value_dim")
    context_out = _axis("attention_context.output", "value_context_dim")
    out_in = _axis("out_proj.input", "value_context_dim")
    out_hidden = _axis("out_proj.output", "hidden_dim")
    graph.add_node(Node("v_proj", "Attention value projection", "value_projection", outputs=[v_out]))
    graph.add_node(Node("attention_context", "Attention context value path", "attention_context", inputs=[context_in], outputs=[context_out], attrs={"value_axis_mapping": mapping}))
    graph.add_node(Node("out_proj", "Attention output projection", "attention_output_projection", inputs=[out_in], outputs=[out_hidden]))
    graph.add_edge(Edge("v_proj", v_out, "attention_context", context_in))
    graph.add_edge(Edge("attention_context", context_out, "out_proj", out_in))
    return Example(
        name="attention-value",
        description="Backward deadness propagation from attention out_proj input to the value projection output.",
        graph=graph,
        seed_facts=[Fact(out_in, FactKind.DEAD, "seed: out_proj input channel j is exactly dead", "seed")],
        interpretation="The out_proj input deadness propagates backward through the proven value-axis mapping and marks v_proj output dead.",
    )


def attention_qk_example() -> Example:
    graph = Graph()
    q_out = _axis("q_proj.output", "head_dim")
    k_out = _axis("k_proj.output", "head_dim")
    score_q = _axis("score_matmul.query_input", "head_dim")
    score_k = _axis("score_matmul.key_input", "head_dim")
    scores = _axis("score_matmul.output", "sequence_dim")
    graph.add_node(Node("q_proj", "Attention query projection", "q_projection", outputs=[q_out]))
    graph.add_node(Node("k_proj", "Attention key projection", "k_projection", outputs=[k_out]))
    graph.add_node(Node("score_matmul", "QK^T score contraction", "score_matmul", inputs=[score_q, score_k], outputs=[scores]))
    graph.add_edge(Edge("q_proj", q_out, "score_matmul", score_q))
    graph.add_edge(Edge("k_proj", k_out, "score_matmul", score_k))
    return Example(
        name="attention-qk",
        description="Blocked simple deadness propagation through the QK^T score contraction.",
        graph=graph,
        seed_facts=[Fact(q_out, FactKind.PRUNED, "seed: attempt to prune query output channel j", "seed")],
        interpretation="Propagation is blocked because QK^T mixes Q/K dimensions.",
    )


def residual_example() -> Example:
    graph = Graph()
    linear_out = _axis("linear.output", "hidden_dim")
    residual_in = _axis("residual_add.branch", "hidden_dim")
    residual_out = _axis("residual_add.output", "hidden_dim")
    norm_in = _axis("layernorm.input", "hidden_dim")
    norm_out = _axis("layernorm.output", "hidden_dim")
    graph.add_node(Node("linear", "Hidden projection", "linear_contract", outputs=[linear_out]))
    graph.add_node(Node("residual_add", "Residual merge", "residual_add", inputs=[residual_in], outputs=[residual_out]))
    graph.add_node(Node("layernorm", "LayerNorm", "layernorm", inputs=[norm_in], outputs=[norm_out]))
    graph.add_edge(Edge("linear", linear_out, "residual_add", residual_in))
    graph.add_edge(Edge("residual_add", residual_out, "layernorm", norm_in))
    return Example(
        name="residual",
        description="Protected residual and LayerNorm hidden dimensions.",
        graph=graph,
        seed_facts=[Fact(linear_out, FactKind.PRUNED, "seed: attempt to prune hidden_dim channel j", "seed")],
        interpretation="Residual and LayerNorm hidden dimensions are protected, so local hidden-width pruning is blocked.",
    )


def get_example(name: str) -> Example:
    examples = {
        "ffn": ffn_example,
        "attention-value": attention_value_example,
        "attention-qk": attention_qk_example,
        "residual": residual_example,
    }
    try:
        return examples[name]()
    except KeyError as exc:
        raise ValueError(f"unknown example: {name}") from exc
