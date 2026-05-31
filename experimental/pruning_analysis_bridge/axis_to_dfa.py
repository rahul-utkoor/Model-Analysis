"""Lower access-derived pruning patterns into semantic DFA graphs."""

from __future__ import annotations

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import RegionAxisSummary
from experimental.axis_transfer_analysis.loop_ir import RegionSpec
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, PatternMatch, recognize_patterns
from experimental.dfa_pruning_propagation.ir import Axis, Edge, Graph, Node
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind
from experimental.dfa_pruning_propagation.semantics import SemanticRole
from experimental.dfa_pruning_propagation.worklist import analyze
from experimental.pruning_analysis_bridge.bridge_ir import BridgeResult, BridgeSeedPolicy, BridgeTraceEvent


def _axis(tensor: str, role: str) -> Axis:
    return Axis(tensor=tensor, dim="channel_j", role=role)


def _single_supported_pattern(pattern_matches: list[PatternMatch]) -> PatternMatch:
    supported = {
        PatternKind.FFN_INTERMEDIATE_CHAIN,
        PatternKind.ATTENTION_VALUE_PATH,
        PatternKind.QK_SCORE_BLOCKER,
        PatternKind.RESIDUAL_HIDDEN_PROTECTED,
        PatternKind.LAYERNORM_HIDDEN_PROTECTED,
        PatternKind.INDEX_PRESERVING_UNARY,
    }
    for pattern in pattern_matches:
        if pattern.pattern_kind in supported and pattern.pattern_kind != PatternKind.INDEX_PRESERVING_UNARY:
            return pattern
    for pattern in pattern_matches:
        if pattern.pattern_kind in supported:
            return pattern
    raise ValueError("no supported axis-transfer pattern was recognized")


def _build_ffn_graph() -> Graph:
    graph = Graph()
    producer_out = _axis("producer_from_axis_summary.output", "intermediate_dim")
    unary_in = _axis("unary_from_axis_summary.input", "intermediate_dim")
    unary_out = _axis("unary_from_axis_summary.output", "intermediate_dim")
    consumer_in = _axis("consumer_from_axis_summary.input", "intermediate_dim")
    consumer_out = _axis("consumer_from_axis_summary.output", "hidden_dim")
    graph.add_node(Node("axis_producer", "producer_from_axis_summary", "Projection", outputs=[producer_out], semantic_role=SemanticRole.EXPANSION_PROJECTION))
    graph.add_node(Node("axis_unary", "unary_from_axis_summary", "Unary", inputs=[unary_in], outputs=[unary_out], semantic_role=SemanticRole.INDEX_PRESERVING_ACTIVATION))
    graph.add_node(Node("axis_consumer", "consumer_from_axis_summary", "Projection", inputs=[consumer_in], outputs=[consumer_out], semantic_role=SemanticRole.CONTRACTION_PROJECTION))
    graph.add_edge(Edge("axis_producer", producer_out, "axis_unary", unary_in))
    graph.add_edge(Edge("axis_unary", unary_out, "axis_consumer", consumer_in))
    return graph


def _build_unary_graph() -> Graph:
    graph = Graph()
    input_axis = _axis("unary_from_axis_summary.input", "intermediate_dim")
    output_axis = _axis("unary_from_axis_summary.output", "intermediate_dim")
    graph.add_node(Node("axis_unary", "unary_from_axis_summary", "Unary", inputs=[input_axis], outputs=[output_axis], semantic_role=SemanticRole.INDEX_PRESERVING_ACTIVATION))
    return graph


def _build_attention_value_graph() -> Graph:
    graph = Graph()
    producer_out = _axis("value_producer_from_axis_summary.output", "value_dim")
    context_in = _axis("context_from_axis_summary.value_input", "value_dim")
    context_out = _axis("context_from_axis_summary.output", "value_context_dim")
    output_in = _axis("output_projection_from_axis_summary.input", "value_context_dim")
    output_hidden = _axis("output_projection_from_axis_summary.output", "hidden_dim")
    graph.add_node(Node("axis_value_producer", "value_producer_from_axis_summary", "Projection", outputs=[producer_out], semantic_role=SemanticRole.VALUE_PROJECTION))
    graph.add_node(
        Node(
            "axis_context",
            "context_from_axis_summary",
            "Contraction",
            inputs=[context_in],
            outputs=[context_out],
            semantic_role=SemanticRole.ATTENTION_CONTEXT,
            attrs={"value_axis_mapping": "proven", "value_axis_mapping_proven": True},
        )
    )
    graph.add_node(Node("axis_output_projection", "output_projection_from_axis_summary", "Projection", inputs=[output_in], outputs=[output_hidden], semantic_role=SemanticRole.ATTENTION_OUTPUT_PROJECTION))
    graph.add_edge(Edge("axis_value_producer", producer_out, "axis_context", context_in))
    graph.add_edge(Edge("axis_context", context_out, "axis_output_projection", output_in))
    return graph


def _build_qk_graph() -> Graph:
    graph = Graph()
    query_out = _axis("query_from_axis_summary.output", "head_dim")
    key_out = _axis("key_from_axis_summary.output", "head_dim")
    score_query = _axis("score_from_axis_summary.query_input", "head_dim")
    score_key = _axis("score_from_axis_summary.key_input", "head_dim")
    score_out = _axis("score_from_axis_summary.output", "score_dim")
    graph.add_node(Node("axis_query", "query_from_axis_summary", "Projection", outputs=[query_out], semantic_role=SemanticRole.QUERY_PROJECTION))
    graph.add_node(Node("axis_key", "key_from_axis_summary", "Projection", outputs=[key_out], semantic_role=SemanticRole.KEY_PROJECTION))
    graph.add_node(Node("axis_score", "score_from_axis_summary", "Contraction", inputs=[score_query, score_key], outputs=[score_out], semantic_role=SemanticRole.SCORE_CONTRACTION))
    graph.add_edge(Edge("axis_query", query_out, "axis_score", score_query))
    graph.add_edge(Edge("axis_key", key_out, "axis_score", score_key))
    return graph


def _build_protected_graph(role: SemanticRole) -> Graph:
    graph = Graph()
    input_axis = _axis("protected_from_axis_summary.input", "hidden_dim")
    output_axis = _axis("protected_from_axis_summary.output", "hidden_dim")
    graph.add_node(Node("axis_protected", "protected_from_axis_summary", "ProtectedBoundary", inputs=[input_axis], outputs=[output_axis], semantic_role=role))
    return graph


def build_dfa_from_axis_patterns(
    region_spec: RegionSpec,
    axis_summary: RegionAxisSummary,
    pattern_matches: list[PatternMatch],
) -> Graph:
    """Construct a generic-label DFA graph from access-derived patterns."""
    del region_spec, axis_summary
    pattern = _single_supported_pattern(pattern_matches)
    if pattern.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN:
        return _build_ffn_graph()
    if pattern.pattern_kind == PatternKind.ATTENTION_VALUE_PATH:
        return _build_attention_value_graph()
    if pattern.pattern_kind == PatternKind.QK_SCORE_BLOCKER:
        return _build_qk_graph()
    if pattern.pattern_kind == PatternKind.RESIDUAL_HIDDEN_PROTECTED:
        return _build_protected_graph(SemanticRole.RESIDUAL_MERGE)
    if pattern.pattern_kind == PatternKind.LAYERNORM_HIDDEN_PROTECTED:
        return _build_protected_graph(SemanticRole.NORMALIZATION)
    if pattern.pattern_kind == PatternKind.INDEX_PRESERVING_UNARY:
        return _build_unary_graph()
    raise ValueError(f"unsupported pattern: {pattern.pattern_kind.value}")


def _find_axis(graph: Graph, tensor: str) -> Axis:
    return next(axis for axis in graph.all_axes() if axis.tensor == tensor)


def seed_facts_for_pattern(pattern_match: PatternMatch, seed_policy: BridgeSeedPolicy, graph: Graph) -> list[Fact]:
    """Create one symbolic seed fact for a lowered DFA graph."""
    if seed_policy.kind == "ffn_consumer_input_dead" and pattern_match.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN:
        axis = _find_axis(graph, "consumer_from_axis_summary.input")
        return [Fact(axis, FactKind.DEAD, seed_policy.explanation, "bridge_seed", pattern_match.evidence)]
    if seed_policy.kind == "attention_output_input_dead" and pattern_match.pattern_kind == PatternKind.ATTENTION_VALUE_PATH:
        axis = _find_axis(graph, "output_projection_from_axis_summary.input")
        return [Fact(axis, FactKind.DEAD, seed_policy.explanation, "bridge_seed", pattern_match.evidence)]
    if seed_policy.kind == "qk_query_output_pruned" and pattern_match.pattern_kind == PatternKind.QK_SCORE_BLOCKER:
        axis = _find_axis(graph, "query_from_axis_summary.output")
        return [Fact(axis, FactKind.PRUNED, seed_policy.explanation, "bridge_seed", pattern_match.evidence)]
    if seed_policy.kind in {"residual_hidden_pruned", "layernorm_hidden_pruned"} and pattern_match.pattern_kind in {
        PatternKind.RESIDUAL_HIDDEN_PROTECTED,
        PatternKind.LAYERNORM_HIDDEN_PROTECTED,
    }:
        axis = _find_axis(graph, "protected_from_axis_summary.input")
        return [Fact(axis, FactKind.PRUNED, seed_policy.explanation, "bridge_seed", pattern_match.evidence)]
    raise ValueError(f"seed policy {seed_policy.kind} does not apply to {pattern_match.pattern_kind.value}")


def run_bridge_analysis(region_spec: RegionSpec, seed_policy: BridgeSeedPolicy, *, example_name: str | None = None, interpretation: str = "") -> BridgeResult:
    """Run access analysis, pattern recognition, lowering, seeding, and DFA propagation."""
    trace = [
        BridgeTraceEvent("access_analysis", "Infer axis transfers from loop IV reuse and indexed tensor accesses.", (region_spec.region_id,)),
    ]
    axis_summary = analyze_region(region_spec)
    relation_count = sum(len(summary.transfers) for summary in axis_summary.op_summaries)
    trace.append(BridgeTraceEvent("access_analysis", f"Inferred {relation_count} axis transfers.", (axis_summary.explanation,)))
    patterns = recognize_patterns(region_spec, axis_summary)
    selected = _single_supported_pattern(patterns)
    trace.append(BridgeTraceEvent("pattern_recognition", f"Recognized {selected.pattern_kind.value}.", (*selected.required_relations, *selected.evidence)))
    graph = build_dfa_from_axis_patterns(region_spec, axis_summary, patterns)
    trace.append(BridgeTraceEvent("dfa_construction", "Lowered access-derived pattern into a generic-label semantic DFA graph.", tuple(graph.nodes)))
    seeds = seed_facts_for_pattern(selected, seed_policy, graph)
    trace.append(BridgeTraceEvent("dfa_seed", f"Seeded {seeds[0].describe()}.", (seed_policy.target_axis_hint, seed_policy.explanation)))
    dfa_result = analyze(graph, seeds)
    trace.append(BridgeTraceEvent("dfa_propagation", "Ran DFA worklist to fixed point.", (f"trace_events={len(dfa_result.trace)}",)))
    dead_axes = [axis.label() for axis, fact in dfa_result.state.items() if fact.kind == FactKind.DEAD]
    blocked_axes = [axis.label() for axis, fact in dfa_result.state.items() if fact.kind == FactKind.BLOCKED]
    protected_axes = [axis.label() for axis, fact in dfa_result.state.items() if fact.kind == FactKind.PROTECTED]
    return BridgeResult(
        example_name=example_name or region_spec.region_id,
        region_spec=region_spec,
        axis_summary=axis_summary,
        pattern_matches=patterns,
        dfa_graph=graph,
        seed_facts=seeds,
        dfa_result=dfa_result,
        bridge_trace=trace,
        summary={
            "recognized_patterns": [pattern.pattern_kind.value for pattern in patterns],
            "selected_pattern": selected.pattern_kind.value,
            "dfa_final_dead_axes": dead_axes,
            "dfa_blocked_axes": blocked_axes,
            "dfa_protected_axes": protected_axes,
            "reached_fixed_point": dfa_result.summary["reached_fixed_point"],
            "interpretation": interpretation or selected.explanation,
        },
    )
