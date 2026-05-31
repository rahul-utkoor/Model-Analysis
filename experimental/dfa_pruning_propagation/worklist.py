"""Queue-based fixed-point engine for pruning/deadness propagation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from experimental.dfa_pruning_propagation.ir import Axis, Graph
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind, join, unknown
from experimental.dfa_pruning_propagation.transfer import TransferEmission, transfer


@dataclass
class TraceEvent:
    step: int
    node: str
    input_fact: str
    output_fact: str
    action: str
    explanation: str


@dataclass
class AnalysisResult:
    graph: Graph
    seed_facts: list[Fact]
    state: dict[Axis, Fact]
    trace: list[TraceEvent] = field(default_factory=list)
    blocked_events: list[TraceEvent] = field(default_factory=list)
    propagated_events: list[TraceEvent] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _enqueue(queue: deque[Fact], emission: TransferEmission, trace: list[TraceEvent], step: int, input_fact: Fact, node: str) -> int:
    event = TraceEvent(
        step=step,
        node=node,
        input_fact=input_fact.describe(),
        output_fact=emission.fact.describe(),
        action=emission.action,
        explanation=emission.explanation,
    )
    trace.append(event)
    queue.append(emission.fact)
    return step + 1


def analyze(graph: Graph, seed_facts: list[Fact]) -> AnalysisResult:
    """Propagate facts until the queue reaches a fixed point."""
    state = {axis: unknown(axis) for axis in graph.all_axes()}
    queue: deque[Fact] = deque(seed_facts)
    trace: list[TraceEvent] = []
    step = 1
    while queue:
        incoming = queue.popleft()
        old = state.get(incoming.axis, unknown(incoming.axis))
        current = join(old, incoming)
        if current == old:
            continue
        state[incoming.axis] = current
        action = "blocked" if current.kind == FactKind.BLOCKED else "joined"
        trace.append(TraceEvent(step, incoming.source_node or "seed", old.describe(), current.describe(), action, current.reason))
        step += 1
        if current.kind == old.kind:
            continue
        for adjacent_axis, edge in graph.adjacent_axes(incoming.axis):
            emission = TransferEmission(
                fact=Fact(adjacent_axis, current.kind, f"edge equivalence from {incoming.axis.label()}", incoming.source_node, (*current.evidence, edge.label())),
                action="blocked" if current.kind == FactKind.BLOCKED else "propagated",
                explanation="Connected edge carries the same structural axis fact.",
            )
            step = _enqueue(queue, emission, trace, step, current, f"edge:{edge.src_node}->{edge.dst_node}")
        for node in graph.touching_nodes(incoming.axis):
            for emission in transfer(node, state, incoming.axis):
                step = _enqueue(queue, emission, trace, step, current, node.node_id)
    blocked = [event for event in trace if event.action == "blocked" or " = BLOCKED:" in event.output_fact]
    propagated = [event for event in trace if event.action == "propagated"]
    protected = sum(fact.kind == FactKind.PROTECTED for fact in state.values())
    return AnalysisResult(
        graph=graph,
        seed_facts=seed_facts,
        state=state,
        trace=trace,
        blocked_events=blocked,
        propagated_events=propagated,
        summary={
            "num_seed_facts": len(seed_facts),
            "num_propagated": len(propagated),
            "num_blocked": len(blocked),
            "num_protected": protected,
            "reached_fixed_point": not queue,
        },
    )


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "graph": {
            "nodes": {node_id: asdict(node) for node_id, node in result.graph.nodes.items()},
            "edges": [asdict(edge) for edge in result.graph.edges],
        },
        "seed_facts": [asdict(fact) for fact in result.seed_facts],
        "state": {axis.key: asdict(fact) for axis, fact in sorted(result.state.items())},
        "trace": [asdict(event) for event in result.trace],
        "blocked_events": [asdict(event) for event in result.blocked_events],
        "summary": result.summary,
    }
