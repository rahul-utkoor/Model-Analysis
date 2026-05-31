"""Small explicit graph IR for the DFA pruning propagation prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class Axis:
    tensor: str
    dim: str
    role: str

    @property
    def key(self) -> str:
        return f"{self.tensor}:{self.dim}:{self.role}"

    def label(self) -> str:
        return f"{self.tensor}.{self.dim}<{self.role}>"


@dataclass
class Node:
    node_id: str
    name: str
    op_kind: str
    inputs: list[Axis] = field(default_factory=list)
    outputs: list[Axis] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def touches(self, axis: Axis) -> bool:
        return axis in self.inputs or axis in self.outputs


@dataclass(frozen=True)
class Edge:
    src_node: str
    src_axis: Axis
    dst_node: str
    dst_axis: Axis

    def label(self) -> str:
        return f"{self.src_node}:{self.src_axis.label()} -> {self.dst_node}:{self.dst_axis.label()}"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if edge.src_node not in self.nodes:
            raise KeyError(f"unknown source node: {edge.src_node}")
        if edge.dst_node not in self.nodes:
            raise KeyError(f"unknown destination node: {edge.dst_node}")
        if edge.src_axis not in self.nodes[edge.src_node].outputs:
            raise ValueError(f"source axis is not an output of {edge.src_node}: {edge.src_axis.label()}")
        if edge.dst_axis not in self.nodes[edge.dst_node].inputs:
            raise ValueError(f"destination axis is not an input of {edge.dst_node}: {edge.dst_axis.label()}")
        self.edges.append(edge)
        return edge

    def find_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def successors(self, node_id: str) -> list[Node]:
        return [self.nodes[edge.dst_node] for edge in self.edges if edge.src_node == node_id]

    def predecessors(self, node_id: str) -> list[Node]:
        return [self.nodes[edge.src_node] for edge in self.edges if edge.dst_node == node_id]

    def touching_nodes(self, axis: Axis) -> list[Node]:
        return [node for node in self.nodes.values() if node.touches(axis)]

    def adjacent_axes(self, axis: Axis) -> list[tuple[Axis, Edge]]:
        out: list[tuple[Axis, Edge]] = []
        for edge in self.edges:
            if edge.src_axis == axis:
                out.append((edge.dst_axis, edge))
            if edge.dst_axis == axis:
                out.append((edge.src_axis, edge))
        return out

    def all_axes(self) -> list[Axis]:
        return sorted({axis for node in self.nodes.values() for axis in [*node.inputs, *node.outputs]})

    def pretty_print(self) -> str:
        lines = ["graph {"]
        for node in self.nodes.values():
            lines.append(f'  {node.node_id}: {node.name} [{node.op_kind}]')
            for axis in node.inputs:
                lines.append(f"    input  {axis.label()}")
            for axis in node.outputs:
                lines.append(f"    output {axis.label()}")
        lines.append("  edges:")
        for edge in self.edges:
            lines.append(f"    {edge.label()}")
        lines.append("}")
        return "\n".join(lines)
