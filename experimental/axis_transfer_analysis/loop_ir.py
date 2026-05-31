"""Small MLIR-inspired loop/access IR for pruning-axis analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoopIV:
    name: str
    role: str | None = None


@dataclass(frozen=True)
class Tensor:
    name: str
    rank: int
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank != len(self.axes):
            raise ValueError(f"tensor {self.name} rank {self.rank} does not match {len(self.axes)} axes")


@dataclass(frozen=True)
class TensorAccess:
    tensor: str
    indices: tuple[str, ...]
    access_kind: str


@dataclass
class OperationSpec:
    op_id: str
    label: str
    op_kind: str
    loops: list[LoopIV]
    reads: list[TensorAccess]
    writes: list[TensorAccess]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionEdge:
    src_op: str
    tensor: str
    dst_op: str


@dataclass
class RegionSpec:
    region_id: str
    label: str
    tensors: dict[str, Tensor]
    ops: list[OperationSpec]
    edges: list[RegionEdge] = field(default_factory=list)

    def tensor(self, name: str) -> Tensor:
        return self.tensors[name]

    def op(self, op_id: str) -> OperationSpec:
        return next(op for op in self.ops if op.op_id == op_id)


def access_form(op: OperationSpec) -> str:
    """Render a compact indexed-access form for one operation."""
    writes = ", ".join(_render_access(access) for access in op.writes) or "_"
    reads = ", ".join(_render_access(access) for access in op.reads) or "_"
    assignment = "+=" if any(access.access_kind == "update" for access in op.writes) else "="
    return f"{writes} {assignment} {op.op_kind}({reads})"


def _render_access(access: TensorAccess) -> str:
    return f"{access.tensor}[{','.join(access.indices)}]"
