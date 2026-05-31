"""Conservative text extraction for selected MLIR operations and indexed accesses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from experimental.mlir_axis_bridge.mlir_artifacts import MlirArtifact


RECOGNIZED_OPS = (
    "affine.for", "scf.for", "linalg.generic", "linalg.matmul", "krnl.matmul", "krnl.iterate",
    "affine.load", "affine.store", "memref.load", "memref.store",
    "onnx.MatMul", "onnx.Gemm", "onnx.Add", "onnx.Mul", "onnx.Erf", "onnx.Tanh",
    "onnx.LayerNormalization", "onnx.Reshape", "onnx.Transpose",
)
ACCESS_RE = re.compile(r"(?P<kind>affine|memref)\.(?P<action>load|store)\s+(?:[^,]+,\s+)?(?P<tensor>%[-\w.$]+)\[(?P<indices>[^]]*)\]")
RESULT_RE = re.compile(r"^\s*(?P<results>%[-\w.$]+(?:\s*,\s*%[-\w.$]+)*)\s*=")


@dataclass(frozen=True)
class MlirOperationRecord:
    op_name: str
    result_names: tuple[str, ...]
    operands: tuple[str, ...]
    source_line: str
    line_no: int
    dialect: str
    attributes_text: str = ""


@dataclass(frozen=True)
class MlirAccessRecord:
    tensor: str
    indices: tuple[str, ...]
    access_kind: str
    line_no: int
    raw_line: str


@dataclass
class MlirParseResult:
    operations: list[MlirOperationRecord] = field(default_factory=list)
    accesses: list[MlirAccessRecord] = field(default_factory=list)


def _strip_ssa(value: str) -> str:
    return value.strip().lstrip("%")


def parse_mlir_text(text: str) -> MlirParseResult:
    result = MlirParseResult()
    for line_no, line in enumerate(text.splitlines(), start=1):
        access = ACCESS_RE.search(line)
        if access:
            result.accesses.append(
                MlirAccessRecord(
                    _strip_ssa(access.group("tensor")),
                    tuple(_strip_ssa(index) for index in access.group("indices").split(",") if index.strip()),
                    "read" if access.group("action") == "load" else "write",
                    line_no,
                    line.strip(),
                )
            )
        matched_ops = [op for op in RECOGNIZED_OPS if op in line]
        for op_name in matched_ops:
            result_match = RESULT_RE.search(line)
            results = tuple(_strip_ssa(item) for item in result_match.group("results").split(",")) if result_match else ()
            operands = tuple(_strip_ssa(item) for item in re.findall(r"%[-\w.$]+", line))
            result.operations.append(MlirOperationRecord(op_name, results, operands, line.strip(), line_no, op_name.split(".", 1)[0], ""))
    return result


def parse_mlir_artifact(artifact: MlirArtifact) -> MlirParseResult:
    return parse_mlir_text(Path(artifact.path).read_text(encoding="utf-8", errors="replace"))
