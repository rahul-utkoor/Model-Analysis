"""Discover emitted MLIR text artifacts and the dialect clues they contain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experimental.mlir_axis_bridge.onnx_mlir_runner import MlirLoweringResult


DIALECT_MARKERS = (
    "onnx.",
    "krnl.",
    "linalg.",
    "scf.for",
    "affine.for",
    "memref.load",
    "affine.load",
    "memref.store",
    "affine.store",
)


@dataclass(frozen=True)
class MlirArtifact:
    path: str
    stage: str
    dialect_hints: tuple[str, ...]
    size_bytes: int
    first_lines: tuple[str, ...]


def detect_dialect_hints(text: str) -> tuple[str, ...]:
    return tuple(marker for marker in DIALECT_MARKERS if marker in text)


def artifact_from_path(path: str | Path, stage: str = "unknown") -> MlirArtifact:
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    return MlirArtifact(str(source), stage, detect_dialect_hints(text), source.stat().st_size, tuple(text.splitlines()[:12]))


def _stage(path: Path) -> str:
    if path.name.endswith(".input.mlir"):
        return "preserved_input"
    if "_onnx" in path.name:
        return "onnx_dialect"
    if "_lowered" in path.name:
        return "lowered_mlir"
    return "preserved_temp"


def discover_mlir_artifacts(lowering: MlirLoweringResult) -> list[MlirArtifact]:
    artifacts: list[MlirArtifact] = []
    for raw_path in lowering.generated_files:
        path = Path(raw_path)
        if path.suffix not in {".mlir", ".tmp"} or not path.is_file():
            continue
        artifacts.append(artifact_from_path(path, _stage(path)))
    return sorted(artifacts, key=lambda artifact: (artifact.stage, artifact.path))
