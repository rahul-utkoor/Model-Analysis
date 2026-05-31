"""Run local ONNX-MLIR lowering stages for one selected subgraph."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MlirCommandResult:
    stage: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    generated_files: tuple[str, ...] = ()


@dataclass
class MlirLoweringResult:
    onnx_path: str
    output_root: str
    commands: list[MlirCommandResult] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _files(root: Path) -> set[str]:
    return {str(path) for path in root.rglob("*") if path.is_file()}


def _run(stage: str, command: list[str], output_root: Path) -> MlirCommandResult:
    before = _files(output_root)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except OSError as exc:
        returncode, stdout, stderr = 127, "", str(exc)
    generated = tuple(sorted(_files(output_root) - before))
    return MlirCommandResult(stage, tuple(command), stdout, stderr, returncode, generated)


def lower_onnx_subgraph_to_mlir(
    onnx_path: Path,
    output_root: Path,
    onnx_mlir_path: Path,
    preserve_mlir: bool = True,
) -> MlirLoweringResult:
    """Emit ONNX-dialect and lowered MLIR artifacts without executing the model."""
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX subgraph does not exist: {source}")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    result = MlirLoweringResult(str(source), str(output_root))
    commands = [
        ("onnx_dialect", [str(onnx_mlir_path), str(source), "--EmitONNXIR", "-o", str(output_root / f"{stem}_onnx")]),
        (
            "lowered_mlir",
            [
                str(onnx_mlir_path),
                str(source),
                "--EmitMLIR",
                *(["--preserveMLIR"] if preserve_mlir else []),
                "-o",
                str(output_root / f"{stem}_lowered"),
            ],
        ),
    ]
    for stage, command in commands:
        command_result = _run(stage, command, output_root)
        result.commands.append(command_result)
        if command_result.returncode:
            result.warnings.append(f"{stage} lowering failed with exit code {command_result.returncode}: {command_result.stderr.strip()}")
    result.generated_files = sorted(_files(output_root))
    if not result.generated_files:
        result.warnings.append("ONNX-MLIR did not emit any local artifacts")
    return result
