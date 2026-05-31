"""Run the optional standalone native MLIR dependence analyzer."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativePassRunResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    json_path: str | None
    warning: str | None = None


def run_native_dependence_tool(input_mlir: str | Path, tool_path: str | Path, output_json: str | Path) -> NativePassRunResult:
    """Run the local native tool without transforming the selected MLIR file."""
    source = Path(input_mlir)
    tool = Path(tool_path)
    output = Path(output_json)
    command = (str(tool), str(source), "--output", str(output))
    if not tool.is_file():
        return NativePassRunResult(command, 127, "", "", None, f"native pass tool does not exist: {tool}")
    if not source.is_file():
        return NativePassRunResult(command, 2, "", "", None, f"native pass input MLIR does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return NativePassRunResult(command, 127, "", str(exc), None, f"native pass tool failed to start: {exc}")
    warning = None
    json_path = str(output) if completed.returncode == 0 and output.is_file() else None
    if completed.returncode:
        warning = f"native pass tool failed with exit code {completed.returncode}: {completed.stderr.strip()}"
    elif json_path is None:
        warning = "native pass tool completed without writing JSON output"
    return NativePassRunResult(command, completed.returncode, completed.stdout, completed.stderr, json_path, warning)
