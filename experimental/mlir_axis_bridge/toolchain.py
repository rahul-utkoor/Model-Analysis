"""Discover the local ONNX-MLIR and MLIR tools without mutating the environment."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolchainStatus:
    onnx_mlir_path: str | None
    mlir_opt_path: str | None
    onnx_mlir_available: bool
    mlir_opt_available: bool
    version_or_help_excerpt: str = ""
    warnings: list[str] = field(default_factory=list)


def default_toolchain_env() -> dict[str, str]:
    """Return suggested local toolchain variables without changing os.environ."""
    home = Path.home()
    onnx_root = Path(os.environ.get("ONNX_MLIR_ROOT", home / "Dev/onnx-mlir-work/onnx-mlir"))
    llvm_build = Path(os.environ.get("LLVM_BUILD", home / "Dev/onnx-mlir-work/llvm-project/build"))
    onnx_build = Path(os.environ.get("ONNX_MLIR_BUILD", onnx_root / "build"))
    return {
        "ONNX_MLIR_ROOT": str(onnx_root),
        "ONNX_MLIR_BUILD": str(onnx_build),
        "LLVM_BUILD": str(llvm_build),
        "MLIR_DIR": os.environ.get("MLIR_DIR", str(llvm_build / "lib/cmake/mlir")),
    }


def _candidate_paths(explicit_path: str | None, env_name: str, defaults: list[Path], executable: str) -> list[Path]:
    if explicit_path:
        return [Path(explicit_path).expanduser()]
    candidates: list[Path] = []
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]).expanduser())
    candidates.extend(defaults)
    from_path = shutil.which(executable)
    if from_path:
        candidates.append(Path(from_path))
    return candidates


def _find(candidates: list[Path], name: str) -> Path:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(f"{name} executable was not found; searched: {searched}")


def find_onnx_mlir(explicit_path: str | None = None) -> Path:
    env = default_toolchain_env()
    return _find(
        _candidate_paths(
            explicit_path,
            "ONNX_MLIR",
            [
                Path(env["ONNX_MLIR_BUILD"]) / "Release/bin/onnx-mlir",
                Path.home() / "Dev/onnx-mlir-work/onnx-mlir/build/Release/bin/onnx-mlir",
            ],
            "onnx-mlir",
        ),
        "onnx-mlir",
    )


def find_mlir_opt(explicit_path: str | None = None) -> Path:
    env = default_toolchain_env()
    return _find(
        _candidate_paths(
            explicit_path,
            "MLIR_OPT",
            [
                Path(env["LLVM_BUILD"]) / "bin/mlir-opt",
                Path.home() / "Dev/onnx-mlir-work/llvm-project/build/bin/mlir-opt",
            ],
            "mlir-opt",
        ),
        "mlir-opt",
    )


def find_native_pass_tool(explicit_path: str | None = None) -> Path:
    """Find the optional standalone pruning-axis dependence analyzer."""
    return _find(
        _candidate_paths(
            explicit_path,
            "PRUNING_AXIS_DEPENDENCE_TOOL",
            [Path(__file__).parent / "native/build/pruning-axis-dependence"],
            "pruning-axis-dependence",
        ),
        "pruning-axis-dependence",
    )


def _help_excerpt(path: Path) -> str:
    try:
        completed = subprocess.run([str(path), "--help"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unable to query help: {exc}"
    lines = (completed.stdout or completed.stderr).splitlines()
    return "\n".join(lines[:4])


def check_toolchain(onnx_mlir_path: str | None = None, mlir_opt_path: str | None = None) -> ToolchainStatus:
    """Return availability diagnostics instead of raising for missing tools."""
    warnings: list[str] = []
    try:
        onnx_mlir = find_onnx_mlir(onnx_mlir_path)
    except FileNotFoundError as exc:
        onnx_mlir = None
        warnings.append(str(exc))
    try:
        mlir_opt = find_mlir_opt(mlir_opt_path)
    except FileNotFoundError as exc:
        mlir_opt = None
        warnings.append(str(exc))
    return ToolchainStatus(
        onnx_mlir_path=str(onnx_mlir) if onnx_mlir else None,
        mlir_opt_path=str(mlir_opt) if mlir_opt else None,
        onnx_mlir_available=onnx_mlir is not None,
        mlir_opt_available=mlir_opt is not None,
        version_or_help_excerpt=_help_excerpt(onnx_mlir) if onnx_mlir else "",
        warnings=warnings,
    )
