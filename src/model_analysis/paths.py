"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root path."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return current.parents[2]


def get_config_path() -> Path:
    """Return the default model registry config path."""
    return get_project_root() / "configs" / "models.yaml"


def safe_model_name(model_name: str) -> str:
    """Convert a model name or HF ID into a filesystem-safe name."""
    return model_name.replace("/", "__")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_hf_model_dir(model_name: str) -> Path:
    """Return the local Hugging Face model directory for a model name or HF ID."""
    return get_project_root() / "data" / "models" / "hf" / safe_model_name(model_name)


def get_onnx_model_dir(model_name: str) -> Path:
    """Return the local ONNX export directory for a model name or HF ID."""
    return get_project_root() / "data" / "models" / "onnx" / safe_model_name(model_name)
