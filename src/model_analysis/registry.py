"""Model registry loading and resolution."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from model_analysis.paths import get_config_path


@lru_cache(maxsize=1)
def load_model_registry() -> list[dict[str, Any]]:
    """Load configured models from ``configs/models.yaml``."""
    config_path = get_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"Expected a list under 'models' in {config_path}")
    return models


def list_models() -> list[str]:
    """Return configured model names."""
    return [model["name"] for model in load_model_registry()]


def resolve_model_name(model_name_or_hf_id: str) -> str:
    """Resolve a short name, HF ID, or HF basename to a configured model name."""
    value = model_name_or_hf_id.strip()
    for model in load_model_registry():
        if value in {model.get("name"), model.get("hf_id")}:
            return model["name"]

    for model in load_model_registry():
        hf_id = model.get("hf_id", "")
        if value == hf_id.split("/")[-1]:
            return model["name"]

    available = ", ".join(list_models())
    raise KeyError(f"Unknown model '{model_name_or_hf_id}'. Available models: {available}")


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return the registry entry for a configured model."""
    resolved = resolve_model_name(model_name)
    for model in load_model_registry():
        if model.get("name") == resolved:
            return model
    raise KeyError(f"Unknown model '{model_name}'")
