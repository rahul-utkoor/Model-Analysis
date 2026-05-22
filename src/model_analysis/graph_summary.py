"""Reusable model structure summary helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from tabulate import tabulate

from model_analysis.torch_utils import count_parameters, count_trainable_parameters


def list_named_modules_by_type(model: torch.nn.Module, module_types: type | tuple[type, ...]) -> list[tuple[str, torch.nn.Module]]:
    """Return named modules matching one or more PyTorch module types."""
    return [(name, module) for name, module in model.named_modules() if name and isinstance(module, module_types)]


def list_attention_like_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Return modules whose names suggest attention behavior."""
    markers = ("attention", "attn", "self_attn", "self_attention")
    matches = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if name and any(marker in lowered for marker in markers):
            matches.append((name, module))
    return matches


def _module_rows(named_modules: Iterable[tuple[str, torch.nn.Module]], limit: int | None = None) -> list[list[str]]:
    rows = []
    for index, (name, module) in enumerate(named_modules):
        if limit is not None and index >= limit:
            break
        rows.append([name, module.__class__.__name__])
    return rows


def generate_markdown_summary(model: torch.nn.Module, config: dict[str, Any], max_rows: int = 200) -> str:
    """Generate a Markdown structural summary for a PyTorch model."""
    linear_layers = list_named_modules_by_type(model, torch.nn.Linear)
    embedding_layers = list_named_modules_by_type(model, torch.nn.Embedding)
    attention_like = list_attention_like_modules(model)
    top_level = [(name, module) for name, module in model.named_children()]

    sections = [
        f"# {config['name']}",
        "",
        "## Metadata",
        "",
        f"- Hugging Face ID: `{config['hf_id']}`",
        f"- Task: `{config['task']}`",
        f"- Parameters: `{count_parameters(model):,}`",
        f"- Trainable parameters: `{count_trainable_parameters(model):,}`",
        "",
        "## Top-Level Modules",
        "",
        tabulate(_module_rows(top_level), headers=["Name", "Type"], tablefmt="github") or "_None_",
        "",
        "## Linear Layers",
        "",
        tabulate(_module_rows(linear_layers, max_rows), headers=["Name", "Type"], tablefmt="github") or "_None_",
        "",
        "## Attention-Like Modules",
        "",
        tabulate(_module_rows(attention_like, max_rows), headers=["Name", "Type"], tablefmt="github") or "_None_",
        "",
        "## Embedding Layers",
        "",
        tabulate(_module_rows(embedding_layers, max_rows), headers=["Name", "Type"], tablefmt="github") or "_None_",
        "",
    ]
    return "\n".join(sections)
