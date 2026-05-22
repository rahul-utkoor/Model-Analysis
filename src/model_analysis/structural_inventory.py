"""PyTorch structural inventory helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch

from model_analysis.torch_utils import count_parameters, count_trainable_parameters


ATTENTION_MARKERS = (
    "attention",
    "attn",
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "query",
    "key",
    "value",
)

MLP_MARKERS = (
    "mlp",
    "ffn",
    "feed_forward",
    "intermediate",
    "output.dense",
    "fc1",
    "fc2",
)

NORMALIZATION_TYPES = (
    torch.nn.LayerNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
)


def _module_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _contains_marker(name: str, markers: tuple[str, ...]) -> str | None:
    lowered = name.lower()
    for marker in markers:
        if marker in lowered:
            return marker
    return None


def _parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _linear_layer_entry(name: str, module: torch.nn.Linear) -> dict[str, Any]:
    return {
        "name": name,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "bias": module.bias is not None,
        "parameters": _module_parameter_count(module),
    }


def _embedding_layer_entry(name: str, module: torch.nn.Embedding) -> dict[str, Any]:
    return {
        "name": name,
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "parameters": _module_parameter_count(module),
    }


def _normalization_layer_entry(name: str, module: torch.nn.Module) -> dict[str, Any]:
    return {
        "name": name,
        "type": module.__class__.__name__,
        "parameters": _module_parameter_count(module),
    }


def _find_attention_like_modules(named_modules: list[tuple[str, torch.nn.Module]]) -> list[dict[str, str]]:
    entries = []
    for name, module in named_modules:
        if not name:
            continue
        marker = _contains_marker(name, ATTENTION_MARKERS)
        if marker:
            entries.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "reason": f"name contains '{marker}'",
                }
            )
    return entries


def _find_mlp_like_modules(named_modules: list[tuple[str, torch.nn.Module]]) -> list[dict[str, str]]:
    entries = []
    for name, module in named_modules:
        if not name:
            continue
        marker = _contains_marker(name, MLP_MARKERS)
        if marker:
            entries.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "reason": f"name contains '{marker}'",
                }
            )
    return entries


def _projection_role(name: str) -> str | None:
    leaf = _leaf_name(name).lower()
    if leaf in {"q_proj", "query"} or leaf.endswith(".query"):
        return "q"
    if leaf in {"k_proj", "key"} or leaf.endswith(".key"):
        return "k"
    if leaf in {"v_proj", "value"} or leaf.endswith(".value"):
        return "v"
    return None


def _mlp_parent_for(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] in {"intermediate", "output"}:
        return ".".join(parts[:-2])
    return _parent_name(name)


def _find_pruning_relevant_groups(
    linear_layers: list[dict[str, Any]],
    embedding_layers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    qkv_by_parent: dict[str, dict[str, str]] = defaultdict(dict)
    for layer in linear_layers:
        role = _projection_role(layer["name"])
        if role:
            qkv_by_parent[_parent_name(layer["name"])][role] = layer["name"]

    for parent, roles in qkv_by_parent.items():
        if len(roles) >= 2:
            members = [roles[key] for key in ("q", "k", "v") if key in roles]
            groups.append(
                {
                    "group_name": f"{parent or 'root'}:qkv_projections",
                    "group_type": "attention_qkv",
                    "members": members,
                    "reason": "Q/K/V projection layers share a common parent name.",
                    "confidence": "high" if len(roles) == 3 else "medium",
                }
            )

    for layer in linear_layers:
        lowered = layer["name"].lower()
        leaf = _leaf_name(lowered)
        if leaf == "out_proj" or ("attention" in lowered and leaf in {"dense", "o_proj"}):
            groups.append(
                {
                    "group_name": f"{layer['name']}:attention_output_projection",
                    "group_type": "attention_output_projection",
                    "members": [layer["name"]],
                    "reason": "Layer name suggests an attention output projection.",
                    "confidence": "medium",
                }
            )

    mlp_by_parent: dict[str, list[str]] = defaultdict(list)
    for layer in linear_layers:
        name = layer["name"].lower()
        if _contains_marker(name, MLP_MARKERS):
            mlp_by_parent[_mlp_parent_for(layer["name"])].append(layer["name"])

    for parent, members in mlp_by_parent.items():
        if len(members) >= 2:
            groups.append(
                {
                    "group_name": f"{parent or 'root'}:mlp_projections",
                    "group_type": "mlp_projection_pair",
                    "members": members,
                    "reason": "MLP-like expansion/projection layers share a common parent.",
                    "confidence": "medium",
                }
            )

    for layer in embedding_layers:
        groups.append(
            {
                "group_name": f"{layer['name']}:embedding_matrix",
                "group_type": "embedding_matrix",
                "members": [layer["name"]],
                "reason": "Embedding matrices are structurally prunable only with vocabulary/output dependencies considered.",
                "confidence": "low",
            }
        )

    return groups


def summarize_torch_model(model: torch.nn.Module, model_name: str, model_config: dict) -> dict[str, Any]:
    """Build a conservative structural inventory for a PyTorch model."""
    named_modules = list(model.named_modules())
    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    parameter_distribution = Counter()
    for _, module in named_modules:
        parameter_distribution[module.__class__.__name__] += _module_parameter_count(module)

    linear_layers = [
        _linear_layer_entry(name, module)
        for name, module in named_modules
        if name and isinstance(module, torch.nn.Linear)
    ]
    embedding_layers = [
        _embedding_layer_entry(name, module)
        for name, module in named_modules
        if name and isinstance(module, torch.nn.Embedding)
    ]
    normalization_layers = [
        _normalization_layer_entry(name, module)
        for name, module in named_modules
        if name and isinstance(module, NORMALIZATION_TYPES)
    ]

    return {
        "model_name": model_name,
        "hf_id": model_config.get("hf_id"),
        "task": model_config.get("task"),
        "parameter_summary": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "non_trainable_parameters": total_parameters - trainable_parameters,
        },
        "module_summary": {
            "total_modules": len(named_modules),
            "module_type_counts": dict(Counter(module.__class__.__name__ for _, module in named_modules)),
            "parameter_distribution_by_module_type": dict(parameter_distribution),
        },
        "linear_layers": linear_layers,
        "embedding_layers": embedding_layers,
        "normalization_layers": normalization_layers,
        "attention_like_modules": _find_attention_like_modules(named_modules),
        "mlp_like_modules": _find_mlp_like_modules(named_modules),
        "pruning_relevant_groups": _find_pruning_relevant_groups(linear_layers, embedding_layers),
    }
