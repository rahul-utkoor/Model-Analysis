"""Utilities for narrow, reversible PyTorch Linear pruning."""

from __future__ import annotations

from typing import Any

import torch


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """Return a nested module by dotted module name."""
    current = model
    if not module_name:
        return current
    for part in module_name.split("."):
        if not hasattr(current, part):
            raise KeyError(f"Module '{module_name}' not found at '{part}'")
        current = getattr(current, part)
    return current


def replace_module_by_name(model: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
    """Replace a nested module by dotted module name."""
    if "." not in module_name:
        if not hasattr(model, module_name):
            raise KeyError(f"Module '{module_name}' not found")
        setattr(model, module_name, new_module)
        return

    parent_name, child_name = module_name.rsplit(".", 1)
    parent = get_module_by_name(model, parent_name)
    if not hasattr(parent, child_name):
        raise KeyError(f"Module '{module_name}' not found")
    setattr(parent, child_name, new_module)


def _normalize_prune_indices(indices: list[int]) -> list[int]:
    return sorted(set(indices))


def make_keep_indices(size: int, prune_indices: list[int]) -> list[int]:
    """Return indices to keep after validating prune indices."""
    normalized = _normalize_prune_indices(prune_indices)
    if any(index < 0 for index in normalized):
        raise ValueError("Prune indices must be non-negative.")
    if any(index >= size for index in normalized):
        raise ValueError(f"Prune index out of bounds for size {size}.")
    if len(normalized) >= size:
        raise ValueError("Cannot prune all features.")
    return [index for index in range(size) if index not in set(normalized)]


def prune_linear_layer(
    layer: torch.nn.Linear,
    prune_dim: str,
    indices: list[int],
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    """Prune rows or columns of a Linear layer and return a replacement layer."""
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError("prune_linear_layer only supports torch.nn.Linear.")
    if prune_dim not in {"out_features", "in_features"}:
        raise ValueError(f"Unsupported Linear prune_dim '{prune_dim}'.")

    old_weight_shape = list(layer.weight.shape)
    old_bias_shape = list(layer.bias.shape) if layer.bias is not None else None
    if prune_dim == "out_features":
        keep_indices = make_keep_indices(layer.out_features, indices)
        new_layer = torch.nn.Linear(layer.in_features, len(keep_indices), bias=layer.bias is not None)
        with torch.no_grad():
            index_tensor = torch.tensor(keep_indices, device=layer.weight.device)
            new_layer.weight.copy_(layer.weight.index_select(0, index_tensor))
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias.index_select(0, index_tensor))
    else:
        keep_indices = make_keep_indices(layer.in_features, indices)
        new_layer = torch.nn.Linear(len(keep_indices), layer.out_features, bias=layer.bias is not None)
        with torch.no_grad():
            index_tensor = torch.tensor(keep_indices, device=layer.weight.device)
            new_layer.weight.copy_(layer.weight.index_select(1, index_tensor))
            if layer.bias is not None:
                new_layer.bias.copy_(layer.bias)

    new_layer = new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)
    new_layer.weight.requires_grad = layer.weight.requires_grad
    if layer.bias is not None and new_layer.bias is not None:
        new_layer.bias.requires_grad = layer.bias.requires_grad

    metadata = {
        "prune_dim": prune_dim,
        "prune_indices": _normalize_prune_indices(indices),
        "keep_indices": keep_indices,
        "old_weight_shape": old_weight_shape,
        "new_weight_shape": list(new_layer.weight.shape),
        "old_bias_shape": old_bias_shape,
        "new_bias_shape": list(new_layer.bias.shape) if new_layer.bias is not None else None,
    }
    return new_layer, metadata
