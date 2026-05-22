"""Small PyTorch helpers."""

from __future__ import annotations


def count_parameters(model) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
