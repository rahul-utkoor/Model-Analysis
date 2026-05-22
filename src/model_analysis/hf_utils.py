"""Hugging Face loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


MODEL_CLASSES = {
    "masked-lm": AutoModelForMaskedLM,
    "causal-lm": AutoModelForCausalLM,
    "image-classification": AutoModelForImageClassification,
}


def get_model_class(task: str):
    """Return the Hugging Face AutoModel class for a registry task."""
    try:
        return MODEL_CLASSES[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported task '{task}'") from exc


def load_model(config: dict[str, Any], source: str | Path | None = None, cache_dir: str | None = None):
    """Load a model for the registry entry from a local path or HF ID."""
    model_source = str(source or config["hf_id"])
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return get_model_class(config["task"]).from_pretrained(model_source, **kwargs)


def load_tokenizer_or_processor(
    config: dict[str, Any],
    source: str | Path | None = None,
    cache_dir: str | None = None,
):
    """Load the tokenizer or image processor needed for a model."""
    model_source = str(source or config["hf_id"])
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if config["task"] == "image-classification":
        return AutoImageProcessor.from_pretrained(model_source, **kwargs)
    return AutoTokenizer.from_pretrained(model_source, **kwargs)
