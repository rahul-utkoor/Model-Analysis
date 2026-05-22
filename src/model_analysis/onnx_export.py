"""ONNX export helpers."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import torch

from model_analysis.hf_utils import load_model, load_tokenizer_or_processor
from model_analysis.paths import ensure_dir, get_project_root


class LogitsOnlyWrapper(torch.nn.Module):
    """Expose only logits so ONNX export avoids generation/cache outputs."""

    def __init__(self, model: torch.nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs):
        kwargs = dict(zip(self.input_names, inputs, strict=True))
        outputs = self.model(**kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0]


def _source_dir(config: dict[str, Any]) -> Path:
    return get_project_root() / config["local_dir"]


def _text_dummy_inputs(config: dict[str, Any], model: torch.nn.Module, source_dir: Path):
    tokenizer = load_tokenizer_or_processor(config, source=source_dir)
    encoded = tokenizer("This is a sample input for structural model analysis.", return_tensors="pt")
    signature = inspect.signature(model.forward)
    allowed = set(signature.parameters)

    input_names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in encoded and name in allowed]
    inputs = tuple(encoded[name] for name in input_names)
    dynamic_axes = {
        name: {0: "batch_size", 1: "sequence_length"}
        for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}
    return input_names, inputs, dynamic_axes


def _image_dummy_inputs():
    input_names = ["pixel_values"]
    inputs = (torch.randn(1, 3, 224, 224),)
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
    return input_names, inputs, dynamic_axes


def export_model_to_onnx(config: dict[str, Any], opset: int = 17) -> Path:
    """Export a locally downloaded HF model to ONNX and validate it."""
    source_dir = _source_dir(config)
    if not source_dir.exists() or not (source_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Local model not found at {source_dir}. Run: python scripts/download_models.py --model {config['name']}"
        )

    output_dir = ensure_dir(get_project_root() / config["onnx_dir"])
    output_path = output_dir / "model.onnx"
    metadata_path = output_dir / "metadata.json"

    model = load_model(config, source=source_dir)
    model.eval()
    model.config.return_dict = True
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if config["task"] == "image-classification":
        input_names, inputs, dynamic_axes = _image_dummy_inputs()
    else:
        input_names, inputs, dynamic_axes = _text_dummy_inputs(config, model, source_dir)

    wrapper = LogitsOnlyWrapper(model, input_names)
    output_names = ["logits"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    metadata = {
        "model_name": config["name"],
        "hf_id": config["hf_id"],
        "task": config["task"],
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "opset_version": opset,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "source_hf_directory": str(source_dir),
        "onnx_output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path
