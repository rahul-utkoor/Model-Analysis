#!/usr/bin/env python3
"""
Export Hugging Face models to static-shape ONNX for Netron visualization.

This is different from the normal dynamic ONNX export. It intentionally fixes
batch size and sequence length so Netron can show concrete tensor shapes.

Example:

  ./conda-env/bin/python scripts/export_static_shape_onnx.py \
    --model bert-base-uncased \
    --seq-len max \
    --batch-size 1 \
    --opset 17

Output:

  data/models/onnx_static/bert-base-uncased/model.static.onnx
  data/models/onnx_static/bert-base-uncased/metadata.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import onnx
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
)

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, load_model_registry, resolve_model_name


def extract_logits(outputs: Any) -> torch.Tensor:
    """Return the primary logits tensor from a Transformers-style output."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise ValueError("Model output does not expose logits or a first tensor output.")


class TextModelOnnxWrapper(torch.nn.Module):
    """Trace a text model while retaining keyword-based Hugging Face inputs."""

    def __init__(self, model: torch.nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        kwargs = {name: value for name, value in zip(self.input_names, args)}
        return extract_logits(self.model(**kwargs))


class ImageModelOnnxWrapper(torch.nn.Module):
    """Trace an image classifier through its named pixel-values input."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return extract_logits(self.model(pixel_values=pixel_values))


def filter_inputs_for_model(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Keep only tokenizer tensors accepted by ``model.forward``."""
    signature = inspect.signature(model.forward)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = set(signature.parameters)
    filtered: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for name, value in inputs.items():
        if name == "input_ids" or name in accepted or accepts_kwargs:
            filtered[name] = value
        else:
            dropped.append(name)
    return filtered, dropped


def infer_max_seq_len(config: Any, fallback: int = 512) -> int:
    """Infer maximum useful token length from HF config."""
    for attr in [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_sequence_length",
    ]:
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def parse_seq_len(seq_len: str, config: Any) -> int:
    if seq_len == "max":
        return infer_max_seq_len(config)
    value = int(seq_len)
    if value <= 0:
        raise ValueError("--seq-len must be positive or 'max'")
    return value


def load_model_for_task(task: str, model_dir: Path):
    if task == "masked-lm":
        return AutoModelForMaskedLM.from_pretrained(model_dir)
    if task == "causal-lm":
        return AutoModelForCausalLM.from_pretrained(model_dir)
    if task == "image-classification":
        return AutoModelForImageClassification.from_pretrained(model_dir)
    raise ValueError(f"Unsupported task for static ONNX export: {task}")


def build_text_inputs(
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    text = "This is a static shape ONNX export for Netron visualization."
    texts = [text for _ in range(batch_size)]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    inputs: dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].to(device)

    return inputs


def export_static_text_model(
    model,
    tokenizer,
    output_path: Path,
    batch_size: int,
    seq_len: int,
    opset: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.to(device)

    tokenizer_inputs = build_text_inputs(tokenizer, batch_size, seq_len, device)
    inputs, dropped_inputs = filter_inputs_for_model(model, tokenizer_inputs)
    input_names = list(inputs.keys())
    args = tuple(inputs[name] for name in input_names)
    wrapper = TextModelOnnxWrapper(model, input_names)
    wrapper.eval()
    output_names = ["logits"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,  # Important: force static shapes.
            dynamo=False,  # Use legacy exporter for dynamic_axes/static compatibility.
        )

    return {
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "dropped_inputs": dropped_inputs,
    }


def export_static_image_model(
    model,
    output_path: Path,
    batch_size: int,
    image_size: int,
    opset: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.to(device)

    pixel_values = torch.zeros(
        batch_size,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
        device=device,
    )
    wrapper = ImageModelOnnxWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    return {
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
        "input_shapes": {"pixel_values": list(pixel_values.shape)},
        "dropped_inputs": [],
    }


def run_shape_inference(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, str(onnx_path))


def export_one_model(
    model_name_or_id: str,
    seq_len_arg: str,
    batch_size: int,
    image_size: int,
    opset: int,
    device_str: str,
) -> dict[str, Any]:
    resolved = resolve_model_name(model_name_or_id)
    cfg = get_model_config(resolved)

    model_name = cfg["name"]
    hf_id = cfg["hf_id"]
    task = cfg["task"]
    safe_name = safe_model_name(hf_id)

    root = get_project_root()
    model_dir = root / cfg["local_dir"]
    output_dir = root / "data" / "models" / "onnx_static" / safe_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {model_dir}\n"
            f"Run: python scripts/download_models.py --model {model_name}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if device_str == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_str == "auto"
        else device_str
    )

    config = AutoConfig.from_pretrained(model_dir)
    model = load_model_for_task(task, model_dir)

    output_path = output_dir / "model.static.onnx"

    print(f"[static-onnx] model={model_name}")
    print(f"[static-onnx] task={task}")
    print(f"[static-onnx] source={model_dir}")
    print(f"[static-onnx] output={output_path}")
    print(f"[static-onnx] device={device}")

    if task in {"masked-lm", "causal-lm"}:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        seq_len = parse_seq_len(seq_len_arg, config)
        print(f"[static-onnx] batch_size={batch_size}, seq_len={seq_len}")

        export_info = export_static_text_model(
            model=model,
            tokenizer=tokenizer,
            output_path=output_path,
            batch_size=batch_size,
            seq_len=seq_len,
            opset=opset,
            device=device,
        )

        extra = {
            "seq_len": seq_len,
            "batch_size": batch_size,
        }

    elif task == "image-classification":
        # Load processor only to preserve local compatibility; dummy tensor is enough for export.
        _ = AutoImageProcessor.from_pretrained(model_dir)
        print(f"[static-onnx] batch_size={batch_size}, image_size={image_size}")

        export_info = export_static_image_model(
            model=model,
            output_path=output_path,
            batch_size=batch_size,
            image_size=image_size,
            opset=opset,
            device=device,
        )

        extra = {
            "image_size": image_size,
            "batch_size": batch_size,
        }

    else:
        raise ValueError(f"Unsupported task: {task}")

    print("[static-onnx] running ONNX shape inference")
    run_shape_inference(output_path)

    metadata = {
        "model_name": model_name,
        "hf_id": hf_id,
        "task": task,
        "source_model_dir": str(model_dir),
        "output_onnx_path": str(output_path),
        "opset": opset,
        "static_shapes": True,
        "dynamic_axes": False,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
        **export_info,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("[static-onnx] success")
    print(f"[static-onnx] open with:")
    print(f"  netron {output_path}")
    return metadata


def _write_summary(
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    report_dir = get_project_root() / "reports" / "static_onnx_exports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "static_onnx_export_summary.json"
    summary = {
        "static_shapes": True,
        "dynamic_axes": False,
        "requested_model": args.model,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "image_size": args.image_size,
        "opset": args.opset,
        "device": args.device,
        "continue_on_error": args.continue_on_error,
        "num_successful_exports": len(results),
        "num_failed_exports": len(failures),
        "exports": results,
        "failures": failures,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export static-shape ONNX models for Netron visualization."
    )
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    parser.add_argument(
        "--seq-len",
        default="max",
        help="Sequence length for text models. Use integer or 'max'. Default: max",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="For --model all, report failures and continue exporting remaining models.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.model == "all":
        models = [config["name"] for config in load_model_registry()]
    else:
        models = [args.model]

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_name in models:
        try:
            successes.append(
                export_one_model(
                    model_name_or_id=model_name,
                    seq_len_arg=args.seq_len,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    opset=args.opset,
                    device_str=args.device,
                )
            )
        except Exception as exc:
            failure = {"model_name": model_name, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            print(f"[static-onnx][error] failed to export {model_name}: {exc}", file=sys.stderr)
            if not (args.model == "all" and args.continue_on_error):
                summary_path = _write_summary(successes, failures, args)
                print(f"[static-onnx] summary={summary_path}")
                return 1
    summary_path = _write_summary(successes, failures, args)
    print(f"[static-onnx] summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
