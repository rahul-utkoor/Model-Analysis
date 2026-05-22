"""Forward-pass smoke validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch


@dataclass
class ForwardSmokeResult:
    validation_id: str
    model_name: str
    model_dir: str | None
    status: str
    input_kind: str
    output_summary: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _now_id(prefix: str) -> str:
    return f"{prefix}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "type": "Tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def summarize_model_output(output: Any) -> dict[str, Any]:
    """Return a compact structural summary of a forward output."""
    if isinstance(output, torch.Tensor):
        return _tensor_summary(output)
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        summary = {"type": output.__class__.__name__, "logits": _tensor_summary(output.logits)}
        if hasattr(output, "keys"):
            summary["keys"] = list(output.keys())
        return summary
    if isinstance(output, dict) or hasattr(output, "items"):
        items = dict(output.items())
        return {
            "type": output.__class__.__name__,
            "keys": list(items.keys()),
            "tensors": {key: _tensor_summary(value) for key, value in items.items() if isinstance(value, torch.Tensor)},
        }
    if isinstance(output, (tuple, list)):
        tensor_items = [
            {"index": index, **_tensor_summary(value)}
            for index, value in enumerate(output[:8])
            if isinstance(value, torch.Tensor)
        ]
        return {"type": output.__class__.__name__, "length": len(output), "tensors": tensor_items}
    return {"type": output.__class__.__name__, "repr": repr(output)[:500]}


def _select_device(device: str | None) -> torch.device:
    if device in {None, "cpu"}:
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    raise ValueError(f"Unsupported smoke-test device '{device}'.")


def _infer_input_kind(model_config: dict[str, Any] | None, input_kind: str | None) -> str:
    if input_kind and input_kind != "auto":
        return input_kind
    task = (model_config or {}).get("task")
    if task in {"masked-lm", "causal-lm"}:
        return "text"
    if task == "image-classification":
        return "image"
    return "tensor"


def _first_linear_in_features(model: torch.nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features
    return 4


def _prepare_text_inputs(tokenizer: object, device: torch.device) -> dict[str, torch.Tensor]:
    encoding = tokenizer("This is a pruning structural smoke test.", return_tensors="pt")
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    return {
        key: value.to(device)
        for key, value in encoding.items()
        if key in allowed and isinstance(value, torch.Tensor)
    }


def run_forward_smoke_test(
    model: torch.nn.Module,
    model_config: dict | None = None,
    tokenizer_or_processor: object | None = None,
    input_kind: str | None = None,
    device: str | None = None,
) -> ForwardSmokeResult:
    """Run a minimal forward pass and report whether it executes."""
    model_name = (model_config or {}).get("name") or model.__class__.__name__
    model_dir = (model_config or {}).get("model_dir")
    selected_kind = _infer_input_kind(model_config, input_kind)
    validation_id = _now_id(f"smoke__{model_name}".replace("/", "__").replace(" ", "_"))

    try:
        selected_device = _select_device(device)
        model.to(selected_device)
        model.eval()
        with torch.no_grad():
            if selected_kind == "text":
                if tokenizer_or_processor is None:
                    return ForwardSmokeResult(
                        validation_id=validation_id,
                        model_name=model_name,
                        model_dir=model_dir,
                        status="skipped",
                        input_kind=selected_kind,
                        error_message="Text smoke test requires a tokenizer.",
                    )
                output = model(**_prepare_text_inputs(tokenizer_or_processor, selected_device))
            elif selected_kind == "image":
                pixel_values = torch.zeros((1, 3, 224, 224), device=selected_device)
                output = model(pixel_values=pixel_values)
            elif selected_kind == "tensor":
                input_tensor = torch.zeros((1, _first_linear_in_features(model)), device=selected_device)
                output = model(input_tensor)
            else:
                return ForwardSmokeResult(
                    validation_id=validation_id,
                    model_name=model_name,
                    model_dir=model_dir,
                    status="skipped",
                    input_kind=selected_kind,
                    error_message=f"Unsupported input kind '{selected_kind}'.",
                )
        return ForwardSmokeResult(
            validation_id=validation_id,
            model_name=model_name,
            model_dir=model_dir,
            status="passed",
            input_kind=selected_kind,
            output_summary=summarize_model_output(output),
            metadata={"device": str(selected_device)},
        )
    except Exception as exc:  # noqa: BLE001 - smoke tests should return structured failures.
        return ForwardSmokeResult(
            validation_id=validation_id,
            model_name=model_name,
            model_dir=model_dir,
            status="failed",
            input_kind=selected_kind,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            metadata={"device": device or "cpu"},
        )


def forward_smoke_result_to_dict(result: ForwardSmokeResult) -> dict[str, Any]:
    return asdict(result)


def forward_smoke_result_to_markdown(result: ForwardSmokeResult) -> str:
    error = "_None._"
    if result.error_type or result.error_message:
        error = f"- Type: `{result.error_type}`\n- Message: `{result.error_message}`"
    return "\n".join(
        [
            f"# Forward Smoke Test: {result.validation_id}",
            "",
            "## Status",
            "",
            f"- `{result.status}`",
            "",
            "## Input Kind",
            "",
            f"- `{result.input_kind}`",
            "",
            "## Output Summary",
            "",
            "```json",
            str(result.output_summary),
            "```",
            "",
            "## Error",
            "",
            error,
            "",
            "## Caveats",
            "",
            "- Passing a forward smoke test does not prove accuracy preservation.",
            "- Failing a transformer smoke test after local pruning may indicate missing coupled repairs, not necessarily a low-level Linear pruning bug.",
            "- This milestone only repairs paired Linear dimensions.",
            "",
        ]
    )
