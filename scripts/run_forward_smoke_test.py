#!/usr/bin/env python
"""Run a standalone forward smoke test for an original or pruned model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from model_analysis.forward_validation import (
    forward_smoke_result_to_dict,
    forward_smoke_result_to_markdown,
    run_forward_smoke_test,
)
from model_analysis.hf_utils import load_model, load_tokenizer_or_processor
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal forward smoke test.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Configured model name or Hugging Face ID.")
    target.add_argument("--model-dir", help="Local Hugging Face model directory.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--input-kind", choices=["text", "image", "tensor", "auto"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_from_model_name(model_name: str):
    config = get_model_config(model_name)
    root = get_project_root()
    source_dir = root / config["local_dir"]
    if not source_dir.exists():
        raise FileNotFoundError(f"Local model missing. Run: python scripts/download_models.py --model {config['name']}")
    model = load_model(config, source=source_dir)
    try:
        tokenizer_or_processor = load_tokenizer_or_processor(config, source=source_dir)
    except Exception:
        tokenizer_or_processor = None
    return model, tokenizer_or_processor, {**config, "model_dir": str(source_dir)}, safe_model_name(config["hf_id"])


def _load_from_model_dir(model_dir: str):
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    model = AutoModel.from_pretrained(path)
    tokenizer_or_processor = None
    try:
        tokenizer_or_processor = AutoTokenizer.from_pretrained(path)
    except Exception:
        try:
            tokenizer_or_processor = AutoImageProcessor.from_pretrained(path)
        except Exception:
            tokenizer_or_processor = None
    name = path.name
    return model, tokenizer_or_processor, {"name": name, "hf_id": name, "task": "unknown", "model_dir": str(path)}, safe_model_name(name)


def main() -> int:
    args = parse_args()
    try:
        if args.model:
            model, tokenizer_or_processor, config, safe_name = _load_from_model_name(args.model)
        else:
            model, tokenizer_or_processor, config, safe_name = _load_from_model_dir(args.model_dir)
        input_kind = None if args.input_kind == "auto" else args.input_kind
        result = run_forward_smoke_test(
            model=model,
            model_config=config,
            tokenizer_or_processor=tokenizer_or_processor,
            input_kind=input_kind,
            device=args.device,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    root = get_project_root()
    stem = f"{safe_name}__{result.validation_id}"
    write_json(forward_smoke_result_to_dict(result), root / "reports" / "forward_smoke_tests" / f"{stem}.json")
    write_markdown(forward_smoke_result_to_markdown(result), root / "reports" / "forward_smoke_tests" / f"{stem}.md")

    if args.verbose:
        print(f"[smoke] {result.validation_id}")
        print(f"  status: {result.status}")
        print(f"  input_kind: {result.input_kind}")
        print(f"  output_summary: {result.output_summary}")
        if result.error_message:
            print(f"  error: {result.error_message}")

    return 0 if result.status in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
