#!/usr/bin/env python
"""Export locally downloaded models to ONNX."""

from __future__ import annotations

import argparse
import sys

from model_analysis.onnx_export import export_model_to_onnx
from model_analysis.registry import get_model_config, load_model_registry


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export configured local models to ONNX.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version. Defaults to 17.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0

    try:
        configs = select_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    for config in configs:
        try:
            print(f"[export] {config['name']} -> {config['onnx_dir']}/model.onnx")
            output_path = export_model_to_onnx(config, opset=args.opset)
            print(f"[ok] exported and validated {output_path}")
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed to export {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
