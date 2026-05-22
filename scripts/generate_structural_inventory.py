#!/usr/bin/env python
"""Generate PyTorch and optional ONNX structural inventory reports."""

from __future__ import annotations

import argparse
import sys

from model_analysis.hf_utils import load_model
from model_analysis.onnx_graph_analysis import summarize_onnx_graph
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, load_model_registry
from model_analysis.reporting import (
    onnx_summary_to_markdown,
    pruning_hints_to_markdown,
    structural_inventory_to_markdown,
    write_json,
    write_markdown,
)
from model_analysis.structural_inventory import summarize_torch_model


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def _should_write_json(format_arg: str) -> bool:
    return format_arg in {"json", "both"}


def _should_write_markdown(format_arg: str) -> bool:
    return format_arg in {"md", "both"}


def generate_one(config: dict, format_arg: str, require_onnx: bool = False) -> None:
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    source_dir = root / config["local_dir"]
    onnx_path = root / config["onnx_dir"] / "model.onnx"

    if not source_dir.exists() or not (source_dir / "config.json").exists():
        raise FileNotFoundError(f"Model not found locally. Run: python scripts/download_models.py --model {config['name']}")

    if require_onnx and not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Run: python scripts/export_to_onnx.py --model {config['name']}"
        )

    print(f"[load] {config['name']} from {source_dir}")
    model = load_model(config, source=source_dir)
    model.eval()

    torch_summary = summarize_torch_model(model, config["name"], config)
    structural_dir = root / "reports" / "structural_inventory"

    if _should_write_json(format_arg):
        write_json(torch_summary, structural_dir / f"{safe_name}.json")
    if _should_write_markdown(format_arg):
        write_markdown(structural_inventory_to_markdown(torch_summary), structural_dir / f"{safe_name}.md")

    onnx_summary = None
    if onnx_path.exists():
        print(f"[onnx] summarizing {onnx_path}")
        onnx_summary = summarize_onnx_graph(onnx_path, config["name"], config)
        onnx_dir = root / "reports" / "onnx_graphs"
        if _should_write_json(format_arg):
            write_json(onnx_summary, onnx_dir / f"{safe_name}.json")
        if _should_write_markdown(format_arg):
            write_markdown(onnx_summary_to_markdown(onnx_summary), onnx_dir / f"{safe_name}.md")
    else:
        print(f"[skip] ONNX file not found for {config['name']}; pruning hints will use PyTorch evidence only.")

    hints_dir = root / "reports" / "pruning_hints"
    write_markdown(pruning_hints_to_markdown(torch_summary, onnx_summary), hints_dir / f"{safe_name}.md")
    print(f"[ok] generated structural inventory for {config['name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structural inventory reports.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--require-onnx", action="store_true", help="Fail if the matching ONNX export is missing.")
    parser.add_argument(
        "--format",
        choices=["json", "md", "both"],
        default="both",
        help="Report formats to write for structural and ONNX summaries. Defaults to both.",
    )
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
            generate_one(config, args.format, require_onnx=args.require_onnx)
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed for {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
