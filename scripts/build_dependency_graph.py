#!/usr/bin/env python
"""Build pruning dependency graphs from structural inventory reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.dependency_analyzer import analyze_dependency_graph, dependency_analysis_to_markdown
from model_analysis.dependency_graph import (
    augment_dependency_graph_with_onnx_summary,
    build_dependency_graph_from_torch_summary,
    dependency_graph_to_markdown,
    write_dependency_graph_json,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, load_model_registry
from model_analysis.reporting import write_json, write_markdown


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _should_write_json(format_arg: str) -> bool:
    return format_arg in {"json", "both"}


def _should_write_markdown(format_arg: str) -> bool:
    return format_arg in {"md", "both"}


def build_one(config: dict, format_arg: str, require_onnx: bool = False, torch_only: bool = False, verbose: bool = False) -> None:
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    structural_path = root / "reports" / "structural_inventory" / f"{safe_name}.json"
    onnx_path = root / "reports" / "onnx_graphs" / f"{safe_name}.json"

    if not structural_path.exists():
        raise FileNotFoundError(
            f"Structural inventory missing. Run: python scripts/generate_structural_inventory.py --model {config['name']}"
        )
    if require_onnx and not torch_only and not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX summary missing. Run: python scripts/generate_structural_inventory.py --model {config['name']}"
        )

    torch_summary = _load_json(structural_path)
    graph = build_dependency_graph_from_torch_summary(torch_summary)

    if not torch_only and onnx_path.exists():
        onnx_summary = _load_json(onnx_path)
        graph = augment_dependency_graph_with_onnx_summary(graph, onnx_summary)
    elif not torch_only:
        print(f"[skip] ONNX summary not found for {config['name']}; building PyTorch-only dependency graph.")

    analysis = analyze_dependency_graph(graph)

    graph_dir = root / "reports" / "dependency_graphs"
    summary_dir = root / "reports" / "dependency_summaries"
    if _should_write_json(format_arg):
        write_dependency_graph_json(graph, graph_dir / f"{safe_name}.json")
        write_json(analysis, summary_dir / f"{safe_name}.json")
    if _should_write_markdown(format_arg):
        write_markdown(dependency_graph_to_markdown(graph), graph_dir / f"{safe_name}.md")
        write_markdown(dependency_analysis_to_markdown(analysis), summary_dir / f"{safe_name}.md")

    if verbose:
        print(f"[graph] {config['name']}")
        print(f"  prunable units: {analysis['num_prunable_units']}")
        print(f"  dependency edges: {analysis['num_dependency_edges']}")
        print(f"  coupled groups: {analysis['num_coupled_groups']}")
        print(f"  independent units: {analysis['num_independent_units']}")
        print(f"  ambiguous units: {analysis['num_ambiguous_units']}")
    print(f"[ok] built dependency graph for {config['name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pruning dependency graphs from structural inventory reports.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--require-onnx", action="store_true", help="Fail if the ONNX graph summary JSON is missing.")
    parser.add_argument("--torch-only", action="store_true", help="Build from PyTorch structural inventory only.")
    parser.add_argument(
        "--format",
        choices=["json", "md", "both"],
        default="both",
        help="Report formats to write. Defaults to both.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print key graph statistics.")
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
            build_one(
                config,
                args.format,
                require_onnx=args.require_onnx,
                torch_only=args.torch_only,
                verbose=args.verbose,
            )
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed for {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
