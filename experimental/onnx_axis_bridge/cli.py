"""Analyze a selected ONNX subgraph through the experimental axis bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.onnx_axis_bridge.bridge_runner import analyze_onnx_subgraph
from experimental.onnx_axis_bridge.report import render_json, render_markdown, render_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Path to one local ONNX subgraph artifact.")
    parser.add_argument("--hint", choices=["auto", "ffn", "attention-context", "qk-score", "attention-value-path", "residual", "layernorm"], default="auto")
    parser.add_argument("--format", choices=["markdown", "json", "text"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--show-summary", action="store_true")
    parser.add_argument("--show-hints", action="store_true")
    parser.add_argument("--show-axis", action="store_true")
    parser.add_argument("--show-dfa", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze_onnx_subgraph(args.onnx, requested_hint=args.hint)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    if args.format == "markdown":
        text = render_markdown(result)
    elif args.format == "json":
        text = render_json(result)
    else:
        show_all = args.show_all
        text = render_text(
            result,
            show_summary=args.show_summary or show_all,
            show_hints=args.show_hints or show_all,
            show_axis=args.show_axis or show_all,
            show_dfa=args.show_dfa or show_all,
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        if args.verbose:
            print(f"[onnx-axis-bridge] wrote {output}")
    else:
        print(text)
    if args.verbose:
        print(
            f"[onnx-axis-bridge] nodes={result.summary['num_nodes']} "
            f"hints={len(result.summary['recognized_hints'])} "
            f"lowered={result.summary['lowered_regions']} "
            f"dfa={result.summary['dfa_propagation_results']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
