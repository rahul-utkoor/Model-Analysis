"""Run loop/access-evidence-to-DFA bridge teaching examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.examples import get_example
from experimental.pruning_analysis_bridge.report import render_json, render_markdown, render_text


EXAMPLES = [
    "ffn-from-access",
    "attention-value-from-access",
    "qk-blocked-from-access",
    "residual-from-access",
    "layernorm-from-access",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES, required=True)
    parser.add_argument("--format", choices=["markdown", "json", "text"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--show-axis-evidence", action="store_true")
    parser.add_argument("--show-dfa-trace", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example = get_example(args.example)
    result = run_bridge_analysis(
        example.region_spec,
        example.seed_policy,
        example_name=example.example_name,
        interpretation=example.interpretation,
    )
    if args.format == "markdown":
        text = render_markdown(result)
    elif args.format == "json":
        text = render_json(result)
    else:
        text = render_text(
            result,
            show_axis_evidence=args.show_axis_evidence or args.show_all,
            show_dfa_trace=args.show_dfa_trace or args.show_all,
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        if args.verbose:
            print(f"[pruning-bridge] wrote {output}")
    else:
        print(text)
    if args.verbose:
        print(
            f"[pruning-bridge] pattern={result.summary['selected_pattern']} "
            f"fixed_point={result.summary['reached_fixed_point']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
