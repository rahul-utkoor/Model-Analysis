"""Run standalone DFA pruning propagation teaching examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.dfa_pruning_propagation.examples import get_example
from experimental.dfa_pruning_propagation.report import render_json, render_markdown, render_text
from experimental.dfa_pruning_propagation.worklist import analyze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["ffn", "attention-value", "attention-qk", "residual"], required=True)
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example = get_example(args.example)
    result = analyze(example.graph, example.seed_facts)
    if args.format == "markdown":
        text = render_markdown(example, result)
    elif args.format == "json":
        text = render_json(example, result)
    else:
        text = render_text(example, result, show_trace=args.show_trace)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        if args.verbose:
            print(f"[dfa-propagation] wrote {output}")
    else:
        print(text)
    if args.verbose:
        print(f"[dfa-propagation] fixed_point={result.summary['reached_fixed_point']} propagated={result.summary['num_propagated']} blocked={result.summary['num_blocked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
