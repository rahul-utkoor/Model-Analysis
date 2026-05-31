"""Run standalone loop/access axis-transfer analysis examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.examples import get_example
from experimental.axis_transfer_analysis.pattern_recognition import recognize_patterns
from experimental.axis_transfer_analysis.report import render_json, render_markdown, render_text


EXAMPLES = ["activation", "ffn", "qk-score", "attention-context", "attention-value-path", "residual", "layernorm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES, required=True)
    parser.add_argument("--format", choices=["markdown", "json", "text"], default="text")
    parser.add_argument("--output")
    parser.add_argument("--show-relations", action="store_true")
    parser.add_argument("--show-patterns", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example = get_example(args.example)
    summary = analyze_region(example.region)
    patterns = recognize_patterns(example.region, summary)
    if args.format == "markdown":
        text = render_markdown(example, summary, patterns)
    elif args.format == "json":
        text = render_json(example, summary, patterns)
    else:
        text = render_text(example, summary, patterns, show_relations=args.show_relations, show_patterns=args.show_patterns)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        if args.verbose:
            print(f"[axis-transfer] wrote {output}")
    else:
        print(text)
    if args.verbose:
        print(f"[axis-transfer] ops={len(summary.op_summaries)} patterns={len(patterns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
