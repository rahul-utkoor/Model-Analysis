"""Text renderers for static deadbranch propagation reports."""

from __future__ import annotations

from typing import Any

from model_analysis.deadbranch_propagation import deadbranch_report_to_dict


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def deadbranch_report_to_markdown(value: dict[str, Any]) -> str:
    data = deadbranch_report_to_dict(value)
    summary = data.get("summary", {})
    pairs = [
        {
            "layer": pair.get("layer_index"),
            "pair kind": pair.get("pair_kind"),
            "producer": pair.get("producer_op_name"),
            "consumer": pair.get("consumer_op_name"),
            "status": pair.get("status"),
            "required mapping": pair.get("required_mapping"),
            "explanation": pair.get("explanation"),
        }
        for pair in data.get("pairs", [])
    ]
    blocked = [
        {
            "layer": pair.get("layer_index"),
            "Q/K projection": pair.get("producer_region_name"),
            "consumer": pair.get("consumer_region_name"),
            "blocker": pair.get("blocker_type"),
            "explanation": pair.get("explanation"),
        }
        for pair in data.get("blocked_pairs", [])
    ]
    return "\n".join(
        [
            f"# Deadbranch Propagation Analysis: {data.get('model_name')}",
            "",
            "## Executive Summary",
            "",
            f"- Total propagation pairs: `{summary.get('total_pairs', 0)}`",
            f"- FFN pairs: `{summary.get('ffn_pairs', 0)}`",
            f"- Attention value-path pairs: `{summary.get('attention_value_pairs', 0)}`",
            f"- Blocked Q/K pairs: `{summary.get('query_key_blocked_pairs', 0)}`",
            f"- SparseGPT alignment: `{summary.get('sparsegpt_alignment_status', 'unknown')}`",
            "",
            "## Why SparseGPT 2:4 Does Not Trigger Deadbranch Propagation",
            "",
            "SparseGPT 2:4 does not expose dead channels: fine-grained sparse-weight pruning preserves tensor shapes and channel liveness.",
            "",
            "## Why Structural Channel Pruning Does Trigger It",
            "",
            "Channel pruning exposes deadness because an exact zero/dead consumer input column makes the corresponding producer output channel removable when the index mapping is proven.",
            "",
            "## Propagation Pairs",
            "",
            _table(pairs, ["layer", "pair kind", "producer", "consumer", "status", "required mapping", "explanation"]),
            "",
            "## Blocked Attention Paths",
            "",
            "QK^T blocks Q/K simple propagation because score contraction mixes projected channels.",
            "",
            _table(blocked, ["layer", "Q/K projection", "consumer", "blocker", "explanation"]),
            "",
            "## SparseGPT Alignment",
            "",
            f"- Expected observed pairs: `{summary.get('expected_sparsegpt_pairs', 0)}`",
            f"- Predicted pairs: `{summary.get('total_pairs', 0)}`",
            f"- Status: `{summary.get('sparsegpt_alignment_status', 'unknown')}`",
            "",
            "The attention value rule is `v_proj -> out_proj`: dead output-projection input channels can propagate backward through the context/value path when the mapping is proven.",
            "",
            "This is static analysis/reporting only. It does not execute pruning or modify models.",
            "",
        ]
    )


def deadbranch_report_to_text(value: dict[str, Any]) -> str:
    data = deadbranch_report_to_dict(value)
    lines = [f"deadbranch_propagation @{data.get('model_name')} {{"]
    for pair in data.get("pairs", []):
        lines.extend(
            [
                f'  pair "{pair.get("pair_id")}" {{',
                f"    kind = {pair.get('pair_kind')}",
                f"    layer = {pair.get('layer_index')}",
                f"    producer = {pair.get('producer_op_name')}",
                f"    consumer = {pair.get('consumer_op_name')}",
                f"    status = {pair.get('status')}",
                f"    mapping = {pair.get('required_mapping')}:{pair.get('mapping_status')}",
                "  }",
            ]
        )
    for pair in data.get("blocked_pairs", []):
        lines.extend(
            [
                f'  blocked "{pair.get("pair_id")}" {{',
                f"    kind = {pair.get('pair_kind')}",
                f"    blocker = {pair.get('blocker_type')}",
                "  }",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines)
