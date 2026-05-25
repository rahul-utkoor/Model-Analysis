"""Markdown rendering for full-model static analysis reports."""

from __future__ import annotations

from typing import Any


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None._"
    selected = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    if limit and len(rows) > limit:
        lines.append("| " + " | ".join("..." if idx == 0 else "" for idx, _ in enumerate(columns)) + " |")
    return "\n".join(lines)


def _counts_table(counts: dict[str, Any], limit: int = 20) -> str:
    rows = [
        {"item": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[:limit]
    ]
    return _table(rows, ["item", "count"])


def _classification(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("classification", {})


def model_report_to_markdown(report: dict[str, Any]) -> str:
    model = report.get("model_name", "model")
    summary = report.get("model_summary", {})
    lines = [
        f"# Full Static Analysis Report: {model}",
        "",
        "## 1. Purpose",
        "",
        "This is a compiler-style static pruning-analysis report. It summarizes existing TensorIR, op semantics, region semantics, ranking, symbolic plans, validation, and layer subgraph evidence.",
        "",
        "This report is static analysis/reporting/visualization only. ONNX subgraphs are evidence artifacts, not standalone full-model analysis sources.",
        "",
        "## 2. Available artifacts",
        "",
    ]
    rows = []
    for key, value in report.get("available_artifacts", {}).items():
        rows.append({"artifact": key, "status": "present" if value.get("present") else "missing", "path": value.get("path", "")})
    lines.extend([_table(rows, ["artifact", "status", "path"]), "", "## 3. Pipeline summary", ""])
    ranking = summary.get("ranking", {})
    plans = summary.get("plans", {})
    validations = summary.get("plan_validation", {})
    lines.extend(
        [
            f"- Layers generated: `{summary.get('layers_generated', 0)}`",
            f"- Layer subgraphs: `{summary.get('total_subgraphs', 0)}`",
            f"- ONNX exported/skipped/failed: `{summary.get('onnx_exported', 0)}` / `{summary.get('onnx_skipped', 0)}` / `{summary.get('onnx_failed', 0)}`",
            f"- Safe/constrained/blocked/auxiliary/unknown subgraphs: `{summary.get('safe', 0)}` / `{summary.get('constrained', 0)}` / `{summary.get('blocked', 0)}` / `{summary.get('auxiliary', 0)}` / `{summary.get('unknown', 0)}`",
            f"- Ranking safe/constrained/blocked/auxiliary/unknown: `{ranking.get('safe', 0)}` / `{ranking.get('constrained', 0)}` / `{ranking.get('blocked', 0)}` / `{ranking.get('auxiliary', 0)}` / `{ranking.get('unknown', 0)}`",
            f"- Plans ready/incomplete/blocked/unknown: `{plans.get('ready_symbolic', 0)}` / `{plans.get('incomplete', 0)}` / `{plans.get('blocked', 0)}` / `{plans.get('unknown', 0)}`",
            f"- Plan validations valid/warning/invalid/unknown: `{validations.get('valid', 0)}` / `{validations.get('warning', 0)}` / `{validations.get('invalid', 0)}` / `{validations.get('unknown', 0)}`",
            "",
            "### Op Semantic Counts",
            "",
            _counts_table(summary.get("op_semantic_counts", {})),
            "",
            "### Region Semantic Category Counts",
            "",
            _counts_table(summary.get("region_semantic_counts", {})),
            "",
            "## 4. Layer summary table",
            "",
        ]
    )
    lines.append(
        _table(
            report.get("layers", []),
            ["layer_index", "total_subgraphs", "onnx_exported", "safe", "constrained", "blocked", "auxiliary", "unknown", "valid_plan_subgraphs"],
        )
    )
    lines.extend(["", "## 5. Safe pruning opportunities", ""])
    lines.append(_table(report.get("safe_opportunities", []), ["layer", "region_name", "semantic_category", "target_dimension", "plan_id", "validation_status"], limit=100))
    lines.extend(["", "## 6. Constrained opportunities", ""])
    lines.append(_table(report.get("constrained_opportunities", []), ["layer", "region_name", "semantic_category", "candidate_kind", "target_dimension", "blockers"], limit=100))
    lines.extend(["", "## 7. Blocked structures", ""])
    lines.append(_table(report.get("blocked_structures", []), ["layer", "region_name", "semantic_category", "candidate_kind", "blockers"], limit=100))
    lines.extend(["", "## 8. Auxiliary structures", ""])
    lines.append(_table(report.get("auxiliary_structures", []), ["layer", "region_name", "semantic_category", "candidate_kind"], limit=60))
    lines.extend(["", "## 9. Per-layer links", ""])
    for layer in report.get("layers", []):
        lines.append(f"- [Layer {layer.get('layer_index')}]({layer.get('layer_report_path')})")
    lines.extend(
        [
            "",
            "## 10. Research conclusions",
            "",
            "- FFN intermediate_dim pruning is the clean safe opportunity when a valid symbolic plan is present.",
            "- Attention projection pruning is constrained by head-axis mapping evidence.",
            "- Attention score/context contractions are blocked as non-parameterized dataflow contractions.",
            "- Residual and LayerNorm hidden_dim paths are protected by conservative semantics.",
            "",
        ]
    )
    return "\n".join(lines)


def layer_report_to_markdown(layer_data: dict[str, Any]) -> str:
    pack = layer_data.get("pack", {})
    summary = layer_data.get("summary", {})
    model = pack.get("model_name", "model")
    layer = pack.get("layer_index", summary.get("layer_index", 0))
    rows = []
    for item in pack.get("subgraphs", []):
        cls = _classification(item)
        rows.append(
            {
                "#": item.get("ordinal"),
                "Abstract node": item.get("display_name"),
                "Semantic category": item.get("semantic_category"),
                "Primitive ops": len(item.get("primitive_ops", [])),
                "Class": cls.get("pruning_class"),
                "Plan": cls.get("plan_status"),
                "Validation": cls.get("validation_status"),
                "ONNX": item.get("onnx_export", {}).get("status"),
            }
        )
    lines = [
        f"# Layer {layer} Static Analysis Report: {model}",
        "",
        "## Summary",
        "",
        f"- Total subgraphs: `{summary.get('total_subgraphs', 0)}`",
        f"- ONNX exported/skipped/failed: `{summary.get('onnx_exported', 0)}` / `{summary.get('onnx_skipped', 0)}` / `{summary.get('onnx_failed', 0)}`",
        f"- Safe/constrained/blocked/auxiliary/unknown: `{summary.get('safe', 0)}` / `{summary.get('constrained', 0)}` / `{summary.get('blocked', 0)}` / `{summary.get('auxiliary', 0)}` / `{summary.get('unknown', 0)}`",
        f"- Valid plan subgraphs: `{summary.get('valid_plan_subgraphs', 0)}`",
        "",
        "## Ordered subgraph table",
        "",
        _table(rows, ["#", "Abstract node", "Semantic category", "Primitive ops", "Class", "Plan", "Validation", "ONNX"]),
        "",
        "## Stage explanation",
        "",
        "- Q/K/V projections are learned attention projections, constrained by head-axis mapping proof.",
        "- Attention internals include blocked contractions and auxiliary mask/probability flow.",
        "- Attention output/residual structures protect hidden_dim and residual branch agreement.",
        "- Feed-forward block is the safe validated intermediate_dim pruning unit when a plan validates.",
        "- FFN components provide local evidence for producer-output pruning, GELU propagation, and consumer-input repair.",
        "- Residual/LayerNorm paths are protected by default.",
        "",
        "## Per-subgraph details",
        "",
    ]
    for item in pack.get("subgraphs", []):
        cls = _classification(item)
        lines.extend(
            [
                f"### {item.get('ordinal')}. {item.get('display_name')}",
                "",
                f"- Semantic category: `{item.get('semantic_category')}`",
                f"- Class: `{cls.get('pruning_class')}`",
                f"- Plan: `{cls.get('plan_status')}`",
                f"- Validation: `{cls.get('validation_status')}`",
                f"- Why no plan: {item.get('why_no_plan') or 'not applicable'}",
                f"- Verdict: {item.get('verdict')}",
                "",
            ]
        )
    return "\n".join(lines)


def subgraph_explanation_to_markdown(item: dict[str, Any]) -> str:
    cls = _classification(item)
    lines = [
        f"# {item.get('display_name')}",
        "",
        "## What this subgraph is",
        "",
        item.get("verdict") or item.get("explanation", ""),
        "",
        "## Primitive ops",
        "",
        _table(item.get("primitive_ops", []), ["topological_index", "source_name", "op_type"], limit=40),
        "",
        "## Op semantics",
        "",
        _table(item.get("local_op_semantics", []), ["source_name", "semantic_kind", "semantic_category", "parameterized", "direct_pruning"], limit=40),
        "",
        "## Ranking",
        "",
        _table(item.get("local_ranking", []), ["candidate_kind", "pruning_class", "rank_score", "confidence", "target_dimension", "reason"], limit=20),
        "",
        "## Plan",
        "",
        _table(item.get("local_plans", []), ["plan_kind", "plan_status", "target_dimension", "symbolic_index_set"], limit=20),
        "",
        "## Validation",
        "",
        _table(item.get("local_validations", []), ["validation_status", "validation_score", "failed_checks", "warning_checks"], limit=20),
        "",
        "## Verdict",
        "",
        f"- Class: `{cls.get('pruning_class')}`",
        f"- Plan: `{cls.get('plan_status')}`",
        f"- Validation: `{cls.get('validation_status')}`",
        f"- Why no plan: {item.get('why_no_plan') or 'not applicable'}",
        "",
    ]
    return "\n".join(lines)


def compact_model_snapshot_to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("model_summary", {})
    return "\n".join(
        [
            f"# Compact Full-Model Snapshot: {report.get('model_name')}",
            "",
            f"- Layers generated: `{summary.get('layers_generated', 0)}`",
            f"- Total subgraphs: `{summary.get('total_subgraphs', 0)}`",
            f"- ONNX exported/skipped/failed: `{summary.get('onnx_exported', 0)}` / `{summary.get('onnx_skipped', 0)}` / `{summary.get('onnx_failed', 0)}`",
            f"- Safe/constrained/blocked/auxiliary/unknown: `{summary.get('safe', 0)}` / `{summary.get('constrained', 0)}` / `{summary.get('blocked', 0)}` / `{summary.get('auxiliary', 0)}` / `{summary.get('unknown', 0)}`",
            f"- Valid FFN plans: `{summary.get('plan_validation', {}).get('valid', 0)}`",
            "",
            "This is static reporting/visualization only.",
            "",
        ]
    )


def summary_to_markdown(name: str, data: Any) -> str:
    if isinstance(data, dict):
        if all(isinstance(value, (int, str, float, bool)) or value is None for value in data.values()):
            body = _table([{"item": key, "value": value} for key, value in data.items()], ["item", "value"])
        else:
            rows = []
            for key, value in data.items():
                if isinstance(value, list):
                    rows.append({"section": key, "count": len(value)})
                else:
                    rows.append({"section": key, "count": value})
            body = _table(rows, ["section", "count"])
    else:
        body = str(data)
    return "\n".join([f"# {name.replace('_', ' ').title()}", "", body, "", "Static analysis/reporting only.", ""])

