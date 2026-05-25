"""Readable text dump for pruning plan validation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.pruning_plan_validation import PruningPlanValidationSet, pruning_plan_validation_set_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def pruning_plan_validation_to_text(value: PruningPlanValidationSet | dict[str, Any]) -> str:
    data = pruning_plan_validation_set_to_dict(value) if isinstance(value, PruningPlanValidationSet) else value
    lines = [f'pruning_plan_validation @{_escape(data.get("model_name", "model"))} {{']
    for item in data.get("validations", []):
        lines.append(f'  validation "{_escape(item.get("candidate_region_name", item.get("plan_id", "")))} :: intermediate_dim" {{')
        lines.append(f'    status = {item.get("validation_status", "unknown")}')
        lines.append(f'    score = {item.get("validation_score", 0)}')
        lines.append(f'    plan_status = {item.get("plan_status", "unknown")}')
        lines.append(f'    candidate = "{_escape(item.get("candidate_region_name", ""))}"')
        if item.get("checks"):
            lines.append("    checks {")
            for check in item["checks"]:
                lines.append(f'      {check.get("check_type")} {check.get("status")}')
            lines.append("    }")
        evidence = item.get("evidence", {}).get("op_semantics_summary", {})
        if evidence:
            lines.append("    evidence {")
            if evidence.get("producer"):
                lines.append(f'      producer = "{_escape(evidence.get("producer", ""))}"')
            if evidence.get("bias"):
                lines.append(f'      bias = "{_escape(evidence.get("bias", ""))}"')
            if evidence.get("consumer"):
                lines.append(f'      consumer = "{_escape(evidence.get("consumer", ""))}"')
            for source in evidence.get("gelu", [])[:3]:
                lines.append(f'      gelu = "{_escape(source)}"')
            lines.append("    }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_pruning_plan_validation_text(value: PruningPlanValidationSet | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(pruning_plan_validation_to_text(value), encoding="utf-8")
