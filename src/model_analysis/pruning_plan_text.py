"""Readable text dump for symbolic pruning plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.pruning_plan_synthesis import PruningPlanSet, pruning_plan_set_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def pruning_plan_set_to_text(value: PruningPlanSet | dict[str, Any]) -> str:
    data = pruning_plan_set_to_dict(value) if isinstance(value, PruningPlanSet) else value
    lines = [f'pruning_plans @{_escape(data.get("model_name", "model"))} {{']
    for plan in data.get("plans", []):
        index_set = plan.get("symbolic_index_set", {}).get("name", "I_unknown")
        lines.append(f'  plan "{_escape(plan.get("candidate_region_name", plan.get("plan_id", "")))} :: {plan.get("target_dimension", "unknown")}" {{')
        lines.append(f'    status = {plan.get("plan_status", "unknown")}')
        lines.append(f'    candidate = "{_escape(plan.get("candidate_region_name", ""))}"')
        lines.append(f'    score = {plan.get("rank_score", 0)}')
        lines.append(f'    confidence = {plan.get("confidence", "unknown")}')
        lines.append(f"    index_set = {index_set}")
        if plan.get("actions"):
            lines.append("    actions {")
            for action in plan["actions"]:
                if action.get("action_type") == "preserve_output":
                    continue
                lines.append(
                    f'      {action.get("action_type")} "{_escape(action.get("target_source_name", ""))}" '
                    f'axis={action.get("target_axis")} dim={action.get("dimension")} using {action.get("index_set", index_set)}'
                )
            lines.append("    }")
        if plan.get("propagation"):
            lines.append("    propagation {")
            for step in plan["propagation"]:
                lines.append(f'      {step.get("semantic_kind")} preserves {step.get("from_dimension")} using {step.get("index_mapping")}')
            lines.append("    }")
        if plan.get("preserved_dimensions"):
            lines.append("    preserve {")
            for dim in plan["preserved_dimensions"]:
                lines.append(f'      "{_escape(dim.get("location", ""))}" = {dim.get("dimension")}')
            lines.append("    }")
        if plan.get("forbidden_actions"):
            lines.append("    forbidden {")
            for item in plan["forbidden_actions"]:
                lines.append(f'      {item.get("action_type")} {item.get("dimension")} at "{_escape(item.get("location", ""))}"')
            lines.append("    }")
        if plan.get("validation_checks"):
            lines.append("    validation {")
            for check in plan["validation_checks"]:
                lines.append(f'      {check.get("check_type")} {check.get("status")}')
            lines.append("    }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_pruning_plan_text(value: PruningPlanSet | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(pruning_plan_set_to_text(value), encoding="utf-8")

