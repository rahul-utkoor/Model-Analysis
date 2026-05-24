"""Readable text dump for pruning opportunity rankings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.pruning_opportunity_ranking import PruningOpportunityRanking, pruning_opportunity_ranking_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def pruning_opportunity_ranking_to_text(value: PruningOpportunityRanking | dict[str, Any]) -> str:
    data = pruning_opportunity_ranking_to_dict(value) if isinstance(value, PruningOpportunityRanking) else value
    lines = [f'pruning_opportunity_ranking @{_escape(data.get("model_name", "model"))} {{']
    for item in data.get("candidates", []):
        lines.append(f'  candidate "{_escape(item.get("region_name", item.get("candidate_id", "")))} :: {item.get("target_dimension", "unknown")}" {{')
        lines.append(f'    class = {item.get("pruning_class", "unknown")}')
        lines.append(f'    score = {item.get("rank_score", 0)}')
        lines.append(f'    confidence = {item.get("confidence", "unknown")}')
        lines.append(f'    kind = {item.get("candidate_kind", "unknown")}')
        lines.append(f'    target_dimension = {item.get("target_dimension", "unknown")}')
        repairs = [repair.get("obligation_type", "") for repair in item.get("required_repairs", [])]
        blockers = [blocker.get("blocker_type", "") for blocker in item.get("blockers", [])]
        lines.append("    repairs = [" + ", ".join(repair for repair in repairs if repair) + "]")
        lines.append("    blockers = [" + ", ".join(blocker for blocker in blockers if blocker) + "]")
        lines.append(f'    reason = "{_escape(item.get("reason", ""))}"')
        if item.get("op_semantics_evidence"):
            lines.append("    op_evidence {")
            for op in item["op_semantics_evidence"][:20]:
                lines.append(f'      {op.get("semantic_kind", "unknown")} "{_escape(op.get("source_name", ""))}"')
            lines.append("    }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_pruning_opportunity_ranking_text(value: PruningOpportunityRanking | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(pruning_opportunity_ranking_to_text(value), encoding="utf-8")

