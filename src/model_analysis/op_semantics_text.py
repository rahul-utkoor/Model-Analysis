"""Readable text dump for Op Semantics IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.op_semantics import OpSemanticsIR, op_semantics_ir_to_dict
from model_analysis.paths import ensure_dir


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def op_semantics_to_text(value: OpSemanticsIR | dict[str, Any]) -> str:
    data = op_semantics_ir_to_dict(value) if isinstance(value, OpSemanticsIR) else value
    lines = [f'op_semantics @{_escape(data.get("model_name", "model"))} {{']
    for op in data.get("ops", []):
        effect = op.get("pruning_effect", {})
        lines.append(f'  op "{_escape(op.get("source_name", op.get("op_id", "")))}" {{')
        lines.append(f'    op_type = {_escape(op.get("op_type", ""))}')
        lines.append(f'    semantic_kind = {op.get("semantic_kind", "unknown")}')
        lines.append(f'    category = {op.get("semantic_category", "unknown")}')
        lines.append(f'    parameterized = {str(op.get("parameterized", "unknown")).lower()}')
        lines.append(f'    index_behavior = {op.get("index_behavior", "unknown")}')
        if op.get("dimension_roles"):
            lines.append("    roles {")
            for key, role in op["dimension_roles"].items():
                lines.append(f"      {key} = {role}")
            lines.append("    }")
        lines.append("    pruning_effect {")
        lines.append(f'      direct_pruning = {effect.get("direct_pruning", "unknown")}')
        for repair in effect.get("required_repairs", []):
            lines.append(f"      repair = {repair}")
        for blocker in effect.get("blockers", []):
            lines.append(f"      blocker = {blocker}")
        lines.append(f'      reason = "{_escape(effect.get("reason", ""))}"')
        lines.append("    }")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def write_op_semantics_text(value: OpSemanticsIR | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(op_semantics_to_text(value), encoding="utf-8")

