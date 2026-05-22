"""MLIR-like textual dump for the pruning Dimension IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.dimension_ir import PruningIR, pruning_ir_to_dict
from model_analysis.paths import ensure_dir


def _ir_dict(ir: PruningIR | dict[str, Any]) -> dict[str, Any]:
    return pruning_ir_to_dict(ir) if isinstance(ir, PruningIR) else ir


def _quote(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _bool(value: bool) -> str:
    return "true" if value else "false"


def pruning_ir_to_text(ir: PruningIR | dict) -> str:
    """Render a deterministic MLIR-like textual pruning IR dump."""
    data = _ir_dict(ir)
    lines = [f"pruning.module @{data.get('model_name')} {{"]

    for index, dim in enumerate(sorted(data.get("dimension_variables", []), key=lambda item: item["var_id"])):
        alias = f"%d{index}"
        lines.append(
            f"  pruning.dim {alias} owner({_quote(dim.get('owner_name'))}) "
            f"type({_quote(dim.get('owner_type'))}) name({_quote(dim.get('dim_name'))}) "
            f"size({dim.get('size') if dim.get('size') is not None else '?'}) "
            f"role({_quote(dim.get('semantic_role'))}) prunable({_bool(bool(dim.get('prunable')))}) "
            f"confidence({_quote(dim.get('confidence'))})"
        )

    dim_alias = {
        dim["var_id"]: f"%d{index}"
        for index, dim in enumerate(sorted(data.get("dimension_variables", []), key=lambda item: item["var_id"]))
    }
    for index, idx in enumerate(sorted(data.get("index_variables", []), key=lambda item: item["index_var_id"])):
        dim_ref = dim_alias.get(idx.get("dimension_var_id"), _quote(idx.get("dimension_var_id")))
        lines.append(f"  pruning.index %i{index} : !pruning.indexset<{dim_ref}>")

    for index, constraint in enumerate(sorted(data.get("constraint_equations", []), key=lambda item: item["constraint_id"])):
        lhs = dim_alias.get(constraint.get("lhs"), _quote(constraint.get("lhs")))
        rhs = dim_alias.get(constraint.get("rhs"), _quote(constraint.get("rhs")))
        if constraint.get("blocking"):
            lines.append(f"  // blocked: {constraint.get('reason')}")
        if constraint.get("relation") == "unknown":
            lines.append(f"  // unresolved: {constraint.get('constraint_id')}")
        lines.append(
            f"  pruning.constraint %c{index} {constraint.get('relation')}({lhs}, {rhs}) "
            f"type({_quote(constraint.get('constraint_type'))}) direction({_quote(constraint.get('direction'))}) "
            f"blocking({_bool(bool(constraint.get('blocking')))}) confidence({_quote(constraint.get('confidence'))}) "
            f"reason({_quote(constraint.get('reason'))})"
        )

    for index, eq_class in enumerate(sorted(data.get("equivalence_classes", []), key=lambda item: item["class_id"])):
        members = ", ".join(dim_alias.get(member, _quote(member)) for member in eq_class.get("members", []))
        lines.append(
            f"  pruning.eq_class %e{index} members({members}) "
            f"type({_quote(eq_class.get('class_type'))}) "
            f"size({eq_class.get('size') if eq_class.get('size') is not None else '?'}) "
            f"confidence({_quote(eq_class.get('confidence'))})"
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_pruning_ir_text(ir: PruningIR | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(pruning_ir_to_text(ir), encoding="utf-8")
