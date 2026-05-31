"""Status records for the static pruning-analysis pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


STAGE_NAMES = [
    "tensor_ir",
    "op_semantics",
    "structural_region_tree",
    "region_dimension_ir",
    "region_pruning_semantics",
    "pruning_opportunity_ranking",
    "pruning_plan_synthesis",
    "pruning_plan_validation",
    "deadbranch_propagation",
    "layer_subgraph_validation",
    "full_model_report",
    "cross_model_report",
]


@dataclass
class StageStatus:
    stage_name: str
    status: str
    required_inputs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    command_hint: str = ""
    error: str = ""
    duration_seconds: float | None = None
    notes: str = ""


@dataclass
class StaticPipelineModelStatus:
    model_name: str
    generated_at: str
    model_available: bool
    configured_model: dict[str, Any]
    stages: list[StageStatus] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    final_status: str = "skipped"
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def stage_to_dict(stage: StageStatus | dict[str, Any]) -> dict[str, Any]:
    return asdict(stage) if isinstance(stage, StageStatus) else stage


def status_to_dict(status: StaticPipelineModelStatus | dict[str, Any]) -> dict[str, Any]:
    if isinstance(status, dict):
        return status
    data = asdict(status)
    data["stages"] = [stage_to_dict(stage) for stage in status.stages]
    return data


def summarize_model_status(status: StaticPipelineModelStatus | dict[str, Any]) -> dict[str, Any]:
    data = status_to_dict(status)
    stages = [stage_to_dict(stage) for stage in data.get("stages", [])]
    counts = Counter(stage.get("status", "unknown") for stage in stages)
    missing = sorted(
        {
            missing
            for stage in stages
            for missing in stage.get("missing_inputs", [])
            if missing
        }
    )
    produced_reports = sorted(
        output
        for stage in stages
        for output in stage.get("outputs", [])
        if output and output.startswith("reports/")
    )
    produced_artifacts = sorted(
        output
        for stage in stages
        for output in stage.get("outputs", [])
        if output and output.startswith("artifacts/")
    )
    return {
        "completed_stages": counts.get("present_existing", 0) + counts.get("built", 0),
        "skipped_stages": counts.get("skipped", 0),
        "failed_stages": counts.get("failed", 0),
        "not_applicable_stages": counts.get("not_applicable", 0),
        "missing_artifacts": missing,
        "produced_reports": produced_reports,
        "produced_artifacts": produced_artifacts,
        "stage_status_counts": dict(sorted(counts.items())),
    }


def final_status_from_stages(stages: list[StageStatus | dict[str, Any]]) -> str:
    stage_dicts = [stage_to_dict(stage) for stage in stages]
    if any(stage.get("status") == "failed" for stage in stage_dicts):
        return "failed"
    core = [
        "tensor_ir",
        "op_semantics",
        "structural_region_tree",
        "region_dimension_ir",
        "region_pruning_semantics",
        "pruning_opportunity_ranking",
        "layer_subgraph_validation",
        "full_model_report",
    ]
    status_by_stage = {stage.get("stage_name"): stage.get("status") for stage in stage_dicts}
    complete_like = {"present_existing", "built", "not_applicable"}
    if all(status_by_stage.get(stage) in complete_like for stage in core):
        return "complete"
    if any(stage.get("status") in {"present_existing", "built"} for stage in stage_dicts):
        return "partial"
    return "skipped"


def make_model_status(
    *,
    model_name: str,
    configured_model: dict[str, Any],
    stages: list[StageStatus],
    artifacts: dict[str, Any],
    notes: list[str] | None = None,
) -> StaticPipelineModelStatus:
    status = StaticPipelineModelStatus(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_available=bool(configured_model),
        configured_model=configured_model,
        stages=stages,
        artifacts=artifacts,
        notes=notes or [],
    )
    status.final_status = final_status_from_stages(stages)
    status.summary = summarize_model_status(status)
    return status


def write_model_status(status: StaticPipelineModelStatus | dict[str, Any], json_path: Path, md_path: Path) -> None:
    data = status_to_dict(status)
    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    md_path.write_text(status_to_markdown(data), encoding="utf-8")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def status_to_markdown(status: StaticPipelineModelStatus | dict[str, Any]) -> str:
    data = status_to_dict(status)
    stages = [stage_to_dict(stage) for stage in data.get("stages", [])]
    rows = [
        {
            "stage": stage.get("stage_name"),
            "status": stage.get("status"),
            "missing": ", ".join(stage.get("missing_inputs", [])),
            "outputs": ", ".join(stage.get("outputs", [])[:2]),
            "hint": stage.get("command_hint", ""),
        }
        for stage in stages
    ]
    lines = [
        f"# Static Pipeline Status: {data.get('model_name')}",
        "",
        f"- Final status: `{data.get('final_status')}`",
        f"- Completed stages: `{data.get('summary', {}).get('completed_stages', 0)}`",
        f"- Skipped stages: `{data.get('summary', {}).get('skipped_stages', 0)}`",
        f"- Failed stages: `{data.get('summary', {}).get('failed_stages', 0)}`",
        "",
        "## Stage Status",
        "",
        _table(rows, ["stage", "status", "missing", "outputs", "hint"]),
        "",
        "## Missing Artifacts",
        "",
    ]
    missing = data.get("summary", {}).get("missing_artifacts", [])
    if missing:
        lines.extend(f"- `{item}`" for item in missing)
    else:
        lines.append("_None._")
    lines.extend(["", "This is static analysis/reporting only.", ""])
    return "\n".join(lines)
