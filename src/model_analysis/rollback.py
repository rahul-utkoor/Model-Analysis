"""Rollback manifest generation for pruning artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.pruning_execution import PruningExecutionReport, pruning_execution_report_to_dict


def _report_dict(report: PruningExecutionReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, PruningExecutionReport):
        return pruning_execution_report_to_dict(report)
    return report


def create_rollback_manifest(
    execution_report: PruningExecutionReport | dict,
    source_model_dir: Path,
    output_model_dir: Path,
    path: Path,
) -> dict[str, Any]:
    report = _report_dict(execution_report)
    files_created = []
    if output_model_dir.exists():
        files_created = [str(item) for item in output_model_dir.rglob("*") if item.is_file()]
    manifest = {
        "execution_id": report.get("execution_id"),
        "model_name": report.get("model_name"),
        "source_model_dir": str(source_model_dir),
        "output_model_dir": str(output_model_dir),
        "files_created": files_created,
        "applied_modules": [record.get("module_name") for record in report.get("applied_records", [])],
        "how_to_rollback": [
            f"Delete output_model_dir: {output_model_dir}",
            f"Use original source_model_dir: {source_model_dir}",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(path),
    }
    return manifest


def write_rollback_manifest(manifest: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rollback_manifest_to_markdown(manifest: dict[str, Any]) -> str:
    files = "\n".join(f"- `{path}`" for path in manifest.get("files_created", [])) or "_None._"
    modules = "\n".join(f"- `{name}`" for name in manifest.get("applied_modules", [])) or "_None._"
    rollback = "\n".join(f"- {line}" for line in manifest.get("how_to_rollback", []))
    return "\n".join(
        [
            f"# Rollback Manifest: {manifest.get('execution_id')}",
            "",
            f"- Model: `{manifest.get('model_name')}`",
            f"- Source model dir: `{manifest.get('source_model_dir')}`",
            f"- Output model dir: `{manifest.get('output_model_dir')}`",
            f"- Timestamp: `{manifest.get('timestamp')}`",
            "",
            "## Applied Modules",
            "",
            modules,
            "",
            "## Files Created",
            "",
            files,
            "",
            "## How To Roll Back",
            "",
            rollback,
            "",
        ]
    )
