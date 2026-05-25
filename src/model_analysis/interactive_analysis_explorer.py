"""Read-only helpers for the interactive static analysis explorer."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ModelRef:
    model_name: str
    safe_name: str
    model_dir: Path
    index_path: Path


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def discover_models(report_root: Path) -> list[ModelRef]:
    if not report_root.exists():
        return []
    models: list[ModelRef] = []
    for item in sorted(report_root.iterdir(), key=lambda path: path.name):
        if not item.is_dir() or item.name == "cross_model":
            continue
        index = item / "index.json"
        if not index.exists():
            continue
        data = load_json(index)
        models.append(ModelRef(model_name=data.get("model_name") or item.name, safe_name=item.name, model_dir=item, index_path=index))
    return models


def find_model(models: Iterable[ModelRef], value: str) -> ModelRef | None:
    needle = value.strip().lower()
    for model in models:
        if needle in {model.model_name.lower(), model.safe_name.lower()}:
            return model
    for model in models:
        if needle and (needle in model.model_name.lower() or needle in model.safe_name.lower()):
            return model
    return None


def load_model_summary(model: ModelRef) -> dict[str, Any]:
    return load_json(model.index_path)


def discover_layers(model_dir: Path) -> list[dict[str, Any]]:
    layer_root = model_dir / "layers"
    if not layer_root.exists():
        return []
    layers: list[dict[str, Any]] = []
    for layer_dir in sorted(layer_root.glob("layer_*"), key=lambda path: _layer_sort_key(path.name)):
        index = layer_dir / "index.json"
        if not index.exists():
            continue
        data = load_json(index)
        summary = data.get("summary", {})
        layer_index = summary.get("layer_index")
        if layer_index is None:
            layer_index = _layer_sort_key(layer_dir.name)
        layers.append({"layer_index": int(layer_index), "layer_dir": layer_dir, "index_path": index, "summary": summary, "data": data})
    return layers


def load_layer_summary(layer_dir: Path) -> dict[str, Any]:
    return load_json(layer_dir / "index.json").get("summary", {})


def discover_subgraphs(model_dir: Path, layer_index: int, fallback_layer_pack_root: Path | None = None) -> list[dict[str, Any]]:
    layer_dir = model_dir / "layers" / f"layer_{layer_index}" / "subgraphs"
    subgraphs = _load_subgraph_dir(layer_dir)
    if subgraphs:
        return subgraphs
    if fallback_layer_pack_root:
        fallback = fallback_layer_pack_root / model_dir.name / f"layer_{layer_index}"
        return _load_subgraph_dir(fallback)
    return []


def _load_subgraph_dir(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for analysis in sorted(root.glob("*/analysis.json")):
        data = load_json(analysis)
        if not data:
            continue
        data["_analysis_path"] = str(analysis)
        data["_explanation_path"] = str(analysis.with_name("explanation.md"))
        out.append(data)
    out.sort(key=lambda item: int(item.get("ordinal") or 10**9))
    return out


def summarize_subgraph(record: dict[str, Any]) -> dict[str, Any]:
    classification = record.get("classification", {})
    return {
        "display_name": record.get("display_name", ""),
        "semantic_category": record.get("semantic_category", ""),
        "source_region_type": record.get("source_region_type", ""),
        "pruning_class": classification.get("pruning_class", record.get("pruning_class", "unknown")),
        "plan_status": classification.get("plan_status", record.get("plan_status", "unknown")),
        "validation_status": classification.get("validation_status", record.get("validation_status", "unknown")),
        "onnx_status": record.get("onnx_export", {}).get("status", record.get("onnx_status", "skipped")),
        "primitive_op_count": len(record.get("primitive_ops", [])),
        "ranking_count": len(record.get("local_ranking", [])),
        "plan_count": len(record.get("local_plans", [])),
        "validation_count": len(record.get("local_validations", [])),
        "verdict": record.get("verdict") or record.get("explanation", ""),
    }


def search_subgraphs(subgraphs: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    needle = text.lower().strip()
    if not needle:
        return []
    matches: list[dict[str, Any]] = []
    for item in subgraphs:
        fields = [
            item.get("display_name", ""),
            item.get("semantic_category", ""),
            item.get("source_region_type", ""),
            item.get("verdict", ""),
            item.get("explanation", ""),
            item.get("classification", {}).get("pruning_class", ""),
            item.get("classification", {}).get("plan_status", ""),
            item.get("classification", {}).get("validation_status", ""),
        ]
        explanation = read_text(Path(str(item.get("_explanation_path", "")))) if item.get("_explanation_path") else ""
        if needle in " ".join(str(value).lower() for value in [*fields, explanation]):
            matches.append(item)
    return matches


def find_onnx_path(
    model_safe_name: str,
    layer_index: int,
    node_slug: str,
    artifact_root: Path,
    fallback_artifact_root: Path | None = None,
) -> Path | None:
    candidates = [
        artifact_root / model_safe_name / "layers" / f"layer_{layer_index}" / node_slug / "subgraph.onnx",
    ]
    if fallback_artifact_root:
        candidates.append(fallback_artifact_root / model_safe_name / f"layer_{layer_index}" / node_slug / "subgraph.onnx")
    for path in candidates:
        if path.exists():
            return path
    return None


def open_path(path: Path, *, no_open: bool = False, open_command: str | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing file: {path}"
    if no_open:
        return True, str(path)
    command: list[str] | None = None
    if open_command:
        command = [open_command, str(path)]
    elif shutil.which("netron"):
        command = ["netron", str(path)]
    elif platform.system() == "Darwin" and shutil.which("open"):
        command = ["open", str(path)]
    elif shutil.which("xdg-open"):
        command = ["xdg-open", str(path)]
    if not command:
        return False, f"No opener found. Inspect manually: {path}"
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return False, f"Could not open {path}: {exc}"
    return True, " ".join(command)


def validation_summary(summary: dict[str, Any]) -> dict[str, int]:
    counts = summary.get("validation_status_counts", {}) if isinstance(summary.get("validation_status_counts"), dict) else {}
    return {
        "total_validations": int(summary.get("total_validations", summary.get("total_plans", 0)) or 0),
        "valid": int(summary.get("valid", summary.get("valid_plans", counts.get("valid", 0))) or 0),
        "warning": int(summary.get("warning", summary.get("warning_plans", counts.get("warning", 0))) or 0),
        "invalid": int(summary.get("invalid", summary.get("invalid_plans", counts.get("invalid", 0))) or 0),
        "unknown": int(summary.get("unknown", summary.get("unknown_plans", counts.get("unknown", 0))) or 0),
    }


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(none)"
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    sep = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def _layer_sort_key(name: str) -> int:
    try:
        return int(str(name).split("_")[-1])
    except Exception:
        return 10**9


def clear_screen_if_requested(plain: bool) -> None:
    if not plain:
        os.environ.get("TERM") and print("\033[2J\033[H", end="")
