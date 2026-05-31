"""Orchestrate static pruning-analysis coverage across configured models."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from model_analysis.full_model_analysis_report import (
    build_full_model_analysis_report,
    detect_layers,
    discover_model_artifacts,
    missing_required_artifacts,
)
from model_analysis.paths import ensure_dir, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.static_pipeline_status import StageStatus, make_model_status


STAGE_SPECS: dict[str, dict[str, Any]] = {
    "tensor_ir": {
        "outputs": ["reports/tensor_ir/{safe}.json"],
        "inputs": [],
        "hint": "scripts/build_tensor_ir.py --model {model}",
        "build": None,
    },
    "op_semantics": {
        "outputs": ["reports/op_semantics/{safe}.json"],
        "inputs": ["reports/tensor_ir/{safe}.json"],
        "hint": "scripts/build_op_semantics.py --model {model}",
        "build": "scripts/build_op_semantics.py",
    },
    "structural_region_tree": {
        "outputs": ["reports/structural_region_trees/{safe}.json"],
        "inputs": ["reports/tensor_ir/{safe}.json"],
        "hint": "scripts/build_structural_region_tree.py --model {model}",
        "build": "scripts/build_structural_region_tree.py",
    },
    "region_dimension_ir": {
        "outputs": ["reports/region_dimension_ir/{safe}.json"],
        "inputs": ["reports/structural_region_trees/{safe}.json"],
        "hint": "scripts/build_region_dimension_ir.py --model {model}",
        "build": "scripts/build_region_dimension_ir.py",
    },
    "region_pruning_semantics": {
        "outputs": ["reports/region_pruning_semantics/{safe}.json"],
        "inputs": ["reports/tensor_ir/{safe}.json", "reports/structural_region_trees/{safe}.json"],
        "hint": "scripts/build_region_pruning_semantics.py --model {model}",
        "build": "scripts/build_region_pruning_semantics.py",
    },
    "pruning_opportunity_ranking": {
        "outputs": ["reports/pruning_opportunity_rankings/{safe}.json"],
        "inputs": ["reports/region_pruning_semantics/{safe}.json"],
        "hint": "scripts/rank_pruning_opportunities.py --model {model}",
        "build": "scripts/rank_pruning_opportunities.py",
    },
    "pruning_plan_synthesis": {
        "outputs": ["reports/pruning_plans/{safe}.json"],
        "inputs": [
            "reports/pruning_opportunity_rankings/{safe}.json",
            "reports/region_pruning_semantics/{safe}.json",
            "reports/op_semantics/{safe}.json",
        ],
        "hint": "scripts/synthesize_pruning_plans.py --model {model}",
        "build": "scripts/synthesize_pruning_plans.py",
    },
    "pruning_plan_validation": {
        "outputs": ["reports/pruning_plan_validation/{safe}.json"],
        "inputs": [
            "reports/pruning_plans/{safe}.json",
            "reports/pruning_opportunity_rankings/{safe}.json",
            "reports/region_pruning_semantics/{safe}.json",
            "reports/op_semantics/{safe}.json",
        ],
        "hint": "scripts/validate_pruning_plans.py --model {model}",
        "build": "scripts/validate_pruning_plans.py",
        "not_applicable_if_no_plans": True,
    },
    "deadbranch_propagation": {
        "outputs": ["reports/deadbranch_propagation/{safe}.json"],
        "inputs": ["reports/op_semantics/{safe}.json"],
        "hint": "scripts/analyze_deadbranch_propagation.py --model {model}",
        "build": "scripts/analyze_deadbranch_propagation.py",
    },
    "layer_subgraph_validation": {
        "outputs": ["reports/model_analysis_reports/{safe}/layers"],
        "inputs": [
            "reports/tensor_ir/{safe}.json",
            "reports/op_semantics/{safe}.json",
            "reports/structural_region_trees/{safe}.json",
            "reports/region_pruning_semantics/{safe}.json",
            "reports/pruning_opportunity_rankings/{safe}.json",
            "reports/pruning_plans/{safe}.json",
            "reports/pruning_plan_validation/{safe}.json",
        ],
        "hint": "scripts/build_full_model_analysis_report.py --model {model} --layers all",
        "build": None,
    },
    "full_model_report": {
        "outputs": ["reports/model_analysis_reports/{safe}/index.json"],
        "inputs": [
            "reports/op_semantics/{safe}.json",
            "reports/region_pruning_semantics/{safe}.json",
            "reports/pruning_opportunity_rankings/{safe}.json",
        ],
        "hint": "scripts/build_full_model_analysis_report.py --model {model} --layers all",
        "build": None,
    },
    "cross_model_report": {
        "outputs": ["reports/model_analysis_reports/cross_model/index.json"],
        "inputs": [],
        "hint": "scripts/compare_model_analysis_reports.py --models all",
        "build": None,
    },
}


ORDERED_STAGES = [
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


def _format_paths(root: Path, safe: str, templates: list[str]) -> list[Path]:
    return [root / template.format(safe=safe) for template in templates]


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _all_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def _missing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _run_stage_script(root: Path, script: str, model: str, verbose: bool) -> tuple[bool, str, float]:
    start = time.perf_counter()
    cmd = [sys.executable, script, "--model", model]
    if verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    duration = time.perf_counter() - start
    if proc.returncode == 0:
        return True, (proc.stdout + proc.stderr).strip(), duration
    return False, (proc.stdout + proc.stderr).strip(), duration


def _safe_candidate_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return {
        "safe": int(summary.get("safe_candidates", 0) or 0),
        "constrained": int(summary.get("constrained_candidates", 0) or 0),
        "blocked": int(summary.get("blocked_candidates", 0) or 0),
        "auxiliary": int(summary.get("auxiliary_candidates", 0) or 0),
        "unknown": int(summary.get("unknown_candidates", 0) or 0),
        "mlp_safe": int(summary.get("mlp_safe_candidates", 0) or 0),
        "generic_mlp_safe": int(summary.get("generic_mlp_safe_candidates", 0) or 0),
        "generic_mlp_constrained": int(summary.get("generic_mlp_constrained_candidates", 0) or 0),
    }


def _plan_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return {"plans": int(summary.get("total_plans", 0) or 0)}


def _validation_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    status_counts = summary.get("validation_status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    return {
        "valid_plans": int(summary.get("valid_plans", status_counts.get("valid", 0)) or 0),
        "warning_plans": int(summary.get("warning_plans", status_counts.get("warning", 0)) or 0),
        "invalid_plans": int(summary.get("invalid_plans", status_counts.get("invalid", 0)) or 0),
        "unknown_plans": int(summary.get("unknown_plans", status_counts.get("unknown", 0)) or 0),
    }


def _model_report_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("model_summary", {})
    validation = summary.get("plan_validation", {})
    return {
        "layers": int(summary.get("layers_generated", 0) or 0),
        "subgraphs": int(summary.get("total_subgraphs", 0) or 0),
        "valid_plans": int(validation.get("valid", 0) or 0),
    }


def _deadbranch_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
    return {
        "total_pairs": int(summary.get("total_pairs", 0) or 0),
        "ffn_pairs": int(summary.get("ffn_pairs", 0) or 0),
        "attention_value_pairs": int(summary.get("attention_value_pairs", 0) or 0),
        "query_key_blocked_pairs": int(summary.get("query_key_blocked_pairs", 0) or 0),
        "sparsegpt_alignment_status": summary.get("sparsegpt_alignment_status", "unknown"),
    }


def _layer_subgraph_count(root: Path, safe: str) -> int:
    model_report = root / "reports" / "model_analysis_reports" / safe / "index.json"
    if model_report.exists():
        return _model_report_counts(model_report).get("subgraphs", 0)
    layer_root = root / "reports" / "model_analysis_reports" / safe / "layers"
    total = 0
    for path in layer_root.glob("layer_*/index.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        total += int(data.get("summary", {}).get("total_subgraphs", 0) or 0)
    return total


def collect_artifact_summary(root: Path, model_name: str) -> dict[str, Any]:
    safe = safe_model_name(model_name)
    ranking_path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
    plan_path = root / "reports" / "pruning_plans" / f"{safe}.json"
    validation_path = root / "reports" / "pruning_plan_validation" / f"{safe}.json"
    deadbranch_path = root / "reports" / "deadbranch_propagation" / f"{safe}.json"
    report_path = root / "reports" / "model_analysis_reports" / safe / "index.json"
    return {
        "ranking": _safe_candidate_counts(ranking_path),
        "plans": _plan_counts(plan_path),
        "validation": _validation_counts(validation_path),
        "deadbranch": _deadbranch_counts(deadbranch_path),
        "full_model_report": _model_report_counts(report_path),
    }


def _status_for_existing_or_missing(
    *,
    root: Path,
    model_cli_name: str,
    safe: str,
    stage_name: str,
    build_missing_analysis: bool,
    build_layer_packs: bool,
    strict: bool,
    verbose: bool,
) -> StageStatus:
    spec = STAGE_SPECS[stage_name]
    outputs = _format_paths(root, safe, spec.get("outputs", []))
    inputs = _format_paths(root, safe, spec.get("inputs", []))
    rel_outputs = [_rel(root, path) for path in outputs]
    rel_inputs = [_rel(root, path) for path in inputs]
    missing_inputs = [_rel(root, path) for path in _missing(inputs)]
    command_hint = spec.get("hint", "").format(model=model_cli_name)
    if _all_exist(outputs):
        if stage_name == "layer_subgraph_validation" and _layer_subgraph_count(root, safe) == 0:
            return StageStatus(
                stage_name=stage_name,
                status="skipped",
                required_inputs=rel_inputs,
                outputs=rel_outputs,
                command_hint=command_hint,
                notes="No layer/subgraph structure recovered for this model.",
            )
        return StageStatus(
            stage_name=stage_name,
            status="present_existing",
            required_inputs=rel_inputs,
            outputs=rel_outputs,
            command_hint=command_hint,
        )
    if stage_name == "pruning_plan_validation":
        plan_outputs = _format_paths(root, safe, STAGE_SPECS["pruning_plan_synthesis"]["outputs"])
        if not _all_exist(plan_outputs):
            ranking_path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
            counts = _safe_candidate_counts(ranking_path)
            if counts and counts.get("safe", 0) == 0:
                return StageStatus(
                    stage_name=stage_name,
                    status="not_applicable",
                    required_inputs=rel_inputs,
                    missing_inputs=missing_inputs,
                    outputs=rel_outputs,
                    command_hint=command_hint,
                    notes="No safe candidates/plans detected.",
                )
    if missing_inputs:
        return StageStatus(
            stage_name=stage_name,
            status="skipped",
            required_inputs=rel_inputs,
            missing_inputs=missing_inputs,
            outputs=rel_outputs,
            command_hint=command_hint,
            notes="Prerequisites missing.",
        )
    should_build_script = build_missing_analysis and bool(spec.get("build"))
    if should_build_script:
        ok, output, duration = _run_stage_script(root, spec["build"], model_cli_name, verbose)
        if ok and _all_exist(outputs):
            return StageStatus(
                stage_name=stage_name,
                status="built",
                required_inputs=rel_inputs,
                outputs=rel_outputs,
                command_hint=command_hint,
                duration_seconds=duration,
                notes=output[-500:],
            )
        status = StageStatus(
            stage_name=stage_name,
            status="failed",
            required_inputs=rel_inputs,
            outputs=rel_outputs,
            command_hint=command_hint,
            error=output or "Builder did not produce expected output.",
            duration_seconds=duration,
        )
        if strict:
            raise RuntimeError(f"{stage_name} failed for {model_cli_name}: {status.error}")
        return status
    if stage_name in {"layer_subgraph_validation", "full_model_report"} and build_layer_packs:
        return StageStatus(
            stage_name=stage_name,
            status="skipped",
            required_inputs=rel_inputs,
            outputs=rel_outputs,
            command_hint=command_hint,
            notes="Will be built by full-model report stage if supported.",
        )
    return StageStatus(
        stage_name=stage_name,
        status="skipped",
        required_inputs=rel_inputs,
        outputs=rel_outputs,
        command_hint=command_hint,
        notes="Output missing and build flag not enabled for this stage.",
    )


def build_static_pipeline_for_model(
    *,
    root: Path,
    model_name: str,
    build_missing_analysis: bool = False,
    build_layer_packs: bool = False,
    strict: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    config = get_model_config(model_name)
    hf_id = config["hf_id"]
    safe = safe_model_name(hf_id)
    stages: list[StageStatus] = []
    for stage_name in ORDERED_STAGES:
        if stage_name == "cross_model_report":
            stages.append(
                StageStatus(
                    stage_name=stage_name,
                    status="not_applicable",
                    command_hint="scripts/compare_model_analysis_reports.py --models all",
                    notes="Cross-model report is built by the coverage/all-model commands.",
                )
            )
            continue
        if stage_name in {"layer_subgraph_validation", "full_model_report"} and build_layer_packs:
            outputs = _format_paths(root, safe, STAGE_SPECS[stage_name]["outputs"])
            inputs = _format_paths(root, safe, STAGE_SPECS[stage_name]["inputs"])
            missing_inputs = _missing(inputs)
            if _all_exist(outputs):
                if stage_name == "layer_subgraph_validation" and _layer_subgraph_count(root, safe) == 0:
                    stages.append(
                        StageStatus(
                            stage_name=stage_name,
                            status="skipped",
                            required_inputs=[_rel(root, path) for path in inputs],
                            outputs=[_rel(root, path) for path in outputs],
                            command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                            notes="No layer/subgraph structure recovered for this model.",
                        )
                    )
                    continue
                stages.append(
                    StageStatus(
                        stage_name=stage_name,
                        status="present_existing",
                        required_inputs=[_rel(root, path) for path in inputs],
                        outputs=[_rel(root, path) for path in outputs],
                        command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                    )
                )
                continue
            if missing_inputs:
                stages.append(
                    StageStatus(
                        stage_name=stage_name,
                        status="skipped",
                        required_inputs=[_rel(root, path) for path in inputs],
                        missing_inputs=[_rel(root, path) for path in missing_inputs],
                        outputs=[_rel(root, path) for path in outputs],
                        command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                        notes="Prerequisites missing.",
                    )
                )
                continue
            if stage_name == "layer_subgraph_validation":
                stages.append(
                    StageStatus(
                        stage_name=stage_name,
                        status="skipped",
                        required_inputs=[_rel(root, path) for path in inputs],
                        outputs=[_rel(root, path) for path in outputs],
                        command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                        notes="Layer packs are produced as part of full_model_report.",
                    )
                )
                continue
            try:
                available, loaded, missing = discover_model_artifacts(root, hf_id)
                required_missing = missing_required_artifacts(missing)
                if required_missing:
                    raise FileNotFoundError(str(required_missing))
                layers = detect_layers(
                    loaded.get("region_pruning_semantics"), loaded.get("abstract_expansion")
                )
                if not layers:
                    layers = [0]
                start = time.perf_counter()
                build_full_model_analysis_report(
                    root=root,
                    model_name=hf_id,
                    layers=layers,
                    output_root=root / "reports" / "model_analysis_reports",
                    artifact_root=root / "artifacts" / "model_analysis_subgraphs",
                    export_onnx_subgraphs=False,
                    render_svg=False,
                    include_auxiliary=False,
                    strict=strict,
                )
                duration = time.perf_counter() - start
                stages.append(
                    StageStatus(
                        stage_name=stage_name,
                        status="built",
                        required_inputs=[_rel(root, path) for path in inputs],
                        outputs=[_rel(root, path) for path in outputs],
                        command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                        duration_seconds=duration,
                    )
                )
            except Exception as exc:
                if strict:
                    raise
                stages.append(
                    StageStatus(
                        stage_name=stage_name,
                        status="failed",
                        required_inputs=[_rel(root, path) for path in inputs],
                        outputs=[_rel(root, path) for path in outputs],
                        command_hint=STAGE_SPECS[stage_name]["hint"].format(model=hf_id),
                        error=str(exc),
                    )
                )
            continue
        stages.append(
            _status_for_existing_or_missing(
                root=root,
                model_cli_name=hf_id,
                safe=safe,
                stage_name=stage_name,
                build_missing_analysis=build_missing_analysis,
                build_layer_packs=build_layer_packs,
                strict=strict,
                verbose=verbose,
            )
        )
    artifacts = collect_artifact_summary(root, hf_id)
    status = make_model_status(
        model_name=hf_id,
        configured_model=config,
        stages=stages,
        artifacts=artifacts,
        notes=["BERT is the current complete reference model."] if config.get("name") == "bert-base-uncased" else [],
    )
    return status.__dict__ | {"stages": [stage.__dict__ for stage in status.stages]}


def configured_model_names(value: str) -> list[str]:
    if value == "all":
        return list_models()
    return [item.strip() for item in value.split(",") if item.strip()]


def build_static_pipeline_for_models(
    *,
    root: Path,
    models: list[str],
    build_missing_analysis: bool = False,
    build_layer_packs: bool = False,
    strict: bool = False,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    statuses = []
    for model in models:
        try:
            statuses.append(
                build_static_pipeline_for_model(
                    root=root,
                    model_name=model,
                    build_missing_analysis=build_missing_analysis,
                    build_layer_packs=build_layer_packs,
                    strict=strict,
                    verbose=verbose,
                )
            )
        except Exception:
            if strict:
                raise
            config = get_model_config(model)
            stages = [
                StageStatus(
                    stage_name="tensor_ir",
                    status="failed",
                    error=f"orchestrator failed for model {model}",
                )
            ]
            status = make_model_status(
                model_name=config["hf_id"],
                configured_model=config,
                stages=stages,
                artifacts={},
                notes=["Model orchestration failed before status collection."],
            )
            statuses.append(status.__dict__ | {"stages": [stage.__dict__ for stage in status.stages]})
    return statuses
