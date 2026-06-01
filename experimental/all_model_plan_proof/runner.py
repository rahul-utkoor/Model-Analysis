"""Build all-model plan proofs by reusing the MLIR coverage evaluator."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experimental.all_model_plan_proof.aggregate import aggregate_model_proofs
from experimental.all_model_plan_proof.config import AttentionValuePolicy, ModelPlanExpectation, PlanFamily, model_expectations
from experimental.all_model_plan_proof.proof_model import AllModelPlanProof, ModelPlanProof, ModelPlanSummary, PlanProofCell
from experimental.mlir_evidence_coverage.config import ModelSpec, pattern_specs
from experimental.mlir_evidence_coverage.coverage_case import CoveragePatternKind, CoverageResult, CoverageVerdict
from experimental.mlir_evidence_coverage.discovery import match_cases_for_model
from experimental.mlir_evidence_coverage.runner import CoverageRunOptions, run_coverage_case


@dataclass(frozen=True)
class AllModelRunOptions:
    layers: str = "layer0"
    output_root: str = "reports/all_model_plan_proof/artifacts"
    build_missing_value_paths: bool = False
    run_native_pass: bool = True
    native_pass_tool: str | None = None
    onnx_mlir: str | None = None
    mlir_opt: str | None = None
    verbose: bool = False


PROVEN_VERDICTS = {"proven", "fallback_proven"}


def _model_spec(expectation: ModelPlanExpectation) -> ModelSpec:
    return ModelSpec(expectation.model_name, expectation.artifact_name, expectation.short_name, expectation.layer_count)


def _selected_layers(expectation: ModelPlanExpectation, selector: str) -> list[int]:
    return list(range(expectation.layer_count)) if selector == "all" else [0]


def _coverage_options(options: AllModelRunOptions) -> CoverageRunOptions:
    return CoverageRunOptions(
        output_root=options.output_root,
        run_native_pass=options.run_native_pass,
        native_pass_tool=options.native_pass_tool,
        onnx_mlir=options.onnx_mlir,
        mlir_opt=options.mlir_opt,
        verbose=options.verbose,
    )


def _cases_by_layer(expectation: ModelPlanExpectation, pattern: str, layers: str):
    spec = pattern_specs(pattern)[0]
    return {
        case.layer_index: case
        for case in match_cases_for_model(_model_spec(expectation), [spec], layers=layers)
    }


def _map_verdict(result: CoverageResult, family: PlanFamily) -> str:
    if result.verdict in {CoverageVerdict.NATIVE_PROVEN, CoverageVerdict.ACCESS_PROVEN}:
        if family != PlanFamily.QK_BLOCKER and not result.dfa_ran:
            return "partial"
        return "proven"
    if result.verdict == CoverageVerdict.FALLBACK_PROVEN:
        if family != PlanFamily.QK_BLOCKER and not result.dfa_ran:
            return "partial"
        return "fallback_proven"
    if result.verdict == CoverageVerdict.BLOCKED_AS_EXPECTED:
        return "blocked_as_expected"
    if result.verdict in {CoverageVerdict.PARTIAL, CoverageVerdict.UNKNOWN}:
        return "partial"
    return result.verdict.value


def coverage_result_to_cell(result: CoverageResult, family: PlanFamily) -> PlanProofCell:
    return PlanProofCell(
        result.case.model_name,
        next((part for part in Path(result.case.onnx_path).parts if "__" in part or part == result.case.model_name), result.case.model_name),
        result.case.layer_index,
        family,
        found_artifact=result.found,
        evidence_tier=result.evidence_tier.value,
        recognized_pattern=", ".join(result.recognized_patterns),
        dfa_ran=result.dfa_ran,
        dfa_reached_fixed_point=result.dfa_ran,
        verdict=_map_verdict(result, family),
        report_path=result.report_path,
        artifact_path=result.case.onnx_path,
        warnings=list(result.warnings),
    )


def _missing_cell(expectation: ModelPlanExpectation, layer_index: int, family: PlanFamily, warning: str) -> PlanProofCell:
    return PlanProofCell(
        expectation.model_name,
        expectation.artifact_name,
        layer_index,
        family,
        warnings=[warning],
    )


def _unsupported_attention_cell(expectation: ModelPlanExpectation, layer_index: int) -> PlanProofCell:
    return PlanProofCell(
        expectation.model_name,
        expectation.artifact_name,
        layer_index,
        PlanFamily.ATTENTION_VALUE_PATH,
        verdict="unsupported",
        warnings=["fused_qkv_value_path_gap: a separately justified value slice has not been recovered."],
    )


def _run_build_hook(expectation: ModelPlanExpectation, options: AllModelRunOptions) -> list[str]:
    if not options.build_missing_value_paths or expectation.attention_value_policy == AttentionValuePolicy.FUSED_QKV_GAP:
        return []
    target = Path("artifacts/attention_value_path_subgraphs") / expectation.artifact_name / "layers"
    wanted = _selected_layers(expectation, options.layers)
    missing = [index for index in wanted if not list((target / f"layer_{index}").glob("*/subgraph.onnx"))]
    if not missing:
        return []
    command = [
        sys.executable,
        "scripts/build_attention_value_path_subgraphs.py",
        "--model",
        expectation.model_name,
        "--export-onnx",
    ]
    command.extend(["--layers", "all"] if options.layers == "all" else ["--layer", "0"])
    if options.verbose:
        command.append("--verbose")
        print(f"[all-model-proof] build missing value paths: {' '.join(command)}")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    warnings: list[str] = []
    if completed.returncode != 0:
        warnings.append(
            f"value-path build hook failed for {expectation.model_name} with return code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return warnings


def summarize_model(
    expectation: ModelPlanExpectation,
    ffn_cells: list[PlanProofCell],
    attention_cells: list[PlanProofCell],
    qk_cells: list[PlanProofCell],
) -> tuple[ModelPlanSummary, str]:
    plan_cells = [*ffn_cells, *attention_cells]
    summary = ModelPlanSummary(
        ffn_expected=len(ffn_cells),
        ffn_found=sum(cell.found_artifact for cell in ffn_cells),
        ffn_proven=sum(cell.verdict in PROVEN_VERDICTS for cell in ffn_cells),
        attention_expected=len(attention_cells),
        attention_found=sum(cell.found_artifact for cell in attention_cells),
        attention_proven=sum(cell.verdict in PROVEN_VERDICTS for cell in attention_cells),
        attention_partial=sum(cell.verdict == "partial" for cell in attention_cells),
        attention_missing=sum(cell.verdict == "missing" for cell in attention_cells),
        attention_unsupported=sum(cell.verdict == "unsupported" for cell in attention_cells),
        qk_blockers_expected=len(qk_cells),
        qk_blockers_proven=sum(cell.verdict == "blocked_as_expected" for cell in qk_cells),
        total_expected=len(plan_cells),
        total_proven=sum(cell.verdict in PROVEN_VERDICTS for cell in plan_cells),
        native_evidence_count=sum(cell.evidence_tier == "native_mlir_dependence_evidence" for cell in plan_cells),
        fallback_count=sum(cell.verdict == "fallback_proven" for cell in plan_cells),
        partial_count=sum(cell.verdict == "partial" for cell in plan_cells),
        missing_count=sum(cell.verdict == "missing" for cell in plan_cells),
        unsupported_count=sum(cell.verdict == "unsupported" for cell in plan_cells),
        failed_count=sum(cell.verdict == "failed" for cell in plan_cells),
    )
    if summary.failed_count:
        verdict = "failed"
    elif summary.total_proven == summary.total_expected:
        verdict = "complete_plan_proof"
    elif summary.unsupported_count:
        verdict = "unsupported_attention_value_path"
    else:
        verdict = "partial_plan_proof"
    return summary, verdict


def run_model_proof(expectation: ModelPlanExpectation, options: AllModelRunOptions) -> ModelPlanProof:
    layers = _selected_layers(expectation, options.layers)
    hook_warnings = _run_build_hook(expectation, options)
    coverage_options = _coverage_options(options)
    ffn_cases = _cases_by_layer(expectation, "FFN_MLP_INTERMEDIATE", options.layers)
    ffn_cells = [
        coverage_result_to_cell(run_coverage_case(ffn_cases[layer], coverage_options), PlanFamily.FFN_INTERMEDIATE)
        if layer in ffn_cases
        else _missing_cell(expectation, layer, PlanFamily.FFN_INTERMEDIATE, "FFN coverage case was not discovered.")
        for layer in layers
    ]
    if expectation.attention_value_policy == AttentionValuePolicy.FUSED_QKV_GAP:
        attention_cells = [_unsupported_attention_cell(expectation, layer) for layer in layers]
    else:
        attention_cases = _cases_by_layer(expectation, "ATTENTION_VALUE_PATH", options.layers)
        attention_cells = [
            coverage_result_to_cell(run_coverage_case(attention_cases[layer], coverage_options), PlanFamily.ATTENTION_VALUE_PATH)
            if layer in attention_cases
            else _missing_cell(expectation, layer, PlanFamily.ATTENTION_VALUE_PATH, "Attention value-path coverage case was not discovered.")
            for layer in layers
        ]
    qk_cases = _cases_by_layer(expectation, "ATTENTION_QK_SCORE", options.layers)
    qk_cells = [
        coverage_result_to_cell(run_coverage_case(qk_cases[layer], coverage_options), PlanFamily.QK_BLOCKER)
        for layer in layers
        if layer in qk_cases and Path(qk_cases[layer].onnx_path).is_file()
    ]
    if hook_warnings and attention_cells:
        attention_cells[0].warnings.extend(hook_warnings)
    summary, final_verdict = summarize_model(expectation, ffn_cells, attention_cells, qk_cells)
    return ModelPlanProof(
        expectation.model_name,
        expectation.artifact_name,
        len(layers),
        ffn_cells,
        attention_cells,
        qk_cells,
        summary,
        final_verdict,
        expectation.notes,
    )


def run_all_model_proof(selector: str = "all", options: AllModelRunOptions | None = None) -> AllModelPlanProof:
    selected_options = options or AllModelRunOptions()
    models = []
    for expectation in model_expectations(selector):
        if selected_options.verbose:
            print(f"[all-model-proof] analyze {expectation.model_name}")
        model = run_model_proof(expectation, selected_options)
        models.append(model)
        if selected_options.verbose:
            print(
                f"[all-model-proof] {model.model_name}: proven={model.summary.total_proven}/{model.summary.total_expected} "
                f"verdict={model.final_verdict}"
            )
    aggregate = aggregate_model_proofs(models)
    return AllModelPlanProof.create(
        models,
        aggregate,
        [
            "GPT-2 fused-QKV value slices are recovered only when an explicit Split/Slice/Gather branch reaches attention context.",
            "ViT support reflects the local exported graph; this export exposes separate q_proj, k_proj, and v_proj operators.",
            "Residual and LayerNorm protection are not counted in the two-plan-per-layer propagation total.",
            "Fallback evidence remains visibly distinguished from native MLIR dependence evidence.",
            "This report evaluates static evidence only; it does not execute pruning or mutate model weights.",
        ],
    )
