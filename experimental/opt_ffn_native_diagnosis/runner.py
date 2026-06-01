"""Diagnose and repair OPT FFN local evidence artifacts conservatively."""

from __future__ import annotations

from pathlib import Path

from experimental.mlir_axis_bridge.bridge_runner import MlirAxisBridgeResult, analyze_onnx_with_mlir_bridge
from experimental.opt_ffn_native_diagnosis.diagnosis_model import (
    BlockerKind,
    OptFfnNativeDiagnosis,
    OptFfnNativeDiagnosisReport,
)
from experimental.onnx_axis_bridge.onnx_graph_summary import node_by_id, summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import load_onnx_subgraph
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHintKind, infer_pattern_hints


MODEL_NAME = "facebook__opt-125m"


def locate_opt_ffn_artifact(layer_index: int, artifact_root: str | Path = "artifacts/model_analysis_subgraphs") -> Path | None:
    layer = Path(artifact_root) / MODEL_NAME / "layers" / f"layer_{layer_index}"
    for pattern in ("*mlp_block*/subgraph.onnx", "*feed_forward*/subgraph.onnx", "*ffn*/subgraph.onnx"):
        matches = sorted(layer.glob(pattern))
        if matches:
            return matches[0]
    return None


def extract_opt_ffn_core(source: str | Path, output: str | Path) -> Path:
    """Export only the topology-proven projection -> activation -> projection region."""
    try:
        from onnx.utils import extract_model
    except ImportError as exc:
        raise RuntimeError("onnx is required to export an OPT FFN-core evidence artifact") from exc
    subgraph = load_onnx_subgraph(source)
    summary = summarize_subgraph(subgraph)
    hints = [hint for hint in infer_pattern_hints(subgraph, summary) if hint.kind == OnnxPatternHintKind.FFN_LIKE]
    if not hints:
        raise ValueError(f"no FFN projection -> activation -> projection chain was proven in {source}")
    nodes = node_by_id(summary)
    path = [nodes[node_id] for node_id in hints[0].nodes]
    first, last = path[0], path[-1]
    if not first.inputs or not last.outputs:
        raise ValueError(f"FFN core boundary tensors were not recoverable in {source}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    extract_model(str(source), str(output), [first.inputs[0]], [last.outputs[0]], check_model=True)
    return output


def _patterns(result: MlirAxisBridgeResult) -> set[str]:
    return set(result.summary.get("axis_patterns", ()))


def _lowered(result: MlirAxisBridgeResult) -> bool:
    return any(command.stage == "lowered_mlir" and command.returncode == 0 for command in result.lowering_result.commands)


def _original_blocker(result: MlirAxisBridgeResult) -> tuple[BlockerKind, str]:
    lowered = next((command for command in result.lowering_result.commands if command.stage == "lowered_mlir"), None)
    if lowered and lowered.returncode:
        detail = lowered.stderr.strip().splitlines()[0] if lowered.stderr.strip() else f"exit code {lowered.returncode}"
        return (
            BlockerKind.ONNX_MLIR_LOWERING_FAILED,
            "The full OPT MLP block includes unrelated LayerNorm/residual boundary nodes. "
            f"ONNX-MLIR aborts before affine lowering: {detail}",
        )
    if not any("affine.for" in artifact.dialect_hints for artifact in result.artifacts):
        return BlockerKind.NO_AFFINE_LOOPS, "The emitted MLIR contains no affine loop stage."
    if not any(summary.access_records for summary in result.mlir_access_summaries):
        return BlockerKind.NO_LOAD_STORE_ACCESSES, "The emitted MLIR contains no indexed affine/memref accesses."
    return BlockerKind.UNKNOWN, "Native FFN proof was not recovered from the original block artifact."


def run_layer_diagnosis(
    layer_index: int,
    output_root: str | Path,
    *,
    artifact_root: str | Path = "artifacts/model_analysis_subgraphs",
    core_artifact_root: str | Path = "artifacts/opt_ffn_native_subgraphs",
    run_native_pass: bool = True,
    native_pass_tool: str | None = None,
    onnx_mlir: str | None = None,
    mlir_opt: str | None = None,
) -> OptFfnNativeDiagnosis:
    source = locate_opt_ffn_artifact(layer_index, artifact_root)
    if source is None:
        return OptFfnNativeDiagnosis(
            MODEL_NAME,
            layer_index,
            str(Path(artifact_root) / MODEL_NAME / "layers" / f"layer_{layer_index}"),
            blocker_kind=BlockerKind.NO_ONNX_ARTIFACT,
            blocker_explanation="No OPT MLP block ONNX artifact was found.",
            suggested_fix="Regenerate the model-analysis subgraph atlas for this OPT layer.",
        )
    layer_root = Path(output_root) / f"layer_{layer_index}"
    original = analyze_onnx_with_mlir_bridge(
        source,
        layer_root / "original_mlir",
        onnx_mlir,
        mlir_opt,
        "ffn",
        run_native_pass=run_native_pass,
        native_pass_tool=native_pass_tool,
        native_output_dir=layer_root / "original_native",
    )
    blocker, explanation = _original_blocker(original)
    core = (
        Path(core_artifact_root)
        / MODEL_NAME
        / "layers"
        / f"layer_{layer_index}"
        / f"06_opt_decoder_block_{layer_index}_mlp_native_core"
        / "subgraph.onnx"
    )
    extract_opt_ffn_core(source, core)
    repaired = analyze_onnx_with_mlir_bridge(
        core,
        layer_root / "core_mlir",
        onnx_mlir,
        mlir_opt,
        "ffn",
        run_native_pass=run_native_pass,
        native_pass_tool=native_pass_tool,
        native_output_dir=layer_root / "core_native",
    )
    native = repaired.native_dependence_report
    relations = native.relations if native else []
    native_ffn = "FFN_INTERMEDIATE_CHAIN" in _patterns(repaired) and "native_mlir_dependence_evidence" in repaired.evidence_source
    return OptFfnNativeDiagnosis(
        MODEL_NAME,
        layer_index,
        str(source),
        str(core),
        _lowered(repaired),
        [artifact.path for artifact in repaired.artifacts],
        sorted({dialect for artifact in repaired.artifacts for dialect in artifact.dialect_hints}),
        repaired.native_pass_result is not None,
        repaired.native_pass_result.returncode if repaired.native_pass_result else None,
        len(relations),
        sum(relation.relation_kind == "preserved" for relation in relations),
        sum(relation.relation_kind == "reduced" for relation in relations),
        sum(relation.relation_kind == "mixed" for relation in relations),
        native_ffn,
        "FFN_INTERMEDIATE_CHAIN" in _patterns(original),
        blocker,
        explanation,
        "Extract the topology-proven fc1 -> activation -> fc2 core as the local ONNX-MLIR evidence unit.",
        native_ffn,
        {
            "original_mlir": str(layer_root / "original_mlir"),
            "core_mlir": str(layer_root / "core_mlir"),
        },
    )


def run_diagnosis(
    layers: str,
    output_root: str | Path,
    **kwargs,
) -> OptFfnNativeDiagnosisReport:
    indices = list(range(12)) if layers == "all" else [0]
    return OptFfnNativeDiagnosisReport.create([run_layer_diagnosis(index, output_root, **kwargs) for index in indices])
