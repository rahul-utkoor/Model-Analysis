from pathlib import Path

from experimental.mlir_evidence_coverage.config import ModelSpec, pattern_specs
from experimental.mlir_evidence_coverage.coverage_case import CoveragePatternKind
from experimental.mlir_evidence_coverage.discovery import discover_model_subgraphs, match_cases_for_model


def test_discovery_missing_model_not_fatal(tmp_path: Path) -> None:
    assert discover_model_subgraphs("missing-model", tmp_path / "primary", tmp_path / "fallback") == []


def test_discovery_matches_synthetic_subgraph_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    onnx = artifact_root / "synthetic-model/layers/layer_0/03_layer_0_mlp_block/subgraph.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.touch()
    model = ModelSpec("synthetic/model", "synthetic-model", "synthetic", 1)
    cases = match_cases_for_model(model, pattern_specs("FFN_MLP_INTERMEDIATE"), artifact_root=artifact_root)
    assert len(cases) == 1
    assert cases[0].pattern_kind == CoveragePatternKind.FFN_MLP_INTERMEDIATE
    assert cases[0].onnx_path == str(onnx)
    assert cases[0].case_id == "synthetic_layer0_mlp"


def test_discovery_prefers_current_artifact_layout(tmp_path: Path) -> None:
    primary = tmp_path / "current"
    fallback = tmp_path / "fallback"
    current_onnx = primary / "synthetic-model/layers/layer_0/03_layer_0_mlp_block/subgraph.onnx"
    old_onnx = fallback / "synthetic-model/layer_0/03_layer_0_mlp_block/subgraph.onnx"
    current_onnx.parent.mkdir(parents=True)
    old_onnx.parent.mkdir(parents=True)
    current_onnx.touch()
    old_onnx.touch()
    model = ModelSpec("synthetic/model", "synthetic-model", "synthetic", 1)
    cases = match_cases_for_model(model, pattern_specs("FFN_MLP_INTERMEDIATE"), artifact_root=primary, fallback_root=fallback)
    assert cases[0].onnx_path == str(current_onnx)
