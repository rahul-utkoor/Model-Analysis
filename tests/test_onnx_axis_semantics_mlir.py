from __future__ import annotations

from pathlib import Path

from model_analysis.onnx_axis_semantics import (
    AxisRelationKind,
    derive_semantics_from_mlir_evidence,
)
from model_analysis.onnx_axis_semantics_mlir import evidence_from_mlir_text


def test_mlir_access_evidence_derives_preserved_relation(tmp_path: Path) -> None:
    mlir = tmp_path / "preserve.mlir"
    mlir.write_text(
        """
module {
  func.func @main() {
    affine.for %b = 0 to 1 {
      affine.for %s = 0 to 2 {
        affine.for %j = 0 to 4 {
          %v = affine.load %X[%b, %s, %j] : memref<1x2x4xf32>
          affine.store %v, %Y[%b, %s, %j] : memref<1x2x4xf32>
        }
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    evidence, relations = evidence_from_mlir_text(mlir)
    semantics = derive_semantics_from_mlir_evidence(
        node_name="synthetic",
        op_type="DisplayOnly",
        topological_index=0,
        input_names=["X"],
        output_names=["Y"],
        mlir_evidence=evidence,
        axis_relations=relations,
    )

    assert semantics.semantic_class.value == "MLIR_DERIVED_INDEX_PRESERVING"
    assert any(relation.relation == AxisRelationKind.PRESERVED for relation in relations)
    assert semantics.evidence_tier.value == "PYTHON_MLIR_ACCESS"


def test_mlir_access_evidence_derives_reduction_or_blocker(tmp_path: Path) -> None:
    mlir = tmp_path / "qk.mlir"
    mlir.write_text(
        """
module {
  func.func @main() {
    affine.for %q = 0 to 8 {
      affine.for %k = 0 to 8 {
        affine.for %d = 0 to 64 {
          %qv = affine.load %Q[%q, %d] : memref<8x64xf32>
          %kv = affine.load %K[%k, %d] : memref<8x64xf32>
          affine.store %qv, %S[%q, %k] : memref<8x8xf32>
        }
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    evidence, relations = evidence_from_mlir_text(mlir)
    semantics = derive_semantics_from_mlir_evidence(
        node_name="synthetic_qk",
        op_type="DisplayOnly",
        topological_index=0,
        input_names=["Q", "K"],
        output_names=["S"],
        mlir_evidence=evidence,
        axis_relations=relations,
    )

    relation_kinds = {relation.relation for relation in relations}
    assert AxisRelationKind.REDUCED in relation_kinds or AxisRelationKind.MIXED in relation_kinds
    assert semantics.semantic_class.value in {
        "MLIR_DERIVED_MATMUL_QK_SCORE",
        "MLIR_DERIVED_MATMUL_GENERIC",
        "MLIR_DERIVED_REDUCTION",
    }
    if semantics.semantic_class.value == "MLIR_DERIVED_MATMUL_QK_SCORE":
        assert semantics.leader_candidate_kind == "blocker"
