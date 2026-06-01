from experimental.bert_24_plan_proof.runner import build_bert_24_plan_proof


def _plans() -> dict:
    return {
        "plans": [
            {
                "plan_id": f"plan::{layer}",
                "plan_kind": "feedforward_intermediate_dim_plan",
                "candidate_region_name": f"Layer {layer} Feed Forward",
                "plan_status": "ready_symbolic",
            }
            for layer in range(12)
        ]
    }


def _validations() -> dict:
    return {"validations": [{"plan_id": f"plan::{layer}", "validation_status": "valid"} for layer in range(12)]}


def _paths(count: int = 12) -> dict:
    return {
        "paths": [
            {
                "layer_index": layer,
                "analysis_status": "seedable",
                "axis_mapping": {"mapping_status": "proven"},
            }
            for layer in range(count)
        ]
    }


def _coverage(count: int = 12) -> dict:
    return {
        "cases": [
            {
                "case": {"layer_index": layer, "pattern_kind": "ATTENTION_VALUE_PATH"},
                "evidence_tier": "native_mlir_dependence_evidence",
                "verdict": "native_proven",
                "dfa_ran": True,
            }
            for layer in range(count)
        ]
    }


def test_complete_bert_24_plan_proof() -> None:
    proof = build_bert_24_plan_proof(_plans(), _validations(), _paths(), _coverage())
    assert proof.summary.expected_plans == 24
    assert proof.summary.ffn_proven == 12
    assert proof.summary.attention_proven == 12
    assert proof.summary.total_proven == 24
    assert proof.summary.final_verdict == "complete_24_plan_proof"


def test_missing_attention_layer_makes_proof_partial() -> None:
    proof = build_bert_24_plan_proof(_plans(), _validations(), _paths(11), _coverage(11))
    assert proof.summary.attention_proven == 11
    assert proof.summary.missing == 1
    assert proof.summary.final_verdict == "partial"
