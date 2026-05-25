from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_exporter_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "export_compact_analysis_snapshot.py"
    spec = importlib.util.spec_from_file_location("export_compact_analysis_snapshot", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_subgraph(
    ordinal: int,
    slug: str,
    name: str,
    category: str,
    pruning_class: str,
    validation_status: str = "not_applicable",
) -> dict:
    return {
        "ordinal": ordinal,
        "node_slug": slug,
        "display_name": name,
        "source_region_type": "SyntheticRegion",
        "semantic_category": category,
        "classification": {
            "pruning_class": pruning_class,
            "plan_status": "valid_plan" if validation_status == "valid" else "no_plan_expected",
            "validation_status": validation_status,
        },
        "primitive_ops": [
            {
                "op_id": f"op::{ordinal}",
                "source_name": f"/model/bert/encoder/layer.0/{slug}/MatMul",
                "op_type": "MatMul",
                "topological_index": ordinal,
            }
        ],
        "local_op_semantics": [
            {
                "source_name": f"/model/bert/encoder/layer.0/{slug}/MatMul",
                "semantic_kind": category,
                "semantic_category": category,
                "parameterized": False,
                "direct_pruning": "blocked",
            }
        ],
        "local_ranking": [
            {
                "candidate_id": f"candidate::{ordinal}",
                "candidate_kind": category,
                "pruning_class": pruning_class,
                "rank_score": 95 if pruning_class == "safe" else 10,
                "confidence": "high",
                "target_dimension": "intermediate_dim",
                "blockers": [],
                "reason": "synthetic reason",
            }
        ],
        "local_plans": [
            {
                "plan_id": "plan::0",
                "plan_kind": "feedforward_intermediate_dim_plan",
                "plan_status": "ready_symbolic",
                "target_dimension": "intermediate_dim",
                "symbolic_index_set": {"name": "I_layer_0_intermediate"},
                "actions": [
                    {
                        "action_type": "prune_producer_output",
                        "target_source_name": "/model/bert/encoder/layer.0/intermediate/dense/MatMul",
                        "target_axis": "output_dim",
                        "dimension": "intermediate_dim",
                    }
                ],
            }
        ]
        if category == "feed_forward_block"
        else [],
        "local_validations": [
            {
                "validation_id": "validation::0",
                "validation_status": "valid",
                "validation_score": 100,
                "failed_checks": [],
                "warning_checks": [],
            }
        ]
        if validation_status == "valid"
        else [],
        "onnx_export": {"attempted": False, "status": "skipped", "output_path": ""},
    }


def test_compact_snapshot_records_missing_optional_artifacts_and_writes_outputs(tmp_path: Path):
    exporter = load_exporter_module()
    model = "bert-base-uncased"
    layer_dir = tmp_path / "reports" / "layer_subgraph_validation" / model / "layer_0"

    feed_forward = make_subgraph(
        1,
        "01_layer_0_feed_forward",
        "Layer 0 Feed Forward",
        "feed_forward_block",
        "safe",
        validation_status="valid",
    )
    attention_score = make_subgraph(
        2,
        "02_layer_0_attention_score_matmul",
        "Layer 0 Attention Score MatMul",
        "attention_score_matmul",
        "blocked",
    )
    index = {
        "model_name": model,
        "layer_index": 0,
        "subgraphs": [feed_forward, attention_score],
        "summary": {
            "total_subgraphs": 2,
            "onnx_exported": 0,
            "onnx_skipped": 2,
            "onnx_failed": 0,
            "safe_subgraphs": 1,
            "blocked_subgraphs": 1,
            "valid_plan_subgraphs": 1,
        },
    }
    write_json(layer_dir / "index.json", index)
    write_json(layer_dir / feed_forward["node_slug"] / "analysis.json", feed_forward)
    write_json(layer_dir / attention_score["node_slug"] / "analysis.json", attention_score)
    write_json(
        tmp_path / "reports" / "op_semantics" / f"{model}.json",
        {
            "summary": {
                "semantic_kind_counts": {
                    "parameterized_linear_matmul": 1,
                    "attention_score_matmul": 1,
                    "unknown": 0,
                }
            }
        },
    )

    snapshot = exporter.build_snapshot(
        tmp_path,
        model,
        0,
        max_ops_per_subgraph=12,
        max_evidence_per_section=8,
    )

    assert snapshot["pipeline_summary"]["op_semantics"]["present"] is True
    assert snapshot["pipeline_summary"]["layer_subgraph_pack"]["summary"]["total_subgraphs"] == 2
    assert snapshot["missing_artifacts"]
    verdicts = {item["display_name"]: item["verdict"] for item in snapshot["subgraphs"]}
    assert verdicts["Layer 0 Feed Forward"] == "safe; symbolic plan exists and validates."
    assert "Q x K^T contraction" in verdicts["Layer 0 Attention Score MatMul"]

    markdown_path, json_path = exporter.write_outputs(
        snapshot,
        tmp_path / "reports" / "compact_analysis_snapshots",
        write_markdown=True,
        write_json=True,
        max_evidence_per_section=8,
    )

    assert markdown_path is not None and markdown_path.exists()
    assert json_path is not None and json_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## 3. Layer 0 subgraph table" in markdown
    assert "Layer 0 Feed Forward" in markdown
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["subgraphs"][0]["display_name"] == "Layer 0 Feed Forward"
