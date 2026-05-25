#!/usr/bin/env python
"""Build a learner-oriented layer subgraph validation pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.layer_subgraph_validation_pack import build_layer_subgraph_validation_pack
from model_analysis.layer_subgraph_validation_text import write_layer_subgraph_pack_text
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one encoder-layer subgraph validation pack.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--export-onnx", dest="export_onnx", action="store_true", default=True)
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    parser.add_argument("--render-svg", action="store_true")
    parser.add_argument("--max-subgraphs", type=int)
    parser.add_argument("--include-auxiliary", action="store_true")
    parser.add_argument("--strict-onnx-export", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load(path: Path, label: str, hint: str | None = None) -> dict:
    if not path.exists():
        msg = f"[missing] {label} missing: {path}"
        if hint:
            msg += f"\nRun: {hint}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
        safe = safe_model_name(config["hf_id"])
        report_root = Path(args.output_root) if args.output_root else root / "reports" / "layer_subgraph_validation"
        artifact_root = root / "artifacts" / "layer_subgraphs"
        paths = {
            "tensor_ir": root / "reports" / "tensor_ir" / f"{safe}.json",
            "op_semantics": root / "reports" / "op_semantics" / f"{safe}.json",
            "structural_region_tree": root / "reports" / "structural_region_trees" / f"{safe}.json",
            "region_pruning_semantics": root / "reports" / "region_pruning_semantics" / f"{safe}.json",
            "ranking": root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json",
            "plans": root / "reports" / "pruning_plans" / f"{safe}.json",
            "validations": root / "reports" / "pruning_plan_validation" / f"{safe}.json",
            "abstract_expansion": root / "reports" / "abstract_node_expansions" / safe / "abstract_node_expansions_main.json",
        }
        static_onnx = root / "data" / "models" / "onnx_static" / safe / "model.static.onnx"
        dynamic_onnx = root / "data" / "models" / "onnx" / safe / "model.onnx"
        source_onnx = static_onnx if static_onnx.exists() else dynamic_onnx if dynamic_onnx.exists() else None
        pack = build_layer_subgraph_validation_pack(
            model_name=config["hf_id"],
            layer_index=args.layer,
            tensor_ir=_load(paths["tensor_ir"], "Tensor IR", f"python scripts/build_tensor_ir.py --model {config['name']}"),
            op_semantics=_load(paths["op_semantics"], "Op Semantics", f"python scripts/build_op_semantics.py --model {config['name']}"),
            structural_region_tree=_load(paths["structural_region_tree"], "Structural Region Tree", f"python scripts/build_structural_region_tree.py --model {config['name']}"),
            region_pruning_semantics=_load(paths["region_pruning_semantics"], "Region Pruning Semantics", f"python scripts/build_region_pruning_semantics.py --model {config['name']}"),
            ranking=_load(paths["ranking"], "Pruning Opportunity Ranking", f"python scripts/rank_pruning_opportunities.py --model {config['name']}"),
            plans=_load(paths["plans"], "Pruning Plans", f"python scripts/synthesize_pruning_plans.py --model {config['name']}"),
            validations=_load(paths["validations"], "Pruning Plan Validation", f"python scripts/validate_pruning_plans.py --model {config['name']}"),
            abstract_expansion=json.loads(paths["abstract_expansion"].read_text(encoding="utf-8")) if paths["abstract_expansion"].exists() else None,
            source_paths={key: str(value) for key, value in paths.items() if value.exists()},
            report_root=report_root,
            artifact_root=artifact_root,
            source_onnx_path=source_onnx,
            export_onnx=args.export_onnx,
            render_svg=args.render_svg,
            max_subgraphs=args.max_subgraphs,
            include_auxiliary=args.include_auxiliary,
            strict_onnx_export=args.strict_onnx_export,
        )
        dump_path = root / "reports" / "layer_subgraph_validation_dumps" / f"{safe}__layer_{args.layer}.lsubgraph"
        write_layer_subgraph_pack_text(pack, dump_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        summary = pack.summary
        print(f"[layer-subgraph-validation] {pack.model_name} layer={pack.layer_index}")
        print(f"  subgraphs: {summary.get('total_subgraphs', 0)}")
        print(f"  onnx exported: {summary.get('onnx_exported', 0)}")
        print(f"  onnx failed: {summary.get('onnx_failed', 0)}")
        print(f"  safe/constrained/blocked/auxiliary/unknown: {summary.get('safe_subgraphs', 0)}/{summary.get('constrained_subgraphs', 0)}/{summary.get('blocked_subgraphs', 0)}/{summary.get('auxiliary_subgraphs', 0)}/{summary.get('unknown_subgraphs', 0)}")
        print(f"  valid plan subgraphs: {summary.get('valid_plan_subgraphs', 0)}")
        print(f"  index: {report_root / safe / f'layer_{args.layer}' / 'index.md'}")
        print(f"  artifacts: {artifact_root / safe / f'layer_{args.layer}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
