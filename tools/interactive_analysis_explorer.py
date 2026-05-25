#!/usr/bin/env python3
"""Interactive read-only explorer for static pruning-analysis reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_analysis.interactive_analysis_explorer import (  # noqa: E402
    ModelRef,
    discover_layers,
    discover_models,
    discover_subgraphs,
    find_model,
    find_onnx_path,
    load_json,
    load_model_summary,
    open_path,
    print_table,
    read_text,
    search_subgraphs,
    summarize_subgraph,
    validation_summary,
)


class AnalysisExplorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = ROOT
        self.report_root = self.root / Path(args.report_root)
        self.layer_pack_root = self.root / Path(args.layer_pack_root)
        self.artifact_root = self.root / Path(args.artifact_root)
        self.fallback_artifact_root = self.root / "artifacts" / "layer_subgraphs"
        self.no_open = args.no_open
        self.open_command = args.open_command
        self.plain = args.plain
        self.verbose = args.verbose
        self.initial_model = args.model
        self.initial_layer = args.layer
        self.scripted = [item.strip() for item in (args.scripted or "").split(";") if item.strip()]

    def run(self) -> int:
        models = discover_models(self.report_root)
        if not models:
            print("No model analysis reports found.")
            print("Run:")
            print("  ./conda-env/bin/python scripts/build_full_model_analysis_report.py --model bert-base-uncased --layers all --export-onnx-subgraphs --verbose")
            return 1
        model = find_model(models, self.initial_model) if self.initial_model else self.choose_model(models)
        if not model:
            print(f"Model not found: {self.initial_model}")
            return 1
        data = load_model_summary(model)
        if self.initial_layer is not None:
            self.show_model_summary(data)
            return self.layer_loop(model, int(self.initial_layer))
        return self.model_loop(model)

    def prompt(self, label: str) -> str:
        if self.scripted:
            value = self.scripted.pop(0)
            print(f"{label}{value}")
            return value
        try:
            return input(label)
        except EOFError:
            return "q"

    def choose_model(self, models: list[ModelRef]) -> ModelRef | None:
        print("Static Analysis Explorer")
        print("Models available:")
        for idx, model in enumerate(models, start=1):
            print(f"  {idx}. {model.model_name}")
        while True:
            value = self.prompt("Select model [number/name/q]: ").strip()
            if value.lower() in {"q", "quit", "exit"}:
                return None
            if value.isdigit() and 1 <= int(value) <= len(models):
                return models[int(value) - 1]
            model = find_model(models, value)
            if model:
                return model
            print("Unknown model selection.")

    def model_loop(self, model: ModelRef) -> int:
        data = load_model_summary(model)
        self.show_model_summary(data)
        while True:
            cmd = self.prompt("model> ").strip()
            if cmd in {"q", "quit", "exit"}:
                return 0
            if cmd in {"s", "summary"}:
                self.show_model_summary(data)
            elif cmd in {"l", "layers"}:
                self.list_layers(model)
            elif cmd in {"p", "pipeline"}:
                self.show_pipeline(data)
            elif cmd in {"r", "ranking"}:
                self.show_ranking_summary(data)
            elif cmd in {"v", "validation"}:
                self.show_validation_summary(data)
            elif cmd in {"o", "open"}:
                self.open_report(model)
            elif cmd in {"c", "compare"}:
                self.show_compare()
            elif cmd.startswith("find "):
                self.find_in_model(model, cmd[5:])
            elif cmd.startswith("layer "):
                self.layer_loop(model, int(cmd.split(maxsplit=1)[1]))
            elif cmd in {"g", "go"}:
                value = self.prompt("Layer/block number: ").strip()
                if value.isdigit():
                    self.layer_loop(model, int(value))
            else:
                self.print_model_commands()

    def print_model_commands(self) -> None:
        print("Available commands:")
        print("  s/summary, l/layers, layer <n>, g, pipeline, ranking, validation")
        print("  find <text>, open, compare, q")

    def show_model_summary(self, data: dict[str, Any]) -> None:
        summary = data.get("model_summary", {})
        ranking = summary.get("ranking", {})
        plans = summary.get("plans", {})
        validation = summary.get("plan_validation", {})
        print(f"\nModel: {data.get('model_name', 'unknown')}")
        print(f"Layers: {summary.get('layers_generated', 0)}")
        print(f"Subgraphs: {summary.get('total_subgraphs', 0)}")
        print(f"Safe candidates: {ranking.get('safe', 0)}")
        print(f"MLP safe candidates: {ranking.get('mlp_safe_candidates', 0)}")
        print(f"Plans: {plans.get('total_plans', 0)}")
        print(f"Valid plans: {validation.get('valid', validation.get('valid_plans', 0))}")
        self.print_model_commands()

    def list_layers(self, model: ModelRef) -> None:
        rows = []
        for item in discover_layers(model.model_dir):
            summary = item["summary"]
            rows.append(
                {
                    "layer": item["layer_index"],
                    "subgraphs": summary.get("total_subgraphs", 0),
                    "safe": summary.get("safe", 0),
                    "constrained": summary.get("constrained", 0),
                    "blocked": summary.get("blocked", 0),
                    "valid_plans": summary.get("valid_plan_subgraphs", 0),
                }
            )
        print(print_table(rows, ["layer", "subgraphs", "safe", "constrained", "blocked", "valid_plans"]))

    def layer_loop(self, model: ModelRef, layer_index: int) -> int:
        layer_data = load_json(model.model_dir / "layers" / f"layer_{layer_index}" / "index.json")
        summary = layer_data.get("summary", {})
        if not summary:
            print(f"Layer/block {layer_index} report not found for {model.model_name}.")
            return 0
        self.show_layer_summary(model, layer_index, summary)
        while True:
            cmd = self.prompt(f"layer[{layer_index}]> ").strip()
            if cmd in {"q", "quit", "exit"}:
                return 0
            if cmd in {"b", "back"}:
                return 0
            if cmd in {"s", "summary"}:
                self.show_layer_summary(model, layer_index, summary)
            elif cmd in {"n", "nodes"}:
                self.list_subgraphs(model, layer_index)
            elif cmd in {"p", "plans"}:
                self.show_layer_plans(model, layer_index)
            elif cmd.startswith("find "):
                self.find_in_layer(model, layer_index, cmd[5:])
            elif cmd.startswith("subgraph "):
                self.subgraph_loop(model, layer_index, cmd.split(maxsplit=1)[1])
            elif cmd in {"a", "analyze"}:
                chosen = self.choose_subgraph(model, layer_index)
                if chosen:
                    self.subgraph_loop(model, layer_index, chosen)
            else:
                print("Commands: s/summary, n/nodes, a/analyze, subgraph <name|number>, p/plans, find <text>, b/back, q")
        return 0

    def show_layer_summary(self, model: ModelRef, layer_index: int, summary: dict[str, Any]) -> None:
        print(f"\nLayer {layer_index}: {model.model_name}")
        print(f"Subgraphs: {summary.get('total_subgraphs', 0)}")
        print(f"ONNX exported: {summary.get('onnx_exported', 0)}")
        print(
            "Safe/constrained/blocked/auxiliary/unknown: "
            f"{summary.get('safe', 0)}/{summary.get('constrained', 0)}/{summary.get('blocked', 0)}/{summary.get('auxiliary', 0)}/{summary.get('unknown', 0)}"
        )
        print(f"Valid plan subgraphs: {summary.get('valid_plan_subgraphs', 0)}")

    def list_subgraphs(self, model: ModelRef, layer_index: int) -> None:
        rows = []
        for item in discover_subgraphs(model.model_dir, layer_index, self.layer_pack_root):
            cls = item.get("classification", {})
            rows.append(
                {
                    "#": item.get("ordinal"),
                    "Abstract node": item.get("display_name"),
                    "Semantic category": item.get("semantic_category"),
                    "Class": cls.get("pruning_class"),
                    "Plan": cls.get("plan_status"),
                    "Validation": cls.get("validation_status"),
                    "ONNX": item.get("onnx_export", {}).get("status"),
                }
            )
        print(print_table(rows, ["#", "Abstract node", "Semantic category", "Class", "Plan", "Validation", "ONNX"]))

    def choose_subgraph(self, model: ModelRef, layer_index: int) -> str | None:
        self.list_subgraphs(model, layer_index)
        value = self.prompt("Select subgraph [number/name/b]: ").strip()
        if value in {"b", "back"}:
            return None
        return value

    def find_record(self, model: ModelRef, layer_index: int, selector: str) -> dict[str, Any] | None:
        subgraphs = discover_subgraphs(model.model_dir, layer_index, self.layer_pack_root)
        if selector.isdigit():
            ordinal = int(selector)
            return next((item for item in subgraphs if int(item.get("ordinal") or -1) == ordinal), None)
        matches = search_subgraphs(subgraphs, selector)
        return matches[0] if matches else None

    def subgraph_loop(self, model: ModelRef, layer_index: int, selector: str) -> None:
        record = self.find_record(model, layer_index, selector)
        if not record:
            print(f"Subgraph not found: {selector}")
            return
        self.show_subgraph_summary(record)
        while True:
            cmd = self.prompt("subgraph> ").strip()
            if cmd in {"b", "back"}:
                return
            if cmd in {"q", "quit", "exit"}:
                raise SystemExit(0)
            if cmd in {"e", "explanation"}:
                self.show_explanation(record)
            elif cmd in {"j", "json"}:
                self.show_subgraph_json_summary(record)
            elif cmd == "ops":
                print(print_table(record.get("primitive_ops", []), ["topological_index", "source_name", "op_type"]))
            elif cmd == "sem":
                print("Op semantics:")
                print(print_table(record.get("local_op_semantics", []), ["source_name", "semantic_kind", "semantic_category", "parameterized", "direct_pruning"]))
                print("Region semantics:")
                print(print_table(record.get("local_region_semantics", []), ["region_name", "semantic_category", "pruning_role"]))
            elif cmd in {"rank", "ranking"}:
                print(print_table(record.get("local_ranking", []), ["candidate_kind", "pruning_class", "rank_score", "confidence", "target_dimension", "reason"]))
            elif cmd == "plan":
                self.show_plan(record)
            elif cmd in {"val", "validation"}:
                self.show_validation(record)
            elif cmd == "onnx":
                self.open_onnx(model, layer_index, record)
            elif cmd == "path":
                self.show_paths(model, layer_index, record)
            else:
                print("Commands: e, j, ops, sem, rank, plan, val, onnx, path, b")

    def show_subgraph_summary(self, record: dict[str, Any]) -> None:
        summary = summarize_subgraph(record)
        print(f"\nSubgraph: {summary['display_name']}")
        print(f"Class: {summary['pruning_class']}")
        print(f"Semantic category: {summary['semantic_category']}")
        print(f"Plan status: {summary['plan_status']}")
        print(f"Validation: {summary['validation_status']}")
        print(f"ONNX: {summary['onnx_status']}")
        print("Commands: e, j, ops, sem, rank, plan, val, onnx, path, b")

    def show_explanation(self, record: dict[str, Any]) -> None:
        path = Path(str(record.get("_explanation_path") or ""))
        text = read_text(path)
        if not text:
            print(record.get("explanation", "No explanation available."))
            return
        lines = text.splitlines()
        print("\n".join(lines[:80]))
        if len(lines) > 80 and not self.scripted:
            if self.prompt("Show full explanation? [y/N] ").lower().startswith("y"):
                print(text)

    def show_subgraph_json_summary(self, record: dict[str, Any]) -> None:
        rows = [{"field": key, "value": value} for key, value in summarize_subgraph(record).items()]
        print(print_table(rows, ["field", "value"]))

    def show_plan(self, record: dict[str, Any]) -> None:
        plans = record.get("local_plans", [])
        if not plans:
            print("No pruning plan attached to this subgraph.")
            return
        for plan in plans:
            print(f"Plan: {plan.get('plan_id')} status={plan.get('plan_status')} kind={plan.get('plan_kind')}")
            print(f"Index set: {plan.get('symbolic_index_set')}")
            print(print_table(plan.get("actions", []), ["action_type", "target_source_name", "target_axis", "dimension"]))

    def show_validation(self, record: dict[str, Any]) -> None:
        validations = record.get("local_validations", [])
        if not validations:
            print("No validation attached to this subgraph.")
            return
        for item in validations:
            print(f"Validation: {item.get('validation_status')} score={item.get('validation_score')}")
            print(f"Failed checks: {', '.join(item.get('failed_checks', [])) or 'none'}")
            print(f"Warning checks: {', '.join(item.get('warning_checks', [])) or 'none'}")

    def open_onnx(self, model: ModelRef, layer_index: int, record: dict[str, Any]) -> None:
        path = find_onnx_path(model.safe_name, layer_index, record.get("node_slug", ""), self.artifact_root, self.fallback_artifact_root)
        if not path:
            print("ONNX subgraph not found.")
            return
        ok, message = open_path(path, no_open=self.no_open, open_command=self.open_command)
        print(("Opened: " if ok and not self.no_open else "Path: ") + message)

    def show_paths(self, model: ModelRef, layer_index: int, record: dict[str, Any]) -> None:
        print(f"analysis.json: {record.get('_analysis_path')}")
        print(f"explanation.md: {record.get('_explanation_path')}")
        onnx = find_onnx_path(model.safe_name, layer_index, record.get("node_slug", ""), self.artifact_root, self.fallback_artifact_root)
        print(f"subgraph.onnx: {onnx or 'missing'}")

    def show_layer_plans(self, model: ModelRef, layer_index: int) -> None:
        rows = []
        for item in discover_subgraphs(model.model_dir, layer_index, self.layer_pack_root):
            if item.get("local_plans"):
                rows.append(
                    {
                        "#": item.get("ordinal"),
                        "subgraph": item.get("display_name"),
                        "plan": item.get("classification", {}).get("plan_status"),
                        "validation": item.get("classification", {}).get("validation_status"),
                    }
                )
        print(print_table(rows, ["#", "subgraph", "plan", "validation"]))

    def find_in_layer(self, model: ModelRef, layer_index: int, text: str) -> None:
        matches = search_subgraphs(discover_subgraphs(model.model_dir, layer_index, self.layer_pack_root), text)
        rows = [{"layer": layer_index, "#": item.get("ordinal"), "subgraph": item.get("display_name"), "category": item.get("semantic_category")} for item in matches]
        print(print_table(rows, ["layer", "#", "subgraph", "category"]))

    def find_in_model(self, model: ModelRef, text: str) -> None:
        rows = []
        for layer in discover_layers(model.model_dir):
            for item in search_subgraphs(discover_subgraphs(model.model_dir, layer["layer_index"], self.layer_pack_root), text):
                rows.append({"layer": layer["layer_index"], "#": item.get("ordinal"), "subgraph": item.get("display_name"), "category": item.get("semantic_category")})
        print(print_table(rows[:80], ["layer", "#", "subgraph", "category"]))

    def show_pipeline(self, data: dict[str, Any]) -> None:
        print(
            """
Static analysis pipeline:

1. Op Semantics: primitive op pruning behavior.
2. Region Pruning Semantics: region-level pruning roles, repairs, blockers.
3. Opportunity Ranking: safe/constrained/blocked candidates and reasons.
4. Plan Synthesis: symbolic actions over MLP/FFN intermediate_dim.
5. Plan Validation: static legality checks over actions, semantics, repairs, and protected hidden paths.
6. Layer/Subgraph Atlas: ordered learner-facing subgraphs and ONNX evidence.
"""
        )
        self.show_model_summary(data)

    def show_ranking_summary(self, data: dict[str, Any]) -> None:
        ranking = data.get("model_summary", {}).get("ranking", {})
        rows = [{"field": key, "value": value} for key, value in ranking.items()]
        print(print_table(rows, ["field", "value"]))

    def show_validation_summary(self, data: dict[str, Any]) -> None:
        summary = validation_summary(data.get("model_summary", {}).get("plan_validation", {}))
        rows = [{"field": key, "value": value} for key, value in summary.items()]
        print(print_table(rows, ["field", "value"]))

    def open_report(self, model: ModelRef) -> None:
        ok, message = open_path(model.model_dir / "index.md", no_open=self.no_open, open_command=self.open_command)
        print(("Opened: " if ok and not self.no_open else "Path: ") + message)

    def show_compare(self) -> None:
        data = load_json(self.root / "reports" / "static_coverage_study" / "index.json")
        models = data.get("models", [])
        rows = []
        for model in models:
            artifacts = model.get("artifacts", {})
            full_report = artifacts.get("full_model_report", {})
            ranking = artifacts.get("ranking", {})
            plans = artifacts.get("plans", {})
            validation = artifacts.get("validation", {})
            rows.append(
                {
                    "Model": model.get("model_name"),
                    "Layers": full_report.get("layers", 0),
                    "Subgraphs": full_report.get("subgraphs", 0),
                    "Safe": ranking.get("safe", ""),
                    "Plans": plans.get("plans", ""),
                    "Valid": validation.get("valid_plans", full_report.get("valid_plans", 0)),
                    "Status": model.get("final_status"),
                }
            )
        print(print_table(rows, ["Model", "Layers", "Subgraphs", "Safe", "Plans", "Valid", "Status"]))
        print(f"Report: {self.root / 'reports' / 'static_coverage_study' / 'index.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive read-only explorer for static analysis reports.")
    parser.add_argument("--model")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--report-root", default="reports/model_analysis_reports")
    parser.add_argument("--layer-pack-root", default="reports/layer_subgraph_validation")
    parser.add_argument("--artifact-root", default="artifacts/model_analysis_subgraphs")
    parser.add_argument("--open-command")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scripted", help="Semicolon-separated command list for smoke testing.")
    return parser.parse_args()


def main() -> int:
    return AnalysisExplorer(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
