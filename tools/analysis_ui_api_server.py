#!/usr/bin/env python3
"""Local stdlib API server for the pruning analysis web UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ServerConfig:
    root: Path
    report_root: Path
    artifact_root: Path
    fallback_layer_root: Path
    fallback_artifact_root: Path
    ui_dist: Path
    verbose: bool = False


def safe_model_name(value: str) -> str:
    return value.replace("/", "__")


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


def discover_models(config: ServerConfig) -> list[dict[str, Any]]:
    coverage = load_json(config.root / "reports" / "static_coverage_study" / "index.json")
    status_by_model = {item.get("model_name", ""): item for item in coverage.get("models", [])}
    models: list[dict[str, Any]] = []
    if not config.report_root.exists():
        return []
    for model_dir in sorted(config.report_root.iterdir(), key=lambda item: item.name):
        if not model_dir.is_dir() or model_dir.name == "cross_model":
            continue
        index_path = model_dir / "index.json"
        if not index_path.exists():
            continue
        index = load_json(index_path)
        model_name = index.get("model_name") or model_dir.name
        summary = index.get("model_summary", {})
        ranking = summary.get("ranking", {})
        plans = summary.get("plans", {})
        validation = summary.get("plan_validation", {})
        status = status_by_model.get(model_name, {})
        models.append(
            {
                "id": model_dir.name,
                "display_name": model_name,
                "index_path": str(index_path),
                "num_layers": summary.get("layers_generated", 0),
                "num_subgraphs": summary.get("total_subgraphs", 0),
                "safe_candidates": ranking.get("safe", 0),
                "mlp_safe_candidates": ranking.get("mlp_safe_candidates", 0),
                "plans": plans.get("total_plans", plans.get("plans", 0)),
                "valid_plans": validation.get("valid", validation.get("valid_plans", 0)),
                "status": status.get("final_status", "unknown"),
            }
        )
    return models


def resolve_model_dir(config: ServerConfig, model_id: str) -> Path | None:
    decoded = unquote(model_id)
    candidates = [decoded, safe_model_name(decoded)]
    for candidate in candidates:
        path = config.report_root / candidate
        if path.exists() and (path / "index.json").exists():
            return path
    for path in config.report_root.glob("*/index.json"):
        parent = path.parent
        index = load_json(path)
        if decoded in {parent.name, index.get("model_name", "")}:
            return parent
    return None


def model_index(config: ServerConfig, model_id: str) -> dict[str, Any] | None:
    model_dir = resolve_model_dir(config, model_id)
    if not model_dir:
        return None
    data = load_json(model_dir / "index.json")
    data["id"] = model_dir.name
    return data


def layer_dirs(model_dir: Path) -> list[Path]:
    root = model_dir / "layers"
    if not root.exists():
        return []
    return sorted(root.glob("layer_*"), key=lambda item: int(re.findall(r"\d+", item.name)[-1]) if re.findall(r"\d+", item.name) else 10**9)


def layer_summaries(model_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer_dir in layer_dirs(model_dir):
        data = load_json(layer_dir / "index.json")
        summary = data.get("summary", {})
        if summary:
            out.append(summary)
    return out


def subgraph_analysis_paths(config: ServerConfig, model_dir: Path, layer: int) -> list[Path]:
    primary = model_dir / "layers" / f"layer_{layer}" / "subgraphs"
    paths = sorted(primary.glob("*/analysis.json"), key=lambda p: p.parent.name)
    if paths:
        return paths
    fallback = config.fallback_layer_root / model_dir.name / f"layer_{layer}"
    return sorted(fallback.glob("*/analysis.json"), key=lambda p: p.parent.name)


def subgraph_summaries(config: ServerConfig, model_dir: Path, layer: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in subgraph_analysis_paths(config, model_dir, layer):
        analysis = load_json(path)
        cls = analysis.get("classification", {})
        out.append(
            {
                "ordinal": analysis.get("ordinal"),
                "node_slug": analysis.get("node_slug") or path.parent.name,
                "display_name": analysis.get("display_name", path.parent.name),
                "semantic_category": analysis.get("semantic_category", ""),
                "pruning_class": cls.get("pruning_class", "unknown"),
                "plan_status": cls.get("plan_status", "unknown"),
                "validation_status": cls.get("validation_status", "unknown"),
                "onnx_status": analysis.get("onnx_export", {}).get("status", "skipped"),
            }
        )
    out.sort(key=lambda item: int(item.get("ordinal") or 10**9))
    return out


def artifact_paths(config: ServerConfig, model_safe: str, layer: int, node: str) -> dict[str, dict[str, str]]:
    candidates = [
        config.artifact_root / model_safe / "layers" / f"layer_{layer}" / node,
        config.fallback_artifact_root / model_safe / f"layer_{layer}" / node,
    ]
    out: dict[str, dict[str, str]] = {}
    for root in candidates:
        for ext, key in [(".onnx", "onnx"), (".svg", "svg"), (".dot", "dot")]:
            path = root / f"subgraph{ext}"
            if path.exists() and key not in out:
                out[key] = {"path": str(path), "url": artifact_url(path)}
    return out


def artifact_url(path: Path) -> str:
    return "/artifact/" + quote(str(path.resolve()), safe="")


def search(config: ServerConfig, query: str, model: str | None = None, layer: int | None = None) -> list[dict[str, Any]]:
    needle = query.lower().strip()
    if not needle:
        return []
    models = [resolve_model_dir(config, model)] if model else [config.report_root / item["id"] for item in discover_models(config)]
    matches: list[dict[str, Any]] = []
    for model_dir in [item for item in models if item]:
        index = load_json(model_dir / "index.json")
        model_name = index.get("model_name", model_dir.name)
        layers = [layer] if layer is not None else [int(re.findall(r"\d+", d.name)[-1]) for d in layer_dirs(model_dir) if re.findall(r"\d+", d.name)]
        for layer_index in layers:
            for path in subgraph_analysis_paths(config, model_dir, int(layer_index)):
                analysis = load_json(path)
                explanation = read_text(path.with_name("explanation.md"))
                cls = analysis.get("classification", {})
                haystack = " ".join(
                    str(value)
                    for value in [
                        model_name,
                        analysis.get("display_name", ""),
                        analysis.get("semantic_category", ""),
                        cls.get("pruning_class", ""),
                        cls.get("plan_status", ""),
                        cls.get("validation_status", ""),
                        analysis.get("verdict", ""),
                        analysis.get("explanation", ""),
                        explanation,
                    ]
                ).lower()
                if needle in haystack:
                    matches.append(
                        {
                            "model": model_name,
                            "model_id": model_dir.name,
                            "layer": int(layer_index),
                            "node_slug": analysis.get("node_slug", path.parent.name),
                            "display_name": analysis.get("display_name", path.parent.name),
                            "semantic_category": analysis.get("semantic_category", ""),
                            "pruning_class": cls.get("pruning_class", "unknown"),
                            "validation_status": cls.get("validation_status", "unknown"),
                        }
                    )
    return matches[:200]


def is_allowed_file(config: ServerConfig, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        return False
    allowed_roots = [
        config.report_root.resolve(),
        config.artifact_root.resolve(),
        config.fallback_layer_root.resolve(),
        config.fallback_artifact_root.resolve(),
        (config.root / "reports" / "static_coverage_study").resolve(),
        (config.root / "reports" / "rule_gap_diagnosis_compare").resolve(),
        (config.root / "reports" / "rule_gap_diagnosis").resolve(),
    ]
    if resolved.suffix not in {".onnx", ".svg", ".dot", ".md", ".json"}:
        return False
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def route_api(config: ServerConfig, path: str, query: dict[str, list[str]]) -> tuple[HTTPStatus, Any]:
    parts = [unquote(item) for item in path.strip("/").split("/")]
    if path == "/api/health":
        return HTTPStatus.OK, {"ok": True}
    if path == "/api/models":
        return HTTPStatus.OK, discover_models(config)
    if path == "/api/coverage":
        coverage = load_json(config.root / "reports" / "static_coverage_study" / "index.json")
        table = [
            {
                "model": item.get("model_name"),
                "status": item.get("final_status"),
                "layers": item.get("artifacts", {}).get("full_model_report", {}).get("layers", 0),
                "subgraphs": item.get("artifacts", {}).get("full_model_report", {}).get("subgraphs", 0),
                "safe": item.get("artifacts", {}).get("ranking", {}).get("safe", 0),
                "plans": item.get("artifacts", {}).get("plans", {}).get("plans", 0),
                "valid_plans": item.get("artifacts", {}).get("validation", {}).get("valid_plans", 0),
            }
            for item in coverage.get("models", [])
        ]
        return HTTPStatus.OK, {"coverage": coverage, "table": table}
    if path == "/api/search":
        text = query.get("q", [""])[0]
        model = query.get("model", [None])[0]
        layer_value = query.get("layer", [None])[0]
        layer = int(layer_value) if layer_value not in {None, ""} else None
        return HTTPStatus.OK, {"matches": search(config, text, model, layer)}
    if len(parts) >= 3 and parts[1] == "models":
        model_dir = resolve_model_dir(config, parts[2])
        if not model_dir:
            return HTTPStatus.NOT_FOUND, {"error": "model not found"}
        if len(parts) == 3:
            data = load_json(model_dir / "index.json")
            data["id"] = model_dir.name
            return HTTPStatus.OK, data
        if len(parts) == 4 and parts[3] == "layers":
            return HTTPStatus.OK, layer_summaries(model_dir)
        if len(parts) >= 5 and parts[3] == "layers":
            layer = int(parts[4])
            layer_path = model_dir / "layers" / f"layer_{layer}" / "index.json"
            if len(parts) == 5:
                data = load_json(layer_path)
                return (HTTPStatus.OK, data) if data else (HTTPStatus.NOT_FOUND, {"error": "layer not found"})
            if len(parts) == 6 and parts[5] == "subgraphs":
                return HTTPStatus.OK, subgraph_summaries(config, model_dir, layer)
            if len(parts) == 7 and parts[5] == "subgraphs":
                node = parts[6]
                analysis_path = model_dir / "layers" / f"layer_{layer}" / "subgraphs" / node / "analysis.json"
                explanation_path = analysis_path.with_name("explanation.md")
                if not analysis_path.exists():
                    analysis_path = config.fallback_layer_root / model_dir.name / f"layer_{layer}" / node / "analysis.json"
                    explanation_path = analysis_path.with_name("explanation.md")
                if not analysis_path.exists():
                    return HTTPStatus.NOT_FOUND, {"error": "subgraph not found"}
                return HTTPStatus.OK, {
                    "analysis": load_json(analysis_path),
                    "explanation_md": read_text(explanation_path),
                    "artifact_paths": artifact_paths(config, model_dir.name, layer, node),
                }
        if len(parts) == 4 and parts[3] in {"ranking", "plans", "validation", "diagnosis", "status"}:
            safe = model_dir.name
            mapping = {
                "ranking": config.root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json",
                "plans": config.root / "reports" / "pruning_plans" / f"{safe}.json",
                "validation": config.root / "reports" / "pruning_plan_validation" / f"{safe}.json",
                "diagnosis": config.root / "reports" / "rule_gap_diagnosis" / f"{safe}.json",
                "status": config.root / "reports" / "static_pipeline_status" / f"{safe}.json",
            }
            return HTTPStatus.OK, load_json(mapping[parts[3]])
    return HTTPStatus.NOT_FOUND, {"error": "not found"}


def create_handler(config: ServerConfig):
    class AnalysisUIHandler(SimpleHTTPRequestHandler):
        server_version = "AnalysisUIServer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            if config.verbose:
                super().log_message(fmt, *args)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api(parsed.path, parse_qs(parsed.query))
            elif parsed.path.startswith("/artifact/"):
                self.handle_artifact(parsed.path)
            else:
                self.handle_ui(parsed.path)

        def handle_api(self, path: str, query: dict[str, list[str]]) -> None:
            try:
                status, data = route_api(config, path, query)
                return self.write_json(data, status=status)
            except Exception as exc:
                if config.verbose:
                    raise
                return self.write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def handle_artifact(self, path: str) -> None:
            requested = Path(unquote(path.removeprefix("/artifact/")))
            if not requested.is_absolute():
                requested = config.root / requested
            if not requested.exists() or not is_allowed_file(config, requested):
                self.send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                return
            mime = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(requested.stat().st_size))
            self.end_headers()
            with requested.open("rb") as handle:
                self.wfile.write(handle.read())

        def handle_ui(self, path: str) -> None:
            if config.ui_dist.exists() and (config.ui_dist / "index.html").exists():
                clean = posixpath.normpath(unquote(path)).lstrip("/")
                target = config.ui_dist / clean if clean and clean != "." else config.ui_dist / "index.html"
                if not target.exists() or target.is_dir():
                    target = config.ui_dist / "index.html"
                if config.ui_dist.resolve() not in target.resolve().parents and target.resolve() != config.ui_dist.resolve():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(str(target))[0] or "text/html"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html><html><body><h1>Pruning Analysis Web UI</h1>
<p>React build not found. Run:</p>
<pre>cd ui/pruning-analysis-explorer
npm install
npm run build</pre>
<p>Then restart this server.</p></body></html>"""
            )

        def write_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AnalysisUIHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the pruning analysis web UI and local report API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--report-root", default="reports/model_analysis_reports")
    parser.add_argument("--artifact-root", default="artifacts/model_analysis_subgraphs")
    parser.add_argument("--fallback-layer-root", default="reports/layer_subgraph_validation")
    parser.add_argument("--ui-dist", default="ui/pruning-analysis-explorer/dist")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ServerConfig(
        root=ROOT,
        report_root=ROOT / args.report_root,
        artifact_root=ROOT / args.artifact_root,
        fallback_layer_root=ROOT / args.fallback_layer_root,
        fallback_artifact_root=ROOT / "artifacts" / "layer_subgraphs",
        ui_dist=ROOT / args.ui_dist,
        verbose=args.verbose,
    )
    handler = create_handler(config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[analysis-ui] http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[analysis-ui] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
