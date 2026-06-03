"""Evidence-unit extraction for strict MLIR-derived ONNX annotations."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvidenceUnit:
    node_name: str
    unit_path: str
    covered_nodes: list[str]
    extraction_status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_evidence_units(input_onnx: str | Path, output_root: str | Path, *, max_nodes: int | None = None) -> tuple[list[EvidenceUnit], list[str]]:
    """Create MLIR evidence units.

    Whole-input mode is intentionally preferred for local subgraphs: it keeps the
    evidence unit faithful and avoids producing semantic claims from ONNX names.
    The resulting MLIR evidence may be mapped back to member nodes, but the
    grouping itself is not treated as proof.
    """
    warnings: list[str] = []
    source = Path(input_onnx)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import onnx  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when onnx is absent
        raise RuntimeError("onnx is required to extract evidence units") from exc

    model = onnx.load(str(source))
    nodes = list(model.graph.node)
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
        warnings.append(f"max_nodes={max_nodes} limits annotation coverage for this run")
    covered_nodes = [node.name or f"{node.op_type}_{index}" for index, node in enumerate(nodes)]
    unit_dir = output / "whole_graph"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "unit.onnx"
    shutil.copyfile(source, unit_path)
    unit = EvidenceUnit(
        node_name="whole_graph",
        unit_path=str(unit_path),
        covered_nodes=covered_nodes,
        extraction_status="whole_input",
        warnings=[
            "Whole-input local subgraph used as the MLIR evidence unit; node names/op types are display metadata only."
        ],
    )
    (output / "evidence_unit_index.json").write_text(
        json.dumps(
            {
                "input_onnx": str(source),
                "units": [unit.to_dict()],
                "warnings": warnings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [unit], warnings
