"""Synthetic proof fixtures for final-report tests."""

from __future__ import annotations

import json
from pathlib import Path


MODELS = (
    ("bert-base-uncased", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("distilbert-base-uncased", 6, 12, 12, 6, 6, 6, 6, 12, 0),
    ("facebook/opt-125m", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("gpt2", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("google/vit-base-patch16-224", 12, 24, 24, 12, 12, 12, 12, 24, 0),
)


def write_all_model_proof(root: Path) -> Path:
    path = root / "reports/all_model_plan_proof/index.json"
    path.parent.mkdir(parents=True)
    models = []
    for name, layers, expected, proven, ffn_expected, ffn_proven, attention_expected, attention_proven, native, fallback in MODELS:
        models.append(
            {
                "model_name": name,
                "layer_count": layers,
                "summary": {
                    "total_expected": expected,
                    "total_proven": proven,
                    "ffn_expected": ffn_expected,
                    "ffn_proven": ffn_proven,
                    "attention_expected": attention_expected,
                    "attention_proven": attention_proven,
                    "native_evidence_count": native,
                    "fallback_count": fallback,
                    "unsupported_count": 0,
                    "partial_count": 0,
                    "missing_count": 0,
                    "failed_count": 0,
                },
                "final_verdict": "complete_plan_proof",
                "notes": "synthetic complete proof",
            }
        )
    path.write_text(
        json.dumps(
            {
                "models": models,
                "aggregate": {
                    "total_expected": 108,
                    "total_proven": 108,
                    "native_evidence_count": 108,
                    "access_evidence_count": 0,
                    "fallback_count": 0,
                    "unsupported_count": 0,
                    "partial_count": 0,
                    "missing_count": 0,
                    "failed_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
