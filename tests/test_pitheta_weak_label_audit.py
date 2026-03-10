from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from omega.eval.pitheta_weak_label_audit import evaluate_weak_label_audit, write_weak_label_audit_artifacts


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


class _FakeProjector:
    def project(self, item):
        text = str(item.text).lower()
        if "attack" in text:
            v = np.array([0.8, 0.0, 0.0, 0.0], dtype=float)
            pol = [1, 0, 0, 0]
        else:
            v = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
            pol = [0, 0, 0, 0]
        return SimpleNamespace(v=v, evidence=SimpleNamespace(polarity=pol))


def test_weak_label_audit_summary_and_sampling():
    rows = [
        {
            "sample_id": "r1",
            "text": "attack payload",
            "wall_labels": [1, 0, 0, 0],
            "polarity": [1, 0, 0, 0],
            "source": "s1",
            "lang": "en",
            "split": "train",
            "label_quality": "weak",
            "is_attack": 1,
        },
        {
            "sample_id": "r2",
            "text": "benign content",
            "wall_labels": [0, 0, 0, 0],
            "polarity": [0, 0, 0, 0],
            "source": "s1",
            "lang": "en",
            "split": "train",
            "label_quality": "silver",
            "is_attack": 0,
        },
        {
            "sample_id": "r3",
            "text": "attack other",
            "wall_labels": [1, 0, 0, 0],
            "polarity": [1, 0, 0, 0],
            "source": "s2",
            "lang": "en",
            "split": "train",
            "label_quality": "gold",
            "is_attack": 1,
        },
    ]
    projector = _FakeProjector()
    summary, review_rows = evaluate_weak_label_audit(
        projector=projector,  # type: ignore[arg-type]
        rows=rows,
        sample_size=2,
        seed=41,
        include_qualities=["weak", "silver"],
    )
    assert summary["sampled_count"] == 2
    assert set(summary["label_quality_counts"].keys()) <= {"weak", "silver"}
    assert len(review_rows) == 2
    assert 0.0 <= float(summary["agreement"]["exact_wall_rows_rate"]) <= 1.0
    assert 0.0 <= float(summary["agreement"]["exact_polarity_rows_rate"]) <= 1.0


def test_weak_label_audit_artifacts_written():
    report = {
        "status": "ok",
        "sampled_count": 1,
        "ascii_count": 1,
        "non_ascii_count": 0,
        "agreement": {
            "exact_wall_rows_rate": 1.0,
            "exact_polarity_rows_rate": 1.0,
            "any_mismatch_rows_rate": 0.0,
        },
    }
    review = [
        {
            "sample_id": "r1",
            "text": "attack payload",
            "priority_review": True,
            "mismatch_any": False,
        }
    ]
    out = _mk_local_tmp("weak-label-audit")
    artifacts = write_weak_label_audit_artifacts(out_dir=out.as_posix(), report=report, review_rows=review)
    assert Path(artifacts["audit_report"]).exists()
    assert Path(artifacts["review_samples"]).exists()
    assert Path(artifacts["priority_review_samples"]).exists()
    payload = json.loads(Path(artifacts["audit_report"]).read_text(encoding="utf-8"))
    assert payload["status"] == "ok"

