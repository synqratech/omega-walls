from __future__ import annotations

import json
from pathlib import Path

from scripts.check_vision_phase1_gate import validate
from scripts.eval_vision_phase1_frozen import evaluate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "data" / "vision_phase1_frozen" / "manifest.jsonl"
FROZEN_REPORT = ROOT / "artifacts" / "vision_phase1" / "frozen" / "vision_phase1_frozen_v1.json"


def test_frozen_vision_phase1_report_is_current_and_passes_release_gate() -> None:
    result = validate(report_path=FROZEN_REPORT, manifest_path=MANIFEST)
    assert result["status"] == "PASS", result["failures"]
    assert result["summary"]["attack_contract_hit_rate"] == 1.0
    assert result["summary"]["benign_false_positive_rate"] == 0.0
    assert result["summary"]["raw_boundary_findings"] == 0


def test_frozen_vision_phase1_eval_is_reproducible(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    report = evaluate(manifest_path=MANIFEST, output_path=out)
    assert report["status"] == "PASS"
    assert report["dataset_sha256"] == json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))["dataset_sha256"]
    result = validate(report_path=out, manifest_path=MANIFEST)
    assert result["status"] == "PASS", result["failures"]
