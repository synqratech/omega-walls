from __future__ import annotations

from pathlib import Path

from scripts.check_vision_wave_b_gate import validate

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vision_wave_b/frozen/vision_wave_b_frozen_v1.json"
MANIFEST = ROOT / "tests/data/vision_wave_b_frozen/manifest.jsonl"


def test_frozen_wave_b_report_is_current_and_passes() -> None:
    result = validate(report_path=REPORT, manifest_path=MANIFEST)
    assert result["status"] == "PASS", result["failures"]
    summary = result["summary"]
    assert summary["modes"]["region_plus_ocr"]["attack_recall"] == 1.0
    assert summary["modes"]["region_plus_ocr"]["benign_false_positive_rate"] == 0.0
    assert summary["raw_media_boundary_findings"] == 0
