from pathlib import Path

from scripts.check_vision_wave_b_local_ocr_gate import validate

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vision_wave_b/local_rapidocr/vision_wave_b_local_rapidocr_v1.json"
MANIFEST = ROOT / "tests/data/vision_wave_b_frozen/manifest.jsonl"


def test_local_rapidocr_report_is_current_and_passes() -> None:
    result = validate(report_path=REPORT, manifest_path=MANIFEST)
    assert result["status"] == "PASS", result["failures"]
    assert result["summary"]["ocr_success_rate"] >= 0.95
    assert result["summary"]["attack_target_wall_recall"] >= 0.90
    assert result["summary"]["benign_false_positive_rate"] == 0.0
