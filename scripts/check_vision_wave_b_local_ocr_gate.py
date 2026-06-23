#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from scripts.eval_vision_wave_b_local_ocr import KEY_IMPLEMENTATION_FILES, SUITE_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(*, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("suite_version") != SUITE_VERSION: failures.append("suite_version_mismatch")
    if report.get("status") != "PASS": failures.append("report_status_not_pass")
    if report.get("manifest_sha256") != _sha(manifest_path): failures.append("manifest_sha256_mismatch")
    hashes = dict(report.get("implementation_hashes", {}) or {})
    for rel in KEY_IMPLEMENTATION_FILES:
        if hashes.get(rel) != _sha(ROOT / rel): failures.append(f"implementation_hash_mismatch:{rel}")
    summary = dict(report.get("summary", {}) or {})
    cfg = load_resolved_config(profile="prod_vision").resolved
    expected_thresholds = {
        key: float(value)
        for key, value in cfg["vision_wave_b"]["local_rapidocr_gate"]["gates"].items()
    }
    if dict(report.get("thresholds", {}) or {}) != expected_thresholds:
        failures.append("thresholds_mismatch")
    if float(summary.get("ocr_success_rate", 0.0)) < expected_thresholds["ocr_success_rate_min"]:
        failures.append("ocr_success_rate_below_threshold")
    if float(summary.get("text_similarity_ge_0_80_rate", 0.0)) < expected_thresholds["text_similarity_rate_min"]:
        failures.append("text_similarity_rate_below_threshold")
    if float(summary.get("attack_target_wall_recall", 0.0)) < expected_thresholds["attack_target_wall_recall_min"]:
        failures.append("attack_recall_below_threshold")
    if float(summary.get("benign_false_positive_rate", 1.0)) > expected_thresholds["benign_false_positive_rate_max"]:
        failures.append("benign_fp_above_threshold")
    if float(summary.get("quality_usable_rate", 0.0)) < expected_thresholds["quality_usable_rate_min"]:
        failures.append("quality_usable_rate_below_threshold")
    if not all(bool(value) for value in dict(report.get("gates", {}) or {}).values()):
        failures.append("embedded_gate_failure")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "summary": summary,
        "thresholds": expected_thresholds,
        "quality_claim": report.get("quality_claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/vision_wave_b/local_rapidocr/vision_wave_b_local_rapidocr_v1.json")
    parser.add_argument("--manifest", default="tests/data/vision_wave_b_frozen/manifest.jsonl")
    args = parser.parse_args()
    report = Path(args.report); manifest = Path(args.manifest)
    if not report.is_absolute(): report = ROOT / report
    if not manifest.is_absolute(): manifest = ROOT / manifest
    result = validate(report_path=report, manifest_path=manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
