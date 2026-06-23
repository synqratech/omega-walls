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

from scripts.eval_vision_wave_b_frozen import KEY_IMPLEMENTATION_FILES, SUITE_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(*, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if report.get("suite_version") != SUITE_VERSION:
        failures.append("suite_version_mismatch")
    if report.get("status") != "PASS":
        failures.append("report_status_not_pass")
    if report.get("manifest_sha256") != _sha(manifest_path):
        failures.append("manifest_sha256_mismatch")
    hashes = dict(report.get("implementation_hashes", {}) or {})
    for rel in KEY_IMPLEMENTATION_FILES:
        if hashes.get(rel) != _sha(ROOT / rel):
            failures.append(f"implementation_hash_mismatch:{rel}")
    summary = dict(report.get("summary", {}) or {})
    modes = dict(summary.get("modes", {}) or {})
    fused = dict(modes.get("region_plus_ocr", {}) or {})
    if float(fused.get("attack_recall", 0.0)) != 1.0:
        failures.append("region_plus_ocr_recall_not_1")
    if float(fused.get("benign_false_positive_rate", 1.0)) != 0.0:
        failures.append("region_plus_ocr_fp_nonzero")
    if int(summary.get("ocr_quality_failures", 1)) != 0:
        failures.append("ocr_quality_failures_nonzero")
    if int(summary.get("spatial_crop_failures", 1)) != 0:
        failures.append("spatial_crop_failures_nonzero")
    if float(summary.get("region_rescue_rate", 0.0)) != 1.0:
        failures.append("region_rescue_rate_not_1")
    if int(summary.get("raw_media_boundary_findings", 1)) != 0:
        failures.append("raw_media_boundary_findings_nonzero")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "summary": summary,
        "quality_claim": report.get("quality_claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/vision_wave_b/frozen/vision_wave_b_frozen_v1.json")
    parser.add_argument("--manifest", default="tests/data/vision_wave_b_frozen/manifest.jsonl")
    args = parser.parse_args()
    report = Path(args.report)
    manifest = Path(args.manifest)
    if not report.is_absolute(): report = ROOT / report
    if not manifest.is_absolute(): manifest = ROOT / manifest
    result = validate(report_path=report, manifest_path=manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
