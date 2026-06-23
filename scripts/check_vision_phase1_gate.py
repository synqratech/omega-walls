#!/usr/bin/env python3
"""Validate a frozen Phase 1 vision gate artifact against current code and data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vision_phase1_frozen import KEY_IMPLEMENTATION_FILES, SUITE_VERSION  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(*, report_path: Path, manifest_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if str(report.get("suite_version", "")) != SUITE_VERSION:
        failures.append("suite_version_mismatch")
    if str(report.get("status", "")) != "PASS":
        failures.append("report_status_not_pass")
    if str(report.get("manifest_sha256", "")) != _sha(manifest_path):
        failures.append("manifest_sha256_mismatch")
    expected_impl = dict(report.get("implementation_hashes", {}) or {})
    for rel in KEY_IMPLEMENTATION_FILES:
        current = _sha(ROOT / rel)
        if expected_impl.get(rel) != current:
            failures.append(f"implementation_hash_mismatch:{rel}")
    summary = dict(report.get("summary", {}) or {})
    if float(summary.get("attack_contract_hit_rate", 0.0)) < 1.0:
        failures.append("attack_contract_hit_rate_below_1")
    if float(summary.get("benign_false_positive_rate", 1.0)) != 0.0:
        failures.append("benign_false_positive_rate_nonzero")
    if int(summary.get("contract_failures", 1)) != 0:
        failures.append("contract_failures_nonzero")
    if int(summary.get("raw_boundary_findings", 1)) != 0:
        failures.append("raw_boundary_findings_nonzero")
    if float(summary.get("vision_status_active_rate", 0.0)) < 1.0:
        failures.append("vision_status_active_rate_below_1")
    return {
        "status": "PASS" if not failures else "FAIL",
        "report": str(report_path),
        "manifest": str(manifest_path),
        "failures": failures,
        "summary": summary,
        "quality_claim": report.get("quality_claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check frozen Phase 1 vision artifact")
    parser.add_argument("--report", default="artifacts/vision_phase1/frozen/vision_phase1_frozen_v1.json")
    parser.add_argument("--manifest", default="tests/data/vision_phase1_frozen/manifest.jsonl")
    args = parser.parse_args()
    report = Path(args.report)
    manifest = Path(args.manifest)
    if not report.is_absolute():
        report = ROOT / report
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    result = validate(report_path=report, manifest_path=manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
