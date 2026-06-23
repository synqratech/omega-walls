#!/usr/bin/env python3
# ruff: noqa: E402
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
from scripts.eval_vision_wave_c_frozen import KEY_IMPLEMENTATION_FILES, SUITE_VERSION


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

    if report.get("tested_profile") != "prod_vision":
        failures.append("tested_profile_mismatch")

    cfg = load_resolved_config(profile="prod_vision").resolved
    thresholds = {
        key: float(value) for key, value in cfg["vision_wave_c"]["gates"].items()
    }
    summary = dict(report.get("summary", {}) or {})
    checks = {
        "exact_asset_count_rate": (">=", "exact_asset_count_rate_min"),
        "provenance_accuracy": (">=", "provenance_accuracy_min"),
        "asset_sha_integrity_rate": (">=", "asset_sha_integrity_rate_min"),
        "multi_image_packet_rate": (">=", "multi_image_packet_rate_min"),
        "remote_reference_ignored_rate": (">=", "remote_reference_ignored_rate_min"),
        "provider_capability_parity_rate": (
            ">=",
            "provider_capability_parity_rate_min",
        ),
        "raw_media_boundary_findings": ("<=", "raw_media_boundary_findings_max"),
    }
    for metric, (op, threshold_key) in checks.items():
        value = float(summary.get(metric, -1 if op == ">=" else 1e18))
        threshold = thresholds[threshold_key]
        if (op == ">=" and value < threshold) or (op == "<=" and value > threshold):
            failures.append(f"{metric}_gate_failed")
    passed = int(summary.get("egress_policy_cases_passed", 0))
    total = int(summary.get("egress_policy_cases_total", -1))
    egress_rate = passed / total if total > 0 else 0.0
    if egress_rate < thresholds["egress_policy_pass_rate_min"]:
        failures.append("egress_policy_cases_failed")
    if summary.get("prod_projector_mode") != "hybrid_api":
        failures.append("prod_vision_not_hybrid_api")
    if summary.get("prod_provider") != "local_vision":
        failures.append("prod_vision_not_local_vision")
    if not all(bool(value) for value in dict(report.get("gates", {}) or {}).values()):
        failures.append("embedded_gate_failure")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "summary": summary,
        "thresholds": thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="artifacts/vision_wave_c/frozen/vision_wave_c_frozen_v1.json",
    )
    parser.add_argument(
        "--manifest", default="tests/data/vision_wave_c_frozen/manifest.jsonl"
    )
    args = parser.parse_args()
    report = Path(args.report)
    manifest = Path(args.manifest)
    if not report.is_absolute():
        report = ROOT / report
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    result = validate(report_path=report, manifest_path=manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
