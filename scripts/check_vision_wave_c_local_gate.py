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
from scripts.eval_vision_wave_c_local import KEY_IMPLEMENTATION_FILES, SUITE_VERSION


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

    if report.get("profile") != "prod_vision":
        failures.append("profile_mismatch")

    cfg = load_resolved_config(profile="prod_vision").resolved
    expected = {
        key: float(value)
        for key, value in cfg["vision_wave_c"]["local_gate"]["gates"].items()
    }
    if dict(report.get("thresholds", {}) or {}) != expected:
        failures.append("thresholds_mismatch")
    summary = dict(report.get("summary", {}) or {})
    comparisons = {
        "extraction_success_rate": (">=", "extraction_success_rate_min"),
        "semantic_success_rate": (">=", "semantic_success_rate_min"),
        "attack_target_wall_recall": (">=", "attack_target_wall_recall_min"),
        "benign_false_positive_rate": ("<=", "benign_false_positive_rate_max"),
        "local_egress_trace_rate": (">=", "local_egress_trace_rate_min"),
        "data_residency_trace_rate": (">=", "data_residency_trace_rate_min"),
        "multi_image_packet_rate": (">=", "multi_image_packet_rate_min"),
        "raw_media_boundary_findings": ("<=", "raw_media_boundary_findings_max"),
    }
    for metric, (op, threshold_key) in comparisons.items():
        value = float(summary.get(metric, -1 if op == ">=" else 1e18))
        threshold = expected[threshold_key]
        if (op == ">=" and value < threshold) or (op == "<=" and value > threshold):
            failures.append(f"{metric}_gate_failed")
    if not all(bool(value) for value in dict(report.get("gates", {}) or {}).values()):
        failures.append("embedded_gate_failure")
    provider = dict(report.get("provider", {}) or {})
    if provider.get("type") != "local_vision" or provider.get("backend") != "ocr_pi0":
        failures.append("unexpected_provider")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "summary": summary,
        "thresholds": expected,
        "quality_claim": report.get("quality_claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", default="artifacts/vision_wave_c/local/vision_wave_c_local_v1.json"
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
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
