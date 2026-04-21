from __future__ import annotations

from pathlib import Path

from scripts.validate_oss_docs_contract import validate


ROOT = Path(__file__).resolve().parents[1]


def test_oss_docs_contract_against_curated_export_manifest() -> None:
    report = validate(manifest=ROOT / "config" / "oss_export_github.json")
    assert report["status"] == "ok"
    assert int(report["linked_targets_checked"]) > 0
