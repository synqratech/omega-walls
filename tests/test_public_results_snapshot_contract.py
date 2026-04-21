from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_public_results_snapshot_paths_and_model_scope() -> None:
    payload = json.loads(_read("docs/public_results_snapshot.json"))
    runs = payload.get("runs", [])
    assert len(runs) >= 2
    for run in runs:
        artifacts = run.get("artifacts_root")
        assert artifacts
        assert (ROOT / str(artifacts)).exists()

    run_b = next(x for x in runs if x["name"] == "frozen_run_b_stateful_vs_baseline_d")
    assert run_b.get("baseline_d_model") == "gpt-5.4-mini"


def test_readme_references_snapshot_and_run_ids() -> None:
    snapshot = json.loads(_read("docs/public_results_snapshot.json"))
    readme = _read("README.md")

    assert "<!-- RESULTS_SNAPSHOT:START -->" in readme
    assert "<!-- RESULTS_SNAPSHOT:END -->" in readme
    assert "docs/public_results_snapshot.json" in readme

    for run in snapshot["runs"]:
        assert str(run["run_id"]) in readme

    assert "gpt-5.4-mini" in readme
    assert "not claimed" in readme.lower()


def test_content_quality_no_mojibake_and_no_suspicious_rule_based_100_claims() -> None:
    files = [
        "README.md",
        "docs/README.md",
        "docs/quickstart.md",
        "docs/framework_integrations_quickstart.md",
        "docs/custom_integration_from_scratch.md",
        "docs/monitoring_alerts.md",
    ]
    bad_fragments = ["вЂ", "Ã", "�"]
    suspicious = [
        "rule-based precision 100%",
        "rule-based + precision",
        "f1=1.0000",
    ]

    for rel in files:
        text = _read(rel)
        for frag in bad_fragments:
            assert frag not in text, f"mojibake marker '{frag}' found in {rel}"
        low = text.lower()
        for s in suspicious:
            assert s not in low, f"suspicious phrase '{s}' found in {rel}"
