from __future__ import annotations

from pathlib import Path


def test_fw001_release_gate_workflow_contract():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "fw001-release-gate.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: fw001-release-gate" in text
    assert 'python-version: "3.13"' in text
    assert "--cov-fail-under=85" in text
    assert "--perf-overhead-max 0.15" in text
    assert "coverage-gate" in text
    assert "perf-gate" in text
    assert "fw001-gate" in text
