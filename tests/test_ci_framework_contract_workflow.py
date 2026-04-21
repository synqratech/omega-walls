from __future__ import annotations

from pathlib import Path


def test_framework_contract_workflow_exists_and_has_required_steps() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "framework-contract-gate.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: framework-contract-gate" in text
    assert "python-version: \"3.13\"" in text
    assert "node-version: \"20\"" in text
    assert "run_framework_matrix_stand.py --layer contract --profile dev --strict" in text

