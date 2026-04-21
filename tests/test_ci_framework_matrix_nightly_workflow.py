from __future__ import annotations

from pathlib import Path


def test_framework_matrix_nightly_workflow_exists_and_has_required_steps() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "framework-matrix-nightly.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: framework-matrix-nightly" in text
    assert "python-version: \"3.13\"" in text
    assert "node-version: \"20\"" in text
    assert "run_framework_matrix_stand.py --layer workflow --profile dev --strict" in text
    assert "run_framework_stress_chaos.py --profile dev --strict" in text

