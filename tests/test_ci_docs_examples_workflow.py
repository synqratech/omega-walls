from __future__ import annotations

from pathlib import Path


def test_docs_examples_smoke_workflow_covers_integration_docs_contract() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docs-examples-smoke.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: docs-examples-smoke" in text
    assert "\"integrations/**\"" in text
    assert "tests/test_docs_integrations_contract.py" in text
    assert "tests/test_docs_public_surface_contract.py" in text
