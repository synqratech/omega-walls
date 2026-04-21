from __future__ import annotations

from pathlib import Path


def test_openclaw_plugin_ci_workflow_exists_and_has_required_steps() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "openclaw-plugin-ci.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: openclaw-plugin-ci" in text
    assert "node-version: \"20\"" in text
    assert "npm run typecheck" in text
    assert "npm run test" in text
    assert "npm run build" in text
