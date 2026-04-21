from __future__ import annotations

from pathlib import Path


def test_package_install_workflow_exists_and_targets_linux_windows() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "package-install-smoke.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "name: package-install-smoke" in text
    assert "ubuntu-latest" in text
    assert "windows-latest" in text
    assert "python scripts/smoke_package_install.py" in text

