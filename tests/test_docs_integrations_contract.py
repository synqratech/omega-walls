from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ("langchain", "langgraph", "llamaindex", "haystack", "autogen", "crewai", "openclaw")
REQUIRED_FILES = ("README.md", "example.py", "requirements.txt", "test_integration.py")


def _read(rel_path: str) -> str:
    path = ROOT / rel_path
    assert path.exists(), f"missing file: {rel_path}"
    return path.read_text(encoding="utf-8")


def _extract_markdown_links(text: str) -> list[str]:
    return [str(m.group(1)).strip() for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text)]


def _assert_local_links_exist(rel_path: str) -> None:
    text = _read(rel_path)
    base = (ROOT / rel_path).parent
    for link in _extract_markdown_links(text):
        if not link or link.startswith("#"):
            continue
        low = link.lower()
        if low.startswith("http://") or low.startswith("https://") or low.startswith("mailto:"):
            continue
        target_raw = link.split("#", 1)[0].strip()
        if not target_raw:
            continue
        target = (base / target_raw).resolve()
        assert target.exists(), f"broken local link in {rel_path}: {link}"


def test_integration_template_exists() -> None:
    template_dir = ROOT / "integrations" / "_template"
    assert template_dir.exists()
    for filename in REQUIRED_FILES:
        assert (template_dir / filename).exists(), f"missing template file: {filename}"


def test_all_integration_packages_have_required_files() -> None:
    for name in INTEGRATIONS:
        package_dir = ROOT / "integrations" / name
        assert package_dir.exists(), f"missing integration package: {name}"
        for filename in REQUIRED_FILES:
            assert (package_dir / filename).exists(), f"{name}: missing {filename}"


def test_framework_quickstart_links_all_packages() -> None:
    text = _read("docs/framework_integrations_quickstart.md")
    for name in INTEGRATIONS:
        assert f"../integrations/{name}/README.md" in text


def test_local_links_are_valid_for_integration_docs() -> None:
    _assert_local_links_exist("README.md")
    _assert_local_links_exist("docs/README.md")
    _assert_local_links_exist("docs/framework_integrations_quickstart.md")
    _assert_local_links_exist("docs/framework_matrix_stand.md")
    _assert_local_links_exist("docs/openclaw_integration.md")
    for name in INTEGRATIONS:
        _assert_local_links_exist(f"integrations/{name}/README.md")
    _assert_local_links_exist("integrations/_template/README.md")


@pytest.mark.skipif(
    os.environ.get("OMEGA_RUN_INTEGRATION_EXEC_TESTS") != "1",
    reason="integration executable launchers are only required in contract gate",
)
def test_integration_launchers_execute_successfully() -> None:
    for name in INTEGRATIONS:
        if name == "openclaw":
            if os.environ.get("OMEGA_RUN_OPENCLAW_LAUNCHER") != "1":
                continue
            if shutil.which("npm.cmd") is None and shutil.which("npm") is None:
                continue
        launcher = ROOT / "integrations" / name / "test_integration.py"
        proc = subprocess.run(
            [sys.executable, str(launcher)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, (
            f"{name} launcher failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
