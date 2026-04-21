from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


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


def test_required_reliability_docs_exist() -> None:
    required = [
        "docs/quickstart.md",
        "docs/debugging_workflow_failures.md",
        "docs/policy_tuning.md",
        "docs/workflow_continuity.md",
        "examples/reliability_quickstart/monitor_quickstart_demo.py",
        "examples/reliability_quickstart/explain_timeline_demo.py",
        "examples/reliability_quickstart/workflow_continuity_demo.py",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"missing required artifact: {rel}"


def test_readme_links_reliability_docs() -> None:
    text = _read("README.md")
    assert "(docs/quickstart.md)" in text
    assert "(docs/debugging_workflow_failures.md)" in text
    assert "(docs/policy_tuning.md)" in text
    assert "(docs/workflow_continuity.md)" in text


def test_docs_index_links_reliability_docs() -> None:
    text = _read("docs/README.md")
    assert "(quickstart.md)" in text
    assert "(debugging_workflow_failures.md)" in text
    assert "(policy_tuning.md)" in text
    assert "(workflow_continuity.md)" in text


def test_quickstart_contains_monitor_report_explain_and_enforce_transition() -> None:
    text = _read("docs/quickstart.md").lower()
    assert "monitor" in text
    assert "omega-walls report" in text
    assert "omega-walls explain" in text
    assert "guard_mode: enforce" in text
    assert "fallback" in text or "continuity" in text


def test_local_links_in_reliability_docs_are_valid() -> None:
    for rel in [
        "README.md",
        "docs/README.md",
        "docs/quickstart.md",
        "docs/debugging_workflow_failures.md",
        "docs/policy_tuning.md",
        "docs/workflow_continuity.md",
    ]:
        _assert_local_links_exist(rel)
