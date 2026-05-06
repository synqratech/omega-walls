from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    "docs/README.md",
    "docs/quickstart.md",
    "docs/config.md",
    "docs/framework_integrations_quickstart.md",
    "docs/enterprise_pilot_guide.md",
    "docs/pilot_operations_runbook.md",
]


def _read(rel_path: str) -> str:
    path = ROOT / rel_path
    assert path.exists(), f"missing file: {rel_path}"
    return path.read_text(encoding="utf-8")


def _extract_markdown_links(text: str) -> list[str]:
    return [str(m.group(1)).strip() for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text)]


def test_docs_index_contains_exact_mvp_pages() -> None:
    text = _read("docs/README.md")
    links = _extract_markdown_links(text)
    local_links = [lnk.split("#", 1)[0].strip() for lnk in links if lnk and not lnk.lower().startswith(("http://", "https://", "mailto:"))]
    expected = {
        "quickstart.md",
        "config.md",
        "framework_integrations_quickstart.md",
        "enterprise_pilot_guide.md",
        "pilot_operations_runbook.md",
    }
    assert set(local_links) == expected


def test_no_internal_docs_links_from_public_surface() -> None:
    for rel in PUBLIC_DOCS:
        text = _read(rel)
        assert "internal_docs/" not in text, f"public doc links to internal_docs: {rel}"


def test_public_docs_are_english_only_contract() -> None:
    cyrillic = re.compile(r"[А-Яа-яЁё]")
    for rel in PUBLIC_DOCS:
        text = _read(rel)
        assert cyrillic.search(text) is None, f"cyrillic detected in public doc: {rel}"


def test_non_stub_docs_do_not_link_internal_docs() -> None:
    for path in (ROOT / "docs").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        normalized = text.lstrip("\ufeff")
        if normalized.startswith("# Internal Document (Moved)"):
            continue
        assert "internal_docs/" not in normalized, f"non-stub doc references internal_docs: {path}"
