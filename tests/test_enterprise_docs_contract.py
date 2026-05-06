from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENG_ROOT = ROOT / "enterprise" / "docs" / "ENG"
RU_ROOT = ROOT / "enterprise" / "docs" / "RU"

CANONICAL_ENG_FILES = {
    "README.md",
    "01_setup_runtime.md",
    "02_control_plane_cli_operations.md",
    "03_incident_export_replay_workflow.md",
    "04_operations_governance_runbook.md",
    "05_pilot_scope_licensing_support.md",
}


def _md_files(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*.md") if p.is_file())


def _relative_set(base: Path) -> set[str]:
    return {str(p.relative_to(base)).replace("\\", "/") for p in _md_files(base)}


def _extract_markdown_links(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text)]


def _assert_local_links_exist(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    base = md_path.parent
    for link in _extract_markdown_links(text):
        if not link or link.startswith("#"):
            continue
        low = link.lower()
        if low.startswith(("http://", "https://", "mailto:")):
            continue
        target_raw = link.split("#", 1)[0].strip()
        if not target_raw:
            continue
        target = (base / target_raw).resolve()
        assert target.exists(), f"broken local link in {md_path}: {link}"


def test_enterprise_docs_eng_exact_mvp5_surface() -> None:
    assert ENG_ROOT.exists(), "missing enterprise/docs/ENG"
    assert _relative_set(ENG_ROOT) == CANONICAL_ENG_FILES


def test_enterprise_docs_language_contract() -> None:
    cyrillic = re.compile(r"[А-Яа-яЁё]")
    for md in _md_files(ENG_ROOT):
        text = md.read_text(encoding="utf-8")
        assert cyrillic.search(text) is None, f"cyrillic detected in ENG doc: {md}"


def test_ru_docs_are_frozen_non_blocking() -> None:
    # RU mirror is intentionally frozen for this iteration; parity is non-blocking.
    if RU_ROOT.exists():
        assert RU_ROOT.is_dir()


def test_enterprise_docs_link_integrity_and_no_legacy_paths() -> None:
    legacy_tokens = (
        "docs/reports/",
        "docs/implementation/",
        "docs/logging_and_audit.md",
    )
    for md in _md_files(ENG_ROOT):
        _assert_local_links_exist(md)
        text = md.read_text(encoding="utf-8")
        for token in legacy_tokens:
            assert token not in text, f"legacy path found in {md}: {token}"


def test_enterprise_public_allowlist_contract() -> None:
    manifest_path = ROOT / "config" / "enterprise_docs_public_allowlist.json"
    assert manifest_path.exists(), "missing enterprise docs publication manifest"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    include = set(payload.get("include", []))
    exclude = set(payload.get("exclude_globs", []))

    for required in (
        "enterprise/docs/ENG/README.md",
        "enterprise/docs/ENG/01_setup_runtime.md",
        "enterprise/docs/ENG/02_control_plane_cli_operations.md",
        "enterprise/docs/ENG/03_incident_export_replay_workflow.md",
        "enterprise/docs/ENG/04_operations_governance_runbook.md",
        "enterprise/docs/ENG/05_pilot_scope_licensing_support.md",
    ):
        assert required in include

    for required in (
        "enterprise/**/out/**",
        "enterprise/reports/weekly/weekly_security_report_*.md",
        "enterprise/reports/weekly/weekly_security_report_*.json",
        "enterprise/internal/**",
        "enterprise/docs/RU/**",
    ):
        assert required in exclude

