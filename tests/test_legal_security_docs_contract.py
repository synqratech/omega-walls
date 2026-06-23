from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    "legal/privacy_policy.md",
    "legal/data_processing_overview.md",
    "legal/terms_boundary_oss_vs_enterprise.md",
    "security/security_overview.md",
    "security/telemetry_and_data_handling.md",
    "security/shared_responsibility.md",
    "security/vulnerability_management.md",
    "security/procurement_faq_lite.md",
]

PRIVATE_DOCS = [
    "enterprise/legal/pilot_terms.md",
    "enterprise/legal/msa_lite.md",
    "enterprise/legal/dpa_lite.md",
    "enterprise/security/support_and_sla.md",
]


def _read(rel: str) -> str:
    p = ROOT / rel
    assert p.exists(), f"missing file: {rel}"
    return p.read_text(encoding="utf-8")


def test_legal_security_mvp12_files_exist() -> None:
    for rel in PUBLIC_DOCS + PRIVATE_DOCS:
        assert (ROOT / rel).exists(), f"missing doc: {rel}"


def test_public_legal_security_docs_english_only() -> None:
    cyr = re.compile(r"[\u0400-\u04FF]")
    for rel in PUBLIC_DOCS:
        text = _read(rel)
        assert cyr.search(text) is None, f"cyrillic detected in {rel}"


def test_public_legal_security_docs_no_unsupported_claims() -> None:
    banned = [
        "soc2 certified",
        "iso 27001 certified",
        "hipaa compliant",
        "gdpr compliant",
    ]
    corpus = "\n".join(_read(rel).lower() for rel in PUBLIC_DOCS)
    for phrase in banned:
        assert phrase not in corpus, f"unsupported compliance claim found: {phrase}"


def test_required_disclaimers_present() -> None:
    privacy = _read("legal/privacy_policy.md").lower()
    security_overview = _read("security/security_overview.md").lower()
    telemetry = _read("security/telemetry_and_data_handling.md").lower()
    vuln = _read("security/vulnerability_management.md").lower()

    assert "tbd before commercial signing" in privacy
    assert "self-hosted" in privacy
    assert "do **not** receive customer prompts" in privacy
    assert "self-hosted" in security_overview
    assert "raw prompts" in telemetry
    assert "targets, not guarantees" in vuln


def test_readme_links_public_legal_security_baseline() -> None:
    readme = _read("README.md")
    assert "[Legal (Public Baseline)](legal/privacy_policy.md)" in readme
    assert "[Security (Public Baseline)](security/security_overview.md)" in readme


def test_public_docs_do_not_link_internal_paths() -> None:
    for rel in PUBLIC_DOCS:
        text = _read(rel)
        assert "internal_docs/" not in text
