from __future__ import annotations

from pathlib import Path

import scripts.validate_oss_docs_contract as mod


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_tmp_base_falls_back_when_preferred_unwritable(monkeypatch) -> None:
    preferred = Path("C:/tmp")
    fallback = ROOT / "artifacts"

    def _fake_is_writable(path: Path) -> bool:
        return path.resolve() == fallback.resolve()

    monkeypatch.setattr(mod, "_is_writable_dir", _fake_is_writable)
    out = mod._resolve_tmp_base(preferred=preferred, fallback=fallback)
    assert out.resolve() == fallback.resolve()


def test_validate_manifest_contract_requires_sensitive_profile_excludes() -> None:
    report = mod.validate(manifest=ROOT / "config" / "oss_export_github.json")
    manifest_contract = report.get("manifest_contract", {})
    required_excludes = set(manifest_contract.get("required_excludes", []))
    assert "config/profiles/sensitive_hybrid_redacted.yml" in required_excludes
    assert "config/profiles/sensitive_local_semantic.yml" in required_excludes

