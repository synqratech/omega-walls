from __future__ import annotations

import json
from pathlib import Path


def test_oss_export_allowlist_manifest_is_curated_and_internal_safe() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "config" / "oss_export_allowlist.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    include = set(payload.get("include", []))
    excludes = set(payload.get("exclude_globs", []))

    assert "docs" not in include
    assert "docs/README.md" in include
    assert "docs/quickstart.md" in include
    assert "docs/config.md" in include
    assert "docs/framework_integrations_quickstart.md" in include
    assert "docs/enterprise_pilot_guide.md" not in include
    assert "docs/pilot_operations_runbook.md" not in include
    assert "docs/limitations_roadmap.md" in include
    assert "SECURITY.md" in include
    assert "CONTRIBUTING.md" in include
    assert "benchmarks" in include
    assert "legal" in include
    assert "security" in include
    assert "integrations" in include

    for required in (
        "internal_docs/**",
        "services/owl-telemetry/**",
        "docs/reports/**",
        "docs/implementation/**",
        "docs/*_ru.md",
        "config/profiles/prod_enterprise.yml",
        "config/profiles/prod_vision_enterprise.yml",
        "config/profiles/sensitive_hybrid_redacted.yml",
        "config/profiles/sensitive_local_semantic.yml",
        "docs/enterprise_pilot_guide.md",
        "docs/pilot_operations_runbook.md",
        "scripts/check_commercial_boundary.py",
        "scripts/deployment/**",
        "scripts/omega_walls_enterprise.py",
        "tests/deployment/**",
        "tests/enterprise/**",
        "tests/test_commercial_boundary_packaging.py",
    ):
        assert required in excludes

    assert "config/profiles/sensitive_rules_only.yml" not in excludes
