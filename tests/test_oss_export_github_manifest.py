from __future__ import annotations

import json
from pathlib import Path


def test_oss_export_github_manifest_has_required_excludes():
    manifest_path = Path(__file__).resolve().parents[1] / "config" / "oss_export_github.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    excludes = set(payload.get("exclude_globs", []))
    include_sources = set()
    include_targets = set()

    for item in payload.get("include", []):
        if isinstance(item, str):
            include_sources.add(item)
            include_targets.add(item)
        else:
            include_sources.add(item.get("src"))
            include_targets.add(item.get("dst"))

    assert "omega" in include_targets
    assert "scripts" in include_targets
    assert "SECURITY.md" in include_targets
    assert "CONTRIBUTING.md" in include_targets
    assert "docs/README.md" in include_targets
    assert "docs/enterprise_pilot_guide.md" not in include_targets
    assert "docs/pilot_operations_runbook.md" not in include_targets
    assert "docs/limitations_roadmap.md" in include_targets
    assert "integrations" in include_targets
    assert "benchmarks" in include_targets
    assert "legal" in include_targets
    assert "security" in include_targets
    assert "README_OSS.md" in include_sources
    assert "README.md" in include_targets

    for required in (
        "internal_docs/**",
        "internal_data/**",
        "services/owl-telemetry/**",
        "redteam/**",
        "data/**",
        "notebooks/**",
        "cloud/**",
        "API_OpenAI.txt",
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/*.pyc",
        ".github/workflows/enterprise-foundation-gate.yml",
        ".github/workflows/supply-chain-build.yml",
        "config/profiles/prod_enterprise.yml",
        "config/profiles/prod_vision_enterprise.yml",
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

    assert "config/profiles/sensitive_hybrid_redacted.yml" in excludes
    assert "config/profiles/sensitive_local_semantic.yml" in excludes
    assert "config/profiles/sensitive_rules_only.yml" not in excludes
