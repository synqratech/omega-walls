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
    assert "docs/enterprise_pilot_guide.md" in include
    assert "docs/pilot_operations_runbook.md" in include

    for required in (
        "internal_docs/**",
        "docs/reports/**",
        "docs/implementation/**",
        "docs/*_ru.md",
    ):
        assert required in excludes

