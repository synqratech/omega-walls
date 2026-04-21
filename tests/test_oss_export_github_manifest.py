from __future__ import annotations

import json
from pathlib import Path


def test_oss_export_github_manifest_has_required_excludes():
    manifest_path = Path(__file__).resolve().parents[1] / "config" / "oss_export_github.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    excludes = set(payload.get("exclude_globs", []))
    include = set(payload.get("include", []))

    assert "omega" in include
    assert "scripts" in include
    assert "docs/README.md" in include

    for required in (
        "internal_data/**",
        "redteam/**",
        "data/**",
        "notebooks/**",
        "cloud/**",
        "API_OpenAI.txt",
        "README_OSS.md",
        "**/node_modules/**",
    ):
        assert required in excludes
