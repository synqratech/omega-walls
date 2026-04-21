from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "openclaw-omega-guard"


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"expected object json: {path}"
    return payload


def test_openclaw_plugin_package_contract_files_exist() -> None:
    required = [
        PLUGIN_ROOT / "package.json",
        PLUGIN_ROOT / "openclaw.plugin.json",
        PLUGIN_ROOT / "tsconfig.json",
        PLUGIN_ROOT / "src" / "index.ts",
        PLUGIN_ROOT / "src" / "omega-client.ts",
        PLUGIN_ROOT / "src" / "hooks.ts",
        PLUGIN_ROOT / "src" / "webfetch.ts",
    ]
    for path in required:
        assert path.exists(), f"missing required OpenClaw plugin artifact: {path}"


def test_openclaw_plugin_manifest_matches_package_entry() -> None:
    package = _read_json(PLUGIN_ROOT / "package.json")
    manifest = _read_json(PLUGIN_ROOT / "openclaw.plugin.json")
    assert manifest["id"] == "omega-openclaw-guard"
    assert package["openclaw"]["extensions"] == ["./dist/index.js"]
    assert manifest["entry"] == "./dist/index.js"


def test_openclaw_plugin_package_scripts_cover_gate_requirements() -> None:
    package = _read_json(PLUGIN_ROOT / "package.json")
    scripts = package.get("scripts", {})
    assert isinstance(scripts, dict)
    assert "typecheck" in scripts
    assert "test" in scripts
    assert "build" in scripts
    assert "smoke" in scripts
