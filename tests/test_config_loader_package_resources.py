from __future__ import annotations

import os
from pathlib import Path
import shutil

from omega.config.loader import load_resolved_config


def test_load_resolved_config_uses_bundled_resources_by_default() -> None:
    # Ensure runtime does not depend on a local ./config directory.
    prev_cwd = Path.cwd()
    sandbox = prev_cwd / ".tmp_config_loader_pkg_test"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    os.chdir(sandbox)
    try:
        snapshot = load_resolved_config(profile="dev")
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(sandbox, ignore_errors=True)

    assert isinstance(snapshot.resolved, dict)
    assert snapshot.resolved.get("profiles", {}).get("env") == "dev"
    assert "omega" in snapshot.resolved
    assert "off_policy" in snapshot.resolved
    assert snapshot.file_hashes
    assert all(str(key).startswith("pkg://omega.config/resources/") for key in snapshot.file_hashes.keys())


def test_load_resolved_config_still_supports_explicit_config_dir() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"
    snapshot = load_resolved_config(config_dir=str(config_dir), profile="dev")

    assert snapshot.resolved.get("profiles", {}).get("env") == "dev"
    assert snapshot.file_hashes
    # Explicit config_dir path should be reflected in file_hashes keys.
    assert any(str(config_dir).replace("\\", "/") in str(key) for key in snapshot.file_hashes.keys())
