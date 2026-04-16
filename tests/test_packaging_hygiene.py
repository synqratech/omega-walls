from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_restricts_package_discovery_to_omega() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    tool = data.get("tool", {}) or {}
    setuptools_cfg = (tool.get("setuptools", {}) or {})
    find_cfg = (setuptools_cfg.get("packages", {}) or {}).get("find", {}) or {}

    includes = list(find_cfg.get("include", []))
    excludes = list(find_cfg.get("exclude", []))

    assert "omega*" in includes
    # Keep package scope tight to avoid shipping repo internals.
    assert "tests*" in excludes
    assert "scripts*" in excludes
    assert "data*" in excludes
    assert "artifacts*" in excludes


def test_manifest_excludes_local_artifacts_and_model_blobs() -> None:
    manifest = Path(__file__).resolve().parents[1] / "MANIFEST.in"
    text = manifest.read_text(encoding="utf-8")

    required_lines = [
        "prune .venv",
        "prune artifacts",
        "prune data",
        "prune e5-small-v2",
        "prune multilingual-e5-small",
        "prune deberta-v3-base",
        "exclude model.safetensors",
        "exclude tokenizer.json",
        "exclude vocab.json",
        "exclude merges.txt",
        "exclude API_OpenAI.txt",
    ]
    for line in required_lines:
        assert line in text
