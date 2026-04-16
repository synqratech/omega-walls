from __future__ import annotations

import importlib
from pathlib import Path
import tomllib


def test_project_scripts_declared_in_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = ((data.get("project") or {}).get("scripts") or {})

    assert scripts.get("omega-walls") == "omega.cli:main"
    assert scripts.get("omega-walls-api") == "omega.api.cli:main"


def test_project_script_targets_importable_callables() -> None:
    targets = ("omega.cli:main", "omega.api.cli:main")
    for target in targets:
        mod_name, func_name = target.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        assert callable(fn)

