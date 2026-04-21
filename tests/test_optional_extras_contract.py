from __future__ import annotations

from pathlib import Path
import tomllib


def _load_pyproject() -> dict:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _project_deps(data: dict) -> list[str]:
    return list((data.get("project") or {}).get("dependencies") or [])


def _optional_deps(data: dict) -> dict[str, list[str]]:
    return dict((data.get("project") or {}).get("optional-dependencies") or {})


def test_required_optional_extras_declared() -> None:
    data = _load_pyproject()
    opt = _optional_deps(data)

    for name in ("api", "integrations", "attachments"):
        assert name in opt
        assert isinstance(opt[name], list)
        assert len(opt[name]) > 0


def test_base_install_stays_lightweight_and_extras_are_explicit() -> None:
    data = _load_pyproject()
    base = " ".join(_project_deps(data)).lower()

    # These packages must remain optional (extras only), not base deps.
    must_be_optional = (
        "fastapi",
        "uvicorn",
        "langchain",
        "llama-index",
        "pypdf",
        "python-docx",
    )
    for marker in must_be_optional:
        assert marker not in base


def test_optional_extras_do_not_duplicate_base_dependencies() -> None:
    data = _load_pyproject()
    base = _project_deps(data)
    base_names = {dep.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for dep in base}

    opt = _optional_deps(data)
    for extra_name in ("api", "integrations", "attachments"):
        for dep in opt.get(extra_name, []):
            dep_name = dep.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
            assert dep_name not in base_names

