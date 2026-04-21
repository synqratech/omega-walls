from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "templates" / "secure_agent_template"


def test_template_root_exists() -> None:
    assert TEMPLATE_ROOT.exists()
    assert (TEMPLATE_ROOT / "copier.yml").exists()
    assert (TEMPLATE_ROOT / "template").exists()


def test_copier_questions_contract() -> None:
    payload = yaml.safe_load((TEMPLATE_ROOT / "copier.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert str(payload.get("_subdirectory", "")) == "template"
    for field in ("project_name", "package_name", "python_version", "framework", "projector_mode", "guard_mode", "include_ci"):
        assert field in payload, f"missing copier question: {field}"

    framework_choices = list(payload["framework"].get("choices", []))
    assert framework_choices == ["none", "langchain", "langgraph", "llamaindex", "haystack", "autogen", "crewai"]
    assert str(payload["framework"].get("default", "")) == "none"
    assert str(payload["python_version"].get("default", "")) == "3.13"
    assert str(payload["projector_mode"].get("default", "")) == "pi0"
    assert str(payload["guard_mode"].get("default", "")) == "monitor"
    assert bool(payload["include_ci"].get("default", False)) is True


def test_template_file_contract_exists() -> None:
    required = [
        "template/app.py.jinja",
        "template/scripts/smoke.py.jinja",
        "template/config/local_dev.yml.jinja",
        "template/requirements.txt.jinja",
        "template/README.md.jinja",
        "template/.env.example",
        "template/RELIABILITY.md.jinja",
        "template/{{ package_name }}/__init__.py.jinja",
    ]
    for rel in required:
        assert (TEMPLATE_ROOT / rel).exists(), f"missing template file: {rel}"
