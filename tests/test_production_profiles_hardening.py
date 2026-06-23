from __future__ import annotations

import copy

import pytest

from omega.api import runtime_factory
from omega.config.loader import load_resolved_config, validate_resolved_config


def test_stable_prod_is_stateful_rules_only_and_has_no_vision_ocr() -> None:
    cfg = load_resolved_config(profile="prod").resolved
    attachments = cfg["retriever"]["sqlite_fts"]["attachments"]
    assert cfg["api"]["runtime"]["mode"] == "stateful"
    assert cfg["api"]["runtime"]["allow_request_override"] is False
    assert cfg["projector"]["mode"] == "pi0"
    assert cfg["projector"]["api_perception"]["enabled"] is False
    assert attachments["visual"]["enabled"] is False
    assert str(attachments["ocr"]["enabled"]).lower() == "false"
    assert cfg["runtime"]["required_components"] == ["attachments"]


def test_prod_vision_is_external_visual_fail_closed_without_ocr() -> None:
    cfg = load_resolved_config(profile="prod_vision").resolved
    attachments = cfg["retriever"]["sqlite_fts"]["attachments"]
    api_perception = cfg["projector"]["api_perception"]
    assert cfg["api"]["runtime"]["mode"] == "stateful"
    assert cfg["projector"]["fallback_to_pi0"] is False
    assert cfg["pi0"]["semantic"]["enabled"] is False
    assert api_perception["semantic_mode"] == "hybrid_cloud"
    assert api_perception["provider"] == "openai"
    assert api_perception["semantic_failure_policy"] == "fail_closed"
    assert attachments["visual"]["enabled"] is True
    assert attachments["visual"]["failure_policy"] == "fail_closed"
    assert str(attachments["ocr"]["enabled"]).lower() == "false"
    assert set(cfg["runtime"]["required_components"]) == {"attachments"}


def test_prod_vision_local_ocr_preserves_optional_enhanced_path() -> None:
    cfg = load_resolved_config(profile="prod_vision_local_ocr").resolved
    attachments = cfg["retriever"]["sqlite_fts"]["attachments"]
    api_perception = cfg["projector"]["api_perception"]
    assert cfg["api"]["runtime"]["mode"] == "stateful"
    assert cfg["projector"]["fallback_to_pi0"] is False
    assert api_perception["semantic_mode"] == "rules_plus_ocr"
    assert api_perception["provider"] == "local_vision"
    assert api_perception["semantic_failure_policy"] == "fail_closed"
    assert attachments["visual"]["enabled"] is True
    assert attachments["visual"]["failure_policy"] == "fail_closed"
    assert str(attachments["ocr"]["enabled"]).lower() == "true"
    assert attachments["ocr"]["failure_policy"] == "degrade"
    assert set(cfg["runtime"]["required_components"]) == {"attachments", "vision"}


def test_prod_api_is_stateful_hybrid_cloud_text_profile() -> None:
    cfg = load_resolved_config(profile="prod_api").resolved
    attachments = cfg["retriever"]["sqlite_fts"]["attachments"]
    api_perception = cfg["projector"]["api_perception"]
    assert cfg["api"]["runtime"]["mode"] == "stateful"
    assert cfg["projector"]["mode"] == "hybrid_api"
    assert api_perception["enabled"] is True
    assert api_perception["semantic_mode"] == "hybrid_cloud"
    assert cfg["pi0"]["semantic"]["enabled"] is False
    assert attachments["visual"]["enabled"] is False
    assert str(attachments["ocr"]["enabled"]).lower() == "false"


def test_prod_contract_rejects_accidental_ocr_or_stateless_drift() -> None:
    cfg = load_resolved_config(profile="prod").resolved

    bad_ocr = copy.deepcopy(cfg)
    bad_ocr["retriever"]["sqlite_fts"]["attachments"]["ocr"]["enabled"] = "auto"
    with pytest.raises(ValueError, match="OCR must be disabled"):
        validate_resolved_config(bad_ocr)

    bad_state = copy.deepcopy(cfg)
    bad_state["api"]["runtime"]["mode"] = "stateless"
    with pytest.raises(ValueError, match="must be stateful"):
        validate_resolved_config(bad_state)


def test_prod_vision_contract_rejects_ocr_auto_and_semantic_drift() -> None:
    cfg = load_resolved_config(profile="prod_vision").resolved

    bad = copy.deepcopy(cfg)
    bad["runtime"]["required_components"] = ["attachments", "vision"]
    with pytest.raises(ValueError, match="must not require local vision"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["retriever"]["sqlite_fts"]["attachments"]["ocr"]["enabled"] = "auto"
    with pytest.raises(ValueError, match="OCR must be disabled"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["projector"]["api_perception"]["semantic_failure_policy"] = "degrade"
    with pytest.raises(ValueError, match="semantic_failure_policy must be fail_closed"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["projector"]["api_perception"]["semantic_mode"] = "rules_plus_ocr"
    with pytest.raises(ValueError, match="semantic_mode must be hybrid_cloud"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["pi0"]["semantic"]["enabled"] = "auto"
    with pytest.raises(ValueError, match="pi0.semantic.enabled must be false"):
        validate_resolved_config(bad)


def test_prod_vision_local_ocr_contract_rejects_ocr_auto_and_bad_failure_policy() -> None:
    cfg = load_resolved_config(profile="prod_vision_local_ocr").resolved

    bad = copy.deepcopy(cfg)
    bad["runtime"]["required_components"] = ["attachments"]
    with pytest.raises(ValueError, match="must include vision"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["retriever"]["sqlite_fts"]["attachments"]["ocr"]["enabled"] = "auto"
    with pytest.raises(ValueError, match="OCR must be explicitly true"):
        validate_resolved_config(bad)

    bad = copy.deepcopy(cfg)
    bad["retriever"]["sqlite_fts"]["attachments"]["ocr"]["failure_policy"] = "quarantine"
    with pytest.raises(ValueError, match="OCR failure_policy must be fail_closed or degrade"):
        validate_resolved_config(bad)


def test_required_runtime_components_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"runtime": {"required_components": ["attachments", "vision"]}}
    available = {"pypdf", "fitz", "docx", "bs4", "lxml", "PIL"}
    monkeypatch.setattr(
        runtime_factory.importlib.util,
        "find_spec",
        lambda name: object() if name in available else None,
    )
    with pytest.raises(RuntimeError, match="vision=rapidocr,onnxruntime"):
        runtime_factory._validate_required_runtime_components(cfg)


def test_required_runtime_components_accept_complete_install(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"runtime": {"required_components": ["attachments", "vision"]}}
    monkeypatch.setattr(runtime_factory.importlib.util, "find_spec", lambda _name: object())
    runtime_factory._validate_required_runtime_components(cfg)
