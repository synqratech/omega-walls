"""Strict resolved-profile invariants for production entrypoints."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def validate_production_profile_contract(config: Dict[str, Any]) -> None:
    profile_env = str(_mapping(config.get("profiles")).get("env", "")).strip().lower()
    text_local_profiles = {"prod", "production", "prod_enterprise"}
    text_api_profiles = {"prod_api"}
    vision_cloud_profiles = {"prod_vision", "prod_vision_enterprise"}
    vision_local_ocr_profiles = {"prod_vision_local_ocr"}
    production_profiles = (
        text_local_profiles
        | text_api_profiles
        | vision_cloud_profiles
        | vision_local_ocr_profiles
    )
    if profile_env not in production_profiles:
        return

    runtime = _mapping(config.get("runtime"))
    api = _mapping(config.get("api"))
    api_runtime = _mapping(api.get("runtime"))
    if str(runtime.get("guard_mode", "")).strip().lower() != "enforce":
        raise ValueError("production runtime.guard_mode must be enforce")
    if str(api_runtime.get("mode", "")).strip().lower() != "stateful":
        raise ValueError("production api.runtime.mode must be stateful")
    if bool(api_runtime.get("allow_request_override", False)):
        raise ValueError("production api.runtime.allow_request_override must be false")

    required = {str(value) for value in list(runtime.get("required_components", []) or [])}
    projector = _mapping(config.get("projector"))
    api_perception = _mapping(projector.get("api_perception"))
    retriever = _mapping(config.get("retriever"))
    sqlite_fts = _mapping(retriever.get("sqlite_fts"))
    attachments = _mapping(sqlite_fts.get("attachments"))
    visual = _mapping(attachments.get("visual"))
    ocr = _mapping(attachments.get("ocr"))

    if profile_env in text_local_profiles:
        if "attachments" not in required or "vision" in required:
            raise ValueError("text production runtime.required_components must include attachments and exclude vision")
        if str(projector.get("mode", "")).strip().lower() != "pi0":
            raise ValueError("text production projector.mode must be pi0")
        if bool(api_perception.get("enabled", False)):
            raise ValueError("text production projector.api_perception.enabled must be false")
        if bool(visual.get("enabled", False)):
            raise ValueError("text production attachment visual extraction must be disabled")
        if str(ocr.get("enabled", "false")).strip().lower() != "false":
            raise ValueError("text production attachment OCR must be disabled")
        return

    if profile_env in text_api_profiles:
        if "attachments" not in required or "vision" in required:
            raise ValueError("text API production runtime.required_components must include attachments and exclude vision")
        if str(projector.get("mode", "")).strip().lower() != "hybrid_api":
            raise ValueError("text API production projector.mode must be hybrid_api")
        if not bool(api_perception.get("enabled", False)):
            raise ValueError("text API production projector.api_perception.enabled must be true")
        if str(api_perception.get("semantic_mode", "")).strip().lower() != "hybrid_cloud":
            raise ValueError("text API production semantic_mode must be hybrid_cloud")
        pi0_semantic = _mapping(_mapping(config.get("pi0")).get("semantic"))
        if str(pi0_semantic.get("enabled", "false")).strip().lower() != "false":
            raise ValueError("text API production pi0.semantic.enabled must be false")
        if bool(visual.get("enabled", False)):
            raise ValueError("text API production attachment visual extraction must be disabled")
        if str(ocr.get("enabled", "false")).strip().lower() != "false":
            raise ValueError("text API production attachment OCR must be disabled")
        return

    if "attachments" not in required:
        raise ValueError("vision production runtime.required_components must include attachments")
    if str(projector.get("mode", "")).strip().lower() != "hybrid_api":
        raise ValueError("vision production projector.mode must be hybrid_api")
    if bool(projector.get("fallback_to_pi0", True)):
        raise ValueError("vision production projector.fallback_to_pi0 must be false")
    if str(api_perception.get("semantic_failure_policy", "")).strip().lower() != "fail_closed":
        raise ValueError("vision production semantic_failure_policy must be fail_closed")
    if not bool(visual.get("enabled", False)):
        raise ValueError("vision production visual extraction must be enabled")
    if str(visual.get("failure_policy", "")).strip().lower() != "fail_closed":
        raise ValueError("vision production visual failure_policy must be fail_closed")
    if profile_env in vision_cloud_profiles:
        if "vision" in required:
            raise ValueError("vision cloud production runtime.required_components must not require local vision")
        if str(api_perception.get("semantic_mode", "")).strip().lower() != "hybrid_cloud":
            raise ValueError("vision production semantic_mode must be hybrid_cloud")
        pi0_semantic = _mapping(_mapping(config.get("pi0")).get("semantic"))
        if str(pi0_semantic.get("enabled", "false")).strip().lower() != "false":
            raise ValueError("vision production pi0.semantic.enabled must be false")
        if str(ocr.get("enabled", "false")).strip().lower() != "false":
            raise ValueError("vision production OCR must be disabled")
        return
    if profile_env in vision_local_ocr_profiles:
        if "vision" not in required:
            raise ValueError("vision local OCR production runtime.required_components must include vision")
        if str(api_perception.get("semantic_mode", "")).strip().lower() != "rules_plus_ocr":
            raise ValueError("vision local OCR production semantic_mode must be rules_plus_ocr")
        if str(ocr.get("enabled", "")).strip().lower() != "true":
            raise ValueError("vision local OCR production OCR must be explicitly true, not auto")
        if str(ocr.get("failure_policy", "")).strip().lower() not in {"fail_closed", "degrade"}:
            raise ValueError("vision local OCR production OCR failure_policy must be fail_closed or degrade")
