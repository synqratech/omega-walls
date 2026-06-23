"""Status composition helpers for APIPerceptionProjector."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def classify_zero_state(*, reason: str, enabled_mode: str) -> Tuple[str, str]:
    rl = str(reason or "").lower()
    if rl.startswith("semantic_mode_rules_only"):
        return "safe_zero", "rule_based_only"
    if rl.startswith("semantic_mode_rules_plus_ocr"):
        return "failed_zero", "semantic_failed"
    if rl.startswith("semantic_mode_local_semantic"):
        return "safe_zero", "rule_based_only"
    if rl.startswith("short_fast_path:"):
        return "safe_zero", "rule_based_only"
    if rl.startswith("orchestrator_rule_only:"):
        return "safe_zero", "rule_based_only"
    if rl.startswith("api_adapter_disabled"):
        return "safe_zero", "rule_based_only"
    if rl.startswith("missing_env:") and str(enabled_mode) != "true":
        return "safe_zero", "rule_based_only"
    return "failed_zero", "semantic_failed"


def compose_api_perception_status(projector: Any) -> Dict[str, Any]:
    total = int(projector._cache_hits + projector._cache_misses)
    hit_rate = float(projector._cache_hits) / float(total) if total > 0 else 0.0
    err = projector._runtime_error if projector._runtime_error else projector._last_error
    policy_effective = (
        str(projector.semantic_failure_policy)
        if projector._uses_outbound_semantic()
        else "inactive_non_outbound_mode"
    )
    semantic_mode = str(projector._effective_semantic_mode())
    trace = projector._current_trace()
    raw_image_outbound_effective = bool(
        semantic_mode == "hybrid_cloud"
        or (
            semantic_mode == "hybrid_redacted"
            and bool(getattr(projector, "hybrid_redacted_allow_raw_image_outbound", False))
        )
    )
    return {
        "api_adapter_active": bool(projector._active),
        "api_adapter_error": err,
        "schema_valid": projector._last_schema_valid,
        "model": projector.model,
        "provider": str(getattr(trace, "provider", "") or projector.provider),
        "provider_preset": str(getattr(projector, "provider_preset", "") or ""),
        "base_url": str(getattr(projector, "base_url", "") or ""),
        "allowed_base_urls": list(getattr(projector, "allowed_base_urls", ()) or ()),
        "allow_redirects": bool(getattr(projector, "allow_provider_redirects", False)),
        "api_key_source": str(getattr(projector, "_api_key_source", "none") or "none"),
        "extra_headers": sorted(list(getattr(projector, "extra_headers", {}) or {})),
        "semantic_mode": semantic_mode,
        "provider_id": str(getattr(trace, "provider_id", "") or projector.provider),
        "provider_capabilities": dict(
            getattr(trace, "provider_capabilities", {}) or getattr(projector, "provider_capabilities", {}) or {}
        ),
        "provider_route": [dict(x) for x in list(getattr(trace, "provider_route", []) or [])],
        "health_state": str(projector._active_health_state),
        "llm_fallback_active": bool(projector._active_llm_fallback),
        "fallback_level": str(projector._active_fallback_level),
        "fallback_reason": projector._active_fallback_reason,
        "quota_signal": projector._active_quota_signal,
        "cache_hit_rate": hit_rate,
        "cache_hits": int(projector._cache_hits),
        "cache_misses": int(projector._cache_misses),
        "schema_errors": int(projector._schema_errors),
        "zero_mode": str(projector._last_zero_mode),
        "semantic_status": str(projector._last_semantic_status),
        "rule_based_only": bool(projector._last_semantic_status == "rule_based_only"),
        "semantic_failed": bool(projector._last_semantic_status == "semantic_failed"),
        "semantic_failure_policy": policy_effective,
        "semantic_failure_policy_configured": str(projector.semantic_failure_policy),
        "vision_attempted": bool(getattr(projector, "_last_vision_attempted", False)),
        "vision_provider_supported": bool(getattr(projector, "_last_vision_provider_supported", False)),
        "vision_failure_policy": str(getattr(projector, "_last_vision_failure_policy", policy_effective)),
        "vision_fallback_used": bool(getattr(projector, "_last_vision_fallback_used", False)),
        "vision_semantic_status": str(getattr(projector, "_last_vision_semantic_status", "none")),
        "semantic_input_kind": str(getattr(projector, "_last_semantic_input_kind", "text_only")),
        "raw_image_outbound_effective": bool(raw_image_outbound_effective),
        "image_region_pass_enabled": bool(getattr(projector, "image_region_pass_enabled", False)),
        "provider_call_count": int(getattr(projector, "_last_provider_call_count", 0)),
        "retry_count": int(getattr(projector, "_last_retry_count", 0)),
        "cache_hit_last_request": bool(getattr(projector, "_last_cache_hit", False)),
        "semantic_latency_ms": getattr(projector, "_last_semantic_latency_ms", None),
        "first_pass_latency_ms": getattr(projector, "_last_first_pass_latency_ms", None),
        "second_pass_latency_ms": getattr(projector, "_last_second_pass_latency_ms", None),
        "second_pass_attempted": bool(getattr(projector, "_last_second_pass_attempted", False)),
        "second_pass_result": str(getattr(projector, "_last_second_pass_result", "not_attempted")),
        "region_trigger_reason": str(getattr(projector, "_last_region_trigger_reason", "none")),
        "region_variant_count": int(getattr(projector, "_last_region_variant_count", 0)),
        "token_usage": dict(getattr(projector, "_last_token_usage", {}) or {}),
        "redaction": dict(projector._last_redaction_meta),
        "tenant_id": str(getattr(trace, "tenant_id", "")),
        "data_region": str(getattr(trace, "data_region", "unspecified")),
        "visual_egress_decision": str(getattr(trace, "visual_egress_decision", "not_evaluated")),
        "visual_egress_reason": str(getattr(trace, "visual_egress_reason", "none")),
        "provider_processing_region": str(getattr(trace, "provider_processing_region", "")),
    }
