"""Offline enterprise licensing configuration invariants."""

from __future__ import annotations

from typing import Any, Dict, Mapping


_ALLOWED_FEATURES = {"text", "attachments", "vision", "control_plane", "incident_replay"}


def validate_licensing_config(config: Dict[str, Any]) -> None:
    raw = config.get("licensing", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("licensing must be a mapping")
    required_features = raw.get("required_features", [])
    if not isinstance(required_features, list):
        raise ValueError("licensing.required_features must be a list")
    unknown = sorted({str(value) for value in required_features} - _ALLOWED_FEATURES)
    if unknown:
        raise ValueError(f"licensing.required_features contains unsupported values: {', '.join(unknown)}")
    for name in ("license_path_env", "keyring_path_env"):
        if not str(raw.get(name, "")).strip():
            raise ValueError(f"licensing.{name} must be non-empty")
    profile = str(((config.get("profiles", {}) or {}).get("env", ""))).strip().lower()
    enterprise_profile = profile in {"prod_enterprise", "prod_vision_enterprise"}
    if enterprise_profile and not bool(raw.get("required", False)):
        raise ValueError("enterprise production profiles require offline license verification")
    if enterprise_profile and bool(raw.get("allow_test_keys", False)):
        raise ValueError("enterprise production profiles forbid test signing keys")
    if enterprise_profile and not bool(raw.get("enforce_release_entitlement", False)):
        raise ValueError("enterprise production profiles must enforce release update entitlement")
    if enterprise_profile and "text" not in {str(value) for value in required_features}:
        raise ValueError("enterprise production profiles require the text entitlement")
