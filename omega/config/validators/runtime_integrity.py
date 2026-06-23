"""Validation for runtime integrity configuration."""

from __future__ import annotations

from typing import Any, Dict


def validate_runtime_integrity_config(config: Dict[str, Any]) -> None:
    cfg = config.get("runtime_integrity", {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("runtime_integrity must be a mapping")
    _ = bool(cfg.get("enabled", True))
    mode = str(cfg.get("mode", "shadow")).strip().lower()
    if mode not in {"shadow", "enforce"}:
        raise ValueError("runtime_integrity.mode must be shadow|enforce")
    _ = bool(cfg.get("emit_artifact_trace", True))

    hard = cfg.get("hard_invariants", {}) or {}
    if hard and not isinstance(hard, dict):
        raise ValueError("runtime_integrity.hard_invariants must be a mapping")
    for key in (
        "skill_source_mismatch",
        "untrusted_skill_install_without_approval",
        "quarantined_source_memory_write",
        "missing_reentry_scan",
    ):
        if key in hard and type(hard.get(key)) is not bool:
            raise ValueError(f"runtime_integrity.hard_invariants.{key} must be boolean")

    resource = cfg.get("resource_limits", {}) or {}
    if resource and not isinstance(resource, dict):
        raise ValueError("runtime_integrity.resource_limits must be a mapping")
    if int(resource.get("max_resource_units", 1000)) <= 0:
        raise ValueError("runtime_integrity.resource_limits.max_resource_units must be > 0")
    if "require_budget" in resource and type(resource.get("require_budget")) is not bool:
        raise ValueError("runtime_integrity.resource_limits.require_budget must be boolean")

    approval = cfg.get("approval_policy", {}) or {}
    if approval and not isinstance(approval, dict):
        raise ValueError("runtime_integrity.approval_policy must be a mapping")
    for key in ("privileged_requires_approval", "resource_heavy_requires_approval"):
        if key in approval and type(approval.get(key)) is not bool:
            raise ValueError(f"runtime_integrity.approval_policy.{key} must be boolean")
