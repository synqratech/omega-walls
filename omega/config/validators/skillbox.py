"""Validation for SkillBox configuration."""

from __future__ import annotations

from typing import Any, Dict


def validate_skillbox_config(config: Dict[str, Any]) -> None:
    cfg = config.get("skillbox", {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("skillbox must be a mapping")
    _ = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "shadow")).strip().lower()
    if mode != "shadow":
        raise ValueError("skillbox.mode must be shadow in wave 1")
    backend = str(cfg.get("ledger_backend", "memory")).strip().lower()
    if backend != "memory":
        raise ValueError("skillbox.ledger_backend must be memory")
    for key in ("require_ledger_for_skill_run", "require_hash_match", "require_manifest"):
        if key in cfg and type(cfg.get(key)) is not bool:
            raise ValueError(f"skillbox.{key} must be boolean")
    enforcement = cfg.get("enforcement", {}) or {}
    if enforcement and not isinstance(enforcement, dict):
        raise ValueError("skillbox.enforcement must be a mapping")
    if "source_mismatch" in enforcement and type(enforcement.get("source_mismatch")) is not bool:
        raise ValueError("skillbox.enforcement.source_mismatch must be boolean")
    dangerous = cfg.get("dangerous_capabilities", [])
    if dangerous is not None and not isinstance(dangerous, list):
        raise ValueError("skillbox.dangerous_capabilities must be a list")
    for idx, value in enumerate(list(dangerous or [])):
        if not str(value).strip():
            raise ValueError(f"skillbox.dangerous_capabilities[{idx}] must be non-empty")
