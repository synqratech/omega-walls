"""Validation for typed effect shadow configuration."""

from __future__ import annotations

from typing import Any, Dict

from omega.effects.schema import CORE_EFFECTS


def validate_effects_config(config: Dict[str, Any]) -> None:
    cfg = config.get("effects", {}) or {}
    if cfg and not isinstance(cfg, dict):
        raise ValueError("effects must be a mapping")
    if not isinstance(cfg, dict):
        return
    _ = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "shadow")).strip().lower()
    if mode != "shadow":
        raise ValueError("effects.mode must be shadow")
    provider = str(cfg.get("provider", "api_perception")).strip().lower()
    if provider != "api_perception":
        raise ValueError("effects.provider must be api_perception")
    min_confidence = float(cfg.get("min_confidence", 0.70))
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("effects.min_confidence must be in [0,1]")
    allowed = cfg.get("allowed_effects", list(CORE_EFFECTS))
    if not isinstance(allowed, list):
        raise ValueError("effects.allowed_effects must be a list")
    unknown = [str(x) for x in allowed if str(x).strip().lower() not in CORE_EFFECTS]
    if unknown:
        raise ValueError("effects.allowed_effects contains unsupported values")
