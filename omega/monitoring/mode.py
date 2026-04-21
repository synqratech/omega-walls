"""Guard mode resolution helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class GuardMode(str, Enum):
    ENFORCE = "ENFORCE"
    MONITOR = "MONITOR"


def _legacy_monitor_equivalent(config: Mapping[str, Any]) -> bool:
    off_cfg = (config.get("off_policy", {}) or {}) if isinstance(config.get("off_policy", {}), Mapping) else {}
    tools_cfg = (config.get("tools", {}) or {}) if isinstance(config.get("tools", {}), Mapping) else {}
    enforcement_mode = str(off_cfg.get("enforcement_mode", "ENFORCE")).strip().upper()
    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).strip().upper()
    return enforcement_mode == "LOG_ONLY" and execution_mode == "DRY_RUN"


def resolve_guard_mode(config: Mapping[str, Any]) -> GuardMode:
    runtime_cfg = (config.get("runtime", {}) or {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    guard_mode_raw = str(runtime_cfg.get("guard_mode", "")).strip().lower()
    if guard_mode_raw in {"monitor", "enforce"}:
        return GuardMode.MONITOR if guard_mode_raw == "monitor" else GuardMode.ENFORCE
    if _legacy_monitor_equivalent(config):
        return GuardMode.MONITOR
    return GuardMode.ENFORCE


def is_monitor_mode(config: Mapping[str, Any]) -> bool:
    return resolve_guard_mode(config) == GuardMode.MONITOR
