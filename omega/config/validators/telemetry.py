from __future__ import annotations

from typing import Any, Dict


def validate_telemetry_config(config: Dict[str, Any]) -> None:
    telemetry_cfg = config.get("telemetry", {}) or {}
    if telemetry_cfg and not isinstance(telemetry_cfg, dict):
        raise ValueError("telemetry must be a mapping")
    if isinstance(telemetry_cfg, dict) and telemetry_cfg:
        _ = bool(telemetry_cfg.get("enabled", True))
        if not str(telemetry_cfg.get("endpoint", "https://telemetry.omega-walls.io/v1/collect")).strip():
            raise ValueError("telemetry.endpoint must be non-empty")
        if int(telemetry_cfg.get("interval_hours", 24)) <= 0:
            raise ValueError("telemetry.interval_hours must be > 0")
        if int(telemetry_cfg.get("max_batch_kb", 50)) <= 0:
            raise ValueError("telemetry.max_batch_kb must be > 0")
        retry_schedule = telemetry_cfg.get("retry_schedule_sec", [60, 300, 900])
        if not isinstance(retry_schedule, list) or not retry_schedule:
            raise ValueError("telemetry.retry_schedule_sec must be a non-empty list")
        for idx, raw in enumerate(list(retry_schedule)):
            if int(raw) <= 0:
                raise ValueError(f"telemetry.retry_schedule_sec[{idx}] must be > 0")
        tier = str(telemetry_cfg.get("tier", "oss")).strip().lower()
        if tier not in {"oss", "enterprise"}:
            raise ValueError("telemetry.tier must be oss|enterprise")
        deployment_mode = str(telemetry_cfg.get("deployment_mode", "auto")).strip().lower()
        if deployment_mode not in {"auto", "lib", "sidecar", "gateway"}:
            raise ValueError("telemetry.deployment_mode must be auto|lib|sidecar|gateway")
        if not str(telemetry_cfg.get("audit_log_path", "artifacts/logs/telemetry_audit.log")).strip():
            raise ValueError("telemetry.audit_log_path must be non-empty")
        if not str(telemetry_cfg.get("state_path", "artifacts/state/telemetry_state.json")).strip():
            raise ValueError("telemetry.state_path must be non-empty")
        policy_urls = telemetry_cfg.get("policy_urls", {}) or {}
        if policy_urls and not isinstance(policy_urls, dict):
            raise ValueError("telemetry.policy_urls must be a mapping")
        if isinstance(policy_urls, dict):
            for key in ("privacy", "dpa"):
                value = policy_urls.get(key, "")
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"telemetry.policy_urls.{key} must be a string")

