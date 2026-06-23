from __future__ import annotations

from typing import Any, Dict

from .common import WALLS_V1_ORDER


def validate_off_policy_config(config: Dict[str, Any]) -> None:
    walls = config["omega"]["walls"]
    if walls != list(WALLS_V1_ORDER):
        raise ValueError("Wall ordering mismatch with v1 contract")

    gamma_omega = config["omega"]["attribution"]["gamma"]
    gamma_policy = config["off_policy"]["block"]["gamma"]
    if abs(float(gamma_omega) - float(gamma_policy)) > 1e-9:
        raise ValueError("gamma mismatch between omega.attribution and off_policy.block")

    enforcement_mode = str(config["off_policy"].get("enforcement_mode", "ENFORCE")).upper()
    if enforcement_mode not in {"ENFORCE", "LOG_ONLY"}:
        raise ValueError("off_policy.enforcement_mode must be ENFORCE or LOG_ONLY")
    control_outcome_cfg = (config.get("off_policy", {}) or {}).get("control_outcome", {}) or {}
    if control_outcome_cfg and not isinstance(control_outcome_cfg, dict):
        raise ValueError("off_policy.control_outcome must be a mapping")
    if isinstance(control_outcome_cfg, dict) and control_outcome_cfg:
        warn_cfg = control_outcome_cfg.get("warn", {}) or {}
        if warn_cfg and not isinstance(warn_cfg, dict):
            raise ValueError("off_policy.control_outcome.warn must be a mapping")
        if isinstance(warn_cfg, dict) and warn_cfg:
            _ = bool(warn_cfg.get("enabled", False))
            if not str(warn_cfg.get("target", "SESSION")).strip():
                raise ValueError("off_policy.control_outcome.warn.target must be non-empty")
            if float(warn_cfg.get("max_p_gte", 0.0)) < 0.0:
                raise ValueError("off_policy.control_outcome.warn.max_p_gte must be >= 0")
            if float(warn_cfg.get("sum_m_next_gte", 0.0)) < 0.0:
                raise ValueError("off_policy.control_outcome.warn.sum_m_next_gte must be >= 0")
        req_cfg = control_outcome_cfg.get("require_approval", {}) or {}
        if req_cfg and not isinstance(req_cfg, dict):
            raise ValueError("off_policy.control_outcome.require_approval must be a mapping")
        if isinstance(req_cfg, dict) and req_cfg:
            _ = bool(req_cfg.get("enabled", False))
            _ = bool(req_cfg.get("on_off", True))
            _ = bool(req_cfg.get("on_warn", True))
            tools = req_cfg.get("tools", [])
            if not isinstance(tools, list):
                raise ValueError("off_policy.control_outcome.require_approval.tools must be a list")
            if int(req_cfg.get("horizon_steps", 0)) < 0:
                raise ValueError("off_policy.control_outcome.require_approval.horizon_steps must be >= 0")
    incident_artifact_cfg = (config.get("off_policy", {}) or {}).get("incident_artifact", {}) or {}
    if incident_artifact_cfg and not isinstance(incident_artifact_cfg, dict):
        raise ValueError("off_policy.incident_artifact must be a mapping")
    if isinstance(incident_artifact_cfg, dict) and incident_artifact_cfg:
        _ = bool(incident_artifact_cfg.get("enabled", False))
        _ = bool(incident_artifact_cfg.get("include_timeline", True))
        _ = bool(incident_artifact_cfg.get("capture_incident_text", False))
        emit_for = incident_artifact_cfg.get("emit_for_outcomes", [])
        if emit_for is not None and not isinstance(emit_for, list):
            raise ValueError("off_policy.incident_artifact.emit_for_outcomes must be a list")
    trust_boundary_cfg = (config.get("off_policy", {}) or {}).get("trust_boundary", {}) or {}
    if trust_boundary_cfg and not isinstance(trust_boundary_cfg, dict):
        raise ValueError("off_policy.trust_boundary must be a mapping")
    if isinstance(trust_boundary_cfg, dict) and trust_boundary_cfg:
        tc_guard_cfg = trust_boundary_cfg.get("trusted_control_guard", {}) or {}
        if tc_guard_cfg and not isinstance(tc_guard_cfg, dict):
            raise ValueError("off_policy.trust_boundary.trusted_control_guard must be a mapping")
        if isinstance(tc_guard_cfg, dict) and tc_guard_cfg:
            policy_action = str(tc_guard_cfg.get("policy_action_on_trigger", "none")).strip().lower()
            if policy_action not in {"none", "warn", "human_escalate"}:
                raise ValueError(
                    "off_policy.trust_boundary.trusted_control_guard.policy_action_on_trigger "
                    "must be none|warn|human_escalate"
                )

    cross_session_cfg = config.get("off_policy", {}).get("cross_session", {})
    if cross_session_cfg:
        backend = str(cross_session_cfg.get("backend", "sqlite")).lower()
        if backend != "sqlite":
            raise ValueError("off_policy.cross_session.backend must be sqlite in v1")
        decay_mode = str(cross_session_cfg.get("decay", {}).get("mode", "exponential")).lower()
        if decay_mode != "exponential":
            raise ValueError("off_policy.cross_session.decay.mode must be exponential in v1")
        half_life = float(cross_session_cfg.get("decay", {}).get("half_life_steps", 120))
        if half_life <= 0:
            raise ValueError("off_policy.cross_session.decay.half_life_steps must be > 0")

