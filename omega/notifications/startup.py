"""Startup preflight and outreach message helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

from omega.monitoring.mode import resolve_guard_mode
from omega.notifications.models import RiskEvent, new_event_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_rank(status: str) -> int:
    key = str(status or "").strip().upper()
    if key == "MISSING":
        return 4
    if key == "WARN":
        return 3
    if key == "OK":
        return 2
    return 1


def _compute_overall_status(items: List[Mapping[str, Any]]) -> str:
    if not items:
        return "DISABLED"
    best = max(_status_rank(str(item.get("status", "DISABLED"))) for item in items)
    if best >= 4:
        return "MISSING"
    if best >= 3:
        return "WARN"
    if best >= 2:
        return "OK"
    return "DISABLED"


def _check_provider_readiness(
    *,
    cfg: Mapping[str, Any],
    provider_name: str,
    providers: Mapping[str, Any],
) -> Dict[str, Any]:
    p_cfg = cfg.get(provider_name, {}) if isinstance(cfg.get(provider_name, {}), Mapping) else {}
    enabled = bool(p_cfg.get("enabled", False))
    if not enabled:
        return {
            "name": f"{provider_name}_channel",
            "status": "DISABLED",
            "details": {"enabled": False},
        }
    if provider_name == "slack":
        token_env = str(p_cfg.get("bot_token_env", "SLACK_BOT_TOKEN")).strip()
        channel_env = str(p_cfg.get("channel_env", "SLACK_ALERT_CHANNEL")).strip()
        token_present = bool(str(os.environ.get(token_env, "")).strip())
        channel_present = bool(str(os.environ.get(channel_env, str(p_cfg.get("channel", ""))).strip()))
        provider_ready = provider_name in providers
        status = "OK" if provider_ready else ("MISSING" if (not token_present or not channel_present) else "WARN")
        return {
            "name": "slack_channel",
            "status": status,
            "details": {
                "enabled": True,
                "provider_ready": bool(provider_ready),
                "token_env": token_env,
                "token_present": bool(token_present),
                "channel_env": channel_env,
                "channel_present": bool(channel_present),
            },
        }
    token_env = str(p_cfg.get("bot_token_env", "TG_BOT_TOKEN")).strip()
    chat_env = str(p_cfg.get("chat_id_env", "TG_ADMIN_CHAT_ID")).strip()
    token_present = bool(str(os.environ.get(token_env, "")).strip())
    chat_present = bool(str(os.environ.get(chat_env, str(p_cfg.get("chat_id", ""))).strip()))
    provider_ready = provider_name in providers
    status = "OK" if provider_ready else ("MISSING" if (not token_present or not chat_present) else "WARN")
    return {
        "name": "telegram_channel",
        "status": status,
        "details": {
            "enabled": True,
            "provider_ready": bool(provider_ready),
            "bot_token_env": token_env,
            "token_present": bool(token_present),
            "chat_id_env": chat_env,
            "chat_id_present": bool(chat_present),
        },
    }


def _semantic_readiness(projector: Any) -> Dict[str, Any]:
    status_fn = getattr(projector, "semantic_status", None)
    raw_status: Dict[str, Any]
    if callable(status_fn):
        try:
            raw_status = dict(status_fn() or {})
        except Exception as exc:  # noqa: BLE001
            raw_status = {"active": False, "attempted": True, "error": str(exc), "enabled_mode": "auto"}
    else:
        raw_status = {
            "active": bool(getattr(projector, "semantic_active", True)),
            "attempted": False,
            "error": None,
            "enabled_mode": "n/a",
        }
    enabled_mode = str(raw_status.get("enabled_mode", "auto")).strip().lower()
    active = bool(raw_status.get("active", False))
    attempted = bool(raw_status.get("attempted", False))
    error = raw_status.get("error")
    if enabled_mode in {"false", "off", "disabled"}:
        state = "DISABLED"
    elif active:
        state = "OK"
    elif attempted:
        state = "WARN"
    else:
        state = "WARN"
    return {
        "name": "semantic_readiness",
        "status": state,
        "details": {
            "enabled_mode": enabled_mode,
            "active": active,
            "attempted": attempted,
            "error": (str(error) if error else ""),
        },
    }


def build_preflight_checklist(
    *,
    config: Mapping[str, Any],
    profile: str,
    surface: str,
    projector: Any,
    providers: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = dict(config or {})
    notifications_cfg = cfg.get("notifications", {}) if isinstance(cfg.get("notifications", {}), Mapping) else {}
    api_cfg = cfg.get("api", {}) if isinstance(cfg.get("api", {}), Mapping) else {}
    tools_cfg = cfg.get("tools", {}) if isinstance(cfg.get("tools", {}), Mapping) else {}
    projector_cfg = cfg.get("projector", {}) if isinstance(cfg.get("projector", {}), Mapping) else {}
    approvals_cfg = notifications_cfg.get("approvals", {}) if isinstance(notifications_cfg.get("approvals", {}), Mapping) else {}
    internal_auth_cfg = approvals_cfg.get("internal_auth", {}) if isinstance(approvals_cfg.get("internal_auth", {}), Mapping) else {}

    items: List[Dict[str, Any]] = []
    guard_mode = str(resolve_guard_mode(cfg).value).lower()
    projector_mode = str(projector_cfg.get("mode", "pi0")).strip().lower()
    items.append({"name": "guard_mode", "status": "OK", "details": {"value": guard_mode}})
    items.append({"name": "projector_mode", "status": "OK", "details": {"value": projector_mode}})
    items.append(_semantic_readiness(projector))

    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).strip().upper()
    items.append(
        {
            "name": "tool_execution_mode",
            "status": "OK" if execution_mode in {"DRY_RUN", "ENFORCE"} else "WARN",
            "details": {"value": execution_mode},
        }
    )
    approvals_enabled = bool(notifications_cfg.get("enabled", False)) and bool(approvals_cfg)
    timeout_sec = int(approvals_cfg.get("timeout_sec", 900)) if approvals_cfg else 900
    require_hmac = bool(internal_auth_cfg.get("require_hmac", True))
    hmac_env = str(internal_auth_cfg.get("hmac_secret_env", "OMEGA_NOTIFICATION_HMAC_SECRET")).strip()
    hmac_present = bool(str(os.environ.get(hmac_env, "")).strip())
    if not approvals_enabled:
        approvals_status = "DISABLED"
    elif require_hmac and not hmac_present:
        approvals_status = "MISSING"
    else:
        approvals_status = "OK"
    items.append(
        {
            "name": "approval_flow",
            "status": approvals_status,
            "details": {
                "enabled": bool(approvals_enabled),
                "timeout_sec": int(timeout_sec),
                "require_hmac": bool(require_hmac),
                "hmac_secret_env": hmac_env,
                "hmac_secret_present": bool(hmac_present),
            },
        }
    )

    items.append(
        {
            "name": "notifications_enabled",
            "status": "OK" if bool(notifications_cfg.get("enabled", False)) else "DISABLED",
            "details": {"enabled": bool(notifications_cfg.get("enabled", False))},
        }
    )
    items.append(_check_provider_readiness(cfg=notifications_cfg, provider_name="slack", providers=providers))
    items.append(_check_provider_readiness(cfg=notifications_cfg, provider_name="telegram", providers=providers))

    if str(surface).lower() == "api":
        api_auth_cfg = api_cfg.get("auth", {}) if isinstance(api_cfg.get("auth", {}), Mapping) else {}
        api_security_cfg = api_cfg.get("security", {}) if isinstance(api_cfg.get("security", {}), Mapping) else {}
        api_keys = list(api_auth_cfg.get("api_keys", [])) if isinstance(api_auth_cfg.get("api_keys", []), list) else []
        require_hmac_api = bool(api_auth_cfg.get("require_hmac", True))
        api_hmac_env = str(api_auth_cfg.get("hmac_secret_env", "OMEGA_API_HMAC_SECRET")).strip()
        api_hmac_present = bool(str(os.environ.get(api_hmac_env, "")).strip())
        keys_present = bool(api_keys)
        auth_status = "OK"
        if not keys_present:
            auth_status = "MISSING"
        elif require_hmac_api and not api_hmac_present:
            auth_status = "WARN"
        items.append(
            {
                "name": "api_auth",
                "status": auth_status,
                "details": {
                    "api_keys_present": bool(keys_present),
                    "require_hmac": bool(require_hmac_api),
                    "hmac_secret_env": api_hmac_env,
                    "hmac_secret_present": bool(api_hmac_present),
                },
            }
        )
        require_https = bool(api_security_cfg.get("require_https", True))
        transport_mode = str(api_security_cfg.get("transport_mode", "proxy_tls")).strip().lower()
        items.append(
            {
                "name": "api_transport_security",
                "status": "OK" if require_https else "WARN",
                "details": {"require_https": bool(require_https), "transport_mode": transport_mode},
            }
        )

    overall_status = _compute_overall_status(items)
    return {
        "profile": str(profile),
        "surface": str(surface),
        "guard_mode": guard_mode,
        "projector_mode": projector_mode,
        "overall_status": overall_status,
        "items": items,
    }


def render_preflight_text(checklist: Mapping[str, Any]) -> str:
    head = (
        f"[Omega Startup] Preflight checklist "
        f"(surface={checklist.get('surface', 'n/a')}, profile={checklist.get('profile', 'n/a')}) "
        f"overall={checklist.get('overall_status', 'n/a')}"
    )
    lines: List[str] = [head]
    for item in list(checklist.get("items", []) or []):
        status = str(item.get("status", "DISABLED")).upper()
        name = str(item.get("name", "item"))
        details = item.get("details", {})
        brief = json.dumps(details, ensure_ascii=False, sort_keys=True)
        lines.append(f"- [{status}] {name}: {brief}")
    return "\n".join(lines)


def build_preflight_event(
    *,
    checklist: Mapping[str, Any],
    text: str,
) -> RiskEvent:
    status = str(checklist.get("overall_status", "WARN")).upper()
    control = "WARN" if status in {"WARN", "MISSING"} else "ALLOW"
    reasons = [
        str(item.get("name", "unknown"))
        for item in list(checklist.get("items", []) or [])
        if str(item.get("status", "")).upper() in {"WARN", "MISSING"}
    ]
    trace_seed = json.dumps({"kind": "startup_preflight", "checklist": checklist}, ensure_ascii=False, sort_keys=True)
    trace_id = f"startup-trace-{hashlib.sha256(trace_seed.encode('utf-8')).hexdigest()[:16]}"
    decision_id = f"startup-decision-{hashlib.sha256((trace_seed + ':decision').encode('utf-8')).hexdigest()[:16]}"
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=_utc_now_iso(),
        surface=str(checklist.get("surface", "runtime")),
        control_outcome=control,
        triggers=["STARTUP_PREFLIGHT"],
        reasons=reasons[:8],
        action_types=[],
        trace_id=trace_id,
        decision_id=decision_id,
        step=0,
        severity="L2" if control == "WARN" else "L1",
        risk_score=0.8 if control == "WARN" else 0.0,
        payload_redacted={
            "event_kind": "startup_preflight",
            "overall_status": status,
            "checklist": checklist,
            "startup_text": str(text),
        },
        event_kind="startup_preflight",
    )


@dataclass(frozen=True)
class StartupOutreachConfig:
    github_url: str
    docs_url: str
    linkedin_url: str
    commercial_cta_enabled: bool


def _outreach_config(config: Mapping[str, Any]) -> StartupOutreachConfig:
    startup_cfg = (
        ((config.get("notifications", {}) or {}).get("startup", {}) or {}).get("outreach", {})
        if isinstance(config.get("notifications", {}), Mapping)
        else {}
    )
    if not isinstance(startup_cfg, Mapping):
        startup_cfg = {}
    return StartupOutreachConfig(
        github_url=str(startup_cfg.get("github_url", "https://github.com/omega-walls/omega-walls")).strip(),
        docs_url=str(startup_cfg.get("docs_url", "https://github.com/omega-walls/omega-walls/tree/main/docs")).strip(),
        linkedin_url=str(startup_cfg.get("linkedin_url", "https://www.linkedin.com/company/omega-walls")).strip(),
        commercial_cta_enabled=bool(startup_cfg.get("commercial_cta_enabled", True)),
    )


def render_outreach_text(*, config: Mapping[str, Any]) -> str:
    cfg = _outreach_config(config)
    cta = (
        f"If you need advanced controls and managed rollout support, message us on LinkedIn: {cfg.linkedin_url}"
        if cfg.commercial_cta_enabled
        else ""
    )
    parts = [
        "[Omega Startup] Thanks for using Omega Walls.",
        f"If this helps your team, please star the project: {cfg.github_url}",
        f"Docs: {cfg.docs_url}",
        f"Questions or feedback: {cfg.linkedin_url}",
    ]
    if cta:
        parts.append(cta)
    return "\n".join(parts)


def build_outreach_event(*, surface: str, text: str) -> RiskEvent:
    trace_seed = f"startup_outreach:{surface}:{text}"
    trace_id = f"startup-trace-{hashlib.sha256(trace_seed.encode('utf-8')).hexdigest()[:16]}"
    decision_id = f"startup-decision-{hashlib.sha256((trace_seed + ':decision').encode('utf-8')).hexdigest()[:16]}"
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=_utc_now_iso(),
        surface=str(surface),
        control_outcome="ALLOW",
        triggers=["STARTUP_OUTREACH"],
        reasons=[],
        action_types=[],
        trace_id=trace_id,
        decision_id=decision_id,
        step=0,
        severity="L1",
        risk_score=0.0,
        payload_redacted={
            "event_kind": "startup_outreach",
            "startup_text": str(text),
        },
        event_kind="startup_outreach",
    )
