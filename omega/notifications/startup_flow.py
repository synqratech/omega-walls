"""Runtime startup preflight and outreach orchestration."""

from __future__ import annotations

import sys
from typing import Any, Dict, Mapping, Optional

from omega.notifications.dispatcher import NotificationDispatcher
from omega.notifications.startup import (
    build_outreach_event,
    build_preflight_checklist,
    build_preflight_event,
    render_outreach_text,
    render_preflight_text,
)


def _startup_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    notifications_cfg = config.get("notifications", {}) if isinstance(config.get("notifications", {}), Mapping) else {}
    startup = notifications_cfg.get("startup", {}) if isinstance(notifications_cfg.get("startup", {}), Mapping) else {}
    preflight = startup.get("preflight", {}) if isinstance(startup.get("preflight", {}), Mapping) else {}
    outreach = startup.get("outreach", {}) if isinstance(startup.get("outreach", {}), Mapping) else {}
    return {
        "preflight": {
            "enabled": bool(preflight.get("enabled", True)),
            "terminal": bool(preflight.get("terminal", True)),
            "channels": bool(preflight.get("channels", True)),
            "once_per_process": bool(preflight.get("once_per_process", True)),
        },
        "outreach": {
            "enabled": bool(outreach.get("enabled", True)),
            "terminal": bool(outreach.get("terminal", True)),
            "channels": bool(outreach.get("channels", True)),
            "once_per_process": bool(outreach.get("once_per_process", True)),
        },
    }


def _emit_terminal(text: str) -> None:
    if not text.strip():
        return
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return
    print(text, flush=True)  # noqa: T201


def run_startup_notifications(
    *,
    config: Mapping[str, Any],
    profile: str,
    surface: str,
    projector: Any,
    dispatcher: Optional[NotificationDispatcher],
) -> Dict[str, Any]:
    cfg = dict(config or {})
    startup_cfg = _startup_cfg(cfg)
    checklist = build_preflight_checklist(
        config=cfg,
        profile=str(profile),
        surface=str(surface),
        projector=projector,
        providers=dict(getattr(dispatcher, "providers", {}) or {}),
    )
    preflight_text = render_preflight_text(checklist)
    if startup_cfg["preflight"]["enabled"] and startup_cfg["preflight"]["terminal"]:
        _emit_terminal(preflight_text)

    preflight_sent = False
    notifications_enabled = bool((cfg.get("notifications", {}) or {}).get("enabled", False))
    if (
        dispatcher is not None
        and notifications_enabled
        and startup_cfg["preflight"]["enabled"]
        and startup_cfg["preflight"]["channels"]
    ):
        preflight_event = build_preflight_event(checklist=checklist, text=preflight_text)
        preflight_sent = bool(
            dispatcher.emit_startup_event(
                preflight_event,
                startup_kind="preflight",
                once_per_process=bool(startup_cfg["preflight"]["once_per_process"]),
            )
        )

    outreach_text = render_outreach_text(config=cfg)
    if startup_cfg["outreach"]["enabled"] and startup_cfg["outreach"]["terminal"]:
        _emit_terminal(outreach_text)

    outreach_sent = False
    if (
        dispatcher is not None
        and notifications_enabled
        and startup_cfg["outreach"]["enabled"]
        and startup_cfg["outreach"]["channels"]
    ):
        outreach_event = build_outreach_event(surface=str(surface), text=outreach_text)
        outreach_sent = bool(
            dispatcher.emit_startup_event(
                outreach_event,
                startup_kind="outreach",
                once_per_process=bool(startup_cfg["outreach"]["once_per_process"]),
            )
        )

    return {
        "surface": str(surface),
        "profile": str(profile),
        "preflight": {
            "overall_status": str(checklist.get("overall_status", "WARN")),
            "sent_to_channels": bool(preflight_sent),
            "terminal_enabled": bool(startup_cfg["preflight"]["terminal"]),
        },
        "outreach": {
            "sent_to_channels": bool(outreach_sent),
            "terminal_enabled": bool(startup_cfg["outreach"]["terminal"]),
        },
    }
