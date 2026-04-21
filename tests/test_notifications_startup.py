from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from omega.notifications.dispatcher import NotificationDispatcher
from omega.notifications.models import ActionRequestEvent, RiskEvent, new_event_id, utc_now_iso
from omega.notifications.startup import build_preflight_checklist, render_preflight_text
from omega.notifications.store import InMemoryApprovalStore


class _ProjectorActive:
    semantic_active = True

    def semantic_status(self) -> Dict[str, Any]:
        return {"enabled_mode": "auto", "active": True, "attempted": True, "error": ""}


class _ProjectorFallback:
    semantic_active = False

    def semantic_status(self) -> Dict[str, Any]:
        return {"enabled_mode": "auto", "active": False, "attempted": True, "error": "missing deps"}


class _OkNotifier:
    async def send_alert(self, event: RiskEvent) -> str:
        _ = event
        return "ok"

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        _ = event
        return "ok"


class _FailNotifier:
    async def send_alert(self, event: RiskEvent) -> str:
        _ = event
        raise RuntimeError("boom")

    async def send_action_request(self, event: ActionRequestEvent) -> str:
        _ = event
        return "ok"


def _startup_event(kind: str = "startup_preflight") -> RiskEvent:
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=utc_now_iso(),
        surface="runtime",
        control_outcome="WARN",
        triggers=["STARTUP_PREFLIGHT"],
        reasons=["semantic_readiness"],
        action_types=[],
        trace_id="trc-startup",
        decision_id="dec-startup",
        risk_score=0.8,
        payload_redacted={"event_kind": kind, "startup_text": "startup"},
        event_kind=kind,
    )


def _wait_metric(dispatcher: NotificationDispatcher, key: str, expected: int, timeout_sec: float = 2.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        snap = dispatcher.metrics_snapshot()
        if int(snap.get(key, 0)) >= expected:
            return
        time.sleep(0.05)
    snap = dispatcher.metrics_snapshot()
    raise AssertionError(f"metric {key} expected>={expected}, got={snap.get(key, 0)}")


def test_preflight_builder_semantic_fallback_and_channel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_ALERT_CHANNEL", raising=False)
    cfg = {
        "runtime": {"guard_mode": "enforce"},
        "projector": {"mode": "hybrid_api"},
        "tools": {"execution_mode": "ENFORCE"},
        "notifications": {
            "enabled": True,
            "approvals": {"timeout_sec": 900, "internal_auth": {"require_hmac": True, "hmac_secret_env": "OMEGA_NOTIFICATION_HMAC_SECRET"}},
            "slack": {"enabled": True, "bot_token_env": "SLACK_BOT_TOKEN", "channel_env": "SLACK_ALERT_CHANNEL"},
            "telegram": {"enabled": False},
        },
    }
    checklist = build_preflight_checklist(
        config=cfg,
        profile="dev",
        surface="runtime",
        projector=_ProjectorFallback(),
        providers={},
    )
    statuses = {str(item["name"]): str(item["status"]) for item in checklist["items"]}
    assert statuses["semantic_readiness"] == "WARN"
    assert statuses["slack_channel"] == "MISSING"
    assert checklist["overall_status"] in {"WARN", "MISSING"}
    text = render_preflight_text(checklist)
    assert "Slack alerts are not fully configured." in text
    assert "Semantic fallback appears active." in text


def test_preflight_builder_notifications_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMEGA_NOTIFICATION_HMAC_SECRET", "x")
    cfg = {
        "runtime": {"guard_mode": "monitor"},
        "projector": {"mode": "pi0"},
        "tools": {"execution_mode": "DRY_RUN"},
        "notifications": {"enabled": False, "slack": {"enabled": True}, "telegram": {"enabled": True}},
    }
    checklist = build_preflight_checklist(
        config=cfg,
        profile="quickstart",
        surface="runtime",
        projector=_ProjectorActive(),
        providers={},
    )
    statuses = {str(item["name"]): str(item["status"]) for item in checklist["items"]}
    assert statuses["notifications_enabled"] == "DISABLED"
    assert statuses["semantic_readiness"] == "OK"
    text = render_preflight_text(checklist)
    assert "Channel alerts are disabled globally." in text


def test_dispatcher_startup_dedup_once_per_process() -> None:
    dispatcher = NotificationDispatcher(
        config={"notifications": {"slack": {"enabled": True}}},
        store=InMemoryApprovalStore(),
        providers={"slack": _OkNotifier()},
    )
    try:
        first = dispatcher.emit_startup_event(_startup_event(), startup_kind="preflight", once_per_process=True)
        second = dispatcher.emit_startup_event(_startup_event(), startup_kind="preflight", once_per_process=True)
        assert first is True
        assert second is False
        _wait_metric(dispatcher, "startup_messages_sent", 1)
        snap = dispatcher.metrics_snapshot()
        assert int(snap.get("startup_messages_skipped_dedup", 0)) >= 1
    finally:
        dispatcher.close()


def test_dispatcher_startup_fail_open_on_provider_error() -> None:
    dispatcher = NotificationDispatcher(
        config={"notifications": {"slack": {"enabled": True}}},
        store=InMemoryApprovalStore(),
        providers={"slack": _FailNotifier()},
    )
    try:
        queued = dispatcher.emit_startup_event(_startup_event(), startup_kind="outreach", once_per_process=False)
        assert queued is True
        _wait_metric(dispatcher, "startup_messages_failed", 1)
        snap = dispatcher.metrics_snapshot()
        assert int(snap.get("notifications_failed", 0)) >= 1
    finally:
        dispatcher.close()
