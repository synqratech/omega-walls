from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from omega.notifications.dispatcher import NotificationDispatcher, infer_major_triggers
from omega.notifications.interfaces import Notifier
from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso
from omega.notifications.store import InMemoryApprovalStore


class _FailingNotifier(Notifier):
    async def send_alert(self, event: RiskEvent) -> str:
        _ = event
        raise RuntimeError("simulated_failure")

    async def send_action_request(self, event: Any) -> str:
        _ = event
        raise RuntimeError("simulated_failure")


class _CountingNotifier(Notifier):
    def __init__(self) -> None:
        self.alert_calls = 0

    async def send_alert(self, event: RiskEvent) -> str:
        _ = event
        self.alert_calls += 1
        return "ok"

    async def send_action_request(self, event: Any) -> str:
        _ = event
        return "ok"


def _risk_event(*, session_id: str = "s1") -> RiskEvent:
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=utc_now_iso(),
        surface="runtime",
        control_outcome="SOFT_BLOCK",
        triggers=["BLOCK"],
        reasons=["reason_spike"],
        action_types=["SOFT_BLOCK"],
        trace_id="trc_test",
        decision_id="dec_test",
        tenant_id="runtime",
        session_id=session_id,
        actor_id="a1",
        step=1,
        severity="L2",
        risk_score=0.9,
        payload_redacted={"trace_id": "trc_test"},
    )


def test_infer_major_triggers_maps_major_controls() -> None:
    triggers = infer_major_triggers(
        control_outcome="HUMAN_ESCALATE",
        action_types=["HUMAN_ESCALATE", "REQUIRE_APPROVAL"],
        fallback_active=True,
    )
    assert "BLOCK" in triggers
    assert "HUMAN_ESCALATE" in triggers
    assert "REQUIRE_APPROVAL" in triggers
    assert "FALLBACK" in triggers


def test_dispatcher_fail_open_and_approval_reuse() -> None:
    dispatcher = NotificationDispatcher(
        config={
            "notifications": {
                "enabled": True,
                "slack": {
                    "enabled": True,
                    "triggers": ["BLOCK", "REQUIRE_APPROVAL"],
                    "throttle_windows_sec": {"WARN": 0, "BLOCK": 0},
                },
            }
        },
        store=InMemoryApprovalStore(),
        providers={"slack": _FailingNotifier()},
    )
    try:
        evt = _risk_event(session_id="sess-fail-open")
        dispatcher.emit_risk_event(evt)
        # Allow background worker to process.
        for _ in range(20):
            metrics = dispatcher.metrics_snapshot()
            if int(metrics.get("notifications_failed", 0)) >= 1:
                break
            time.sleep(0.05)
        metrics = dispatcher.metrics_snapshot()
        assert int(metrics.get("notifications_failed", 0)) >= 1

        approval_a = dispatcher.create_action_request(
            risk_event=evt,
            required_action="HUMAN_ESCALATE",
            timeout_sec=120,
        )
        approval_b = dispatcher.create_action_request(
            risk_event=evt,
            required_action="HUMAN_ESCALATE",
            timeout_sec=120,
        )
        assert approval_a.approval_id == approval_b.approval_id
    finally:
        dispatcher.close()


def test_dispatcher_throttle_and_timeout_expiry() -> None:
    notifier = _CountingNotifier()
    dispatcher = NotificationDispatcher(
        config={
            "notifications": {
                "enabled": True,
                "slack": {
                    "enabled": True,
                    "triggers": ["BLOCK"],
                    "throttle_windows_sec": {"WARN": 0, "BLOCK": 60},
                },
            }
        },
        store=InMemoryApprovalStore(),
        providers={"slack": notifier},
    )
    try:
        evt = _risk_event(session_id="sess-throttle")
        dispatcher.emit_risk_event(evt)
        dispatcher.emit_risk_event(evt)
        for _ in range(20):
            if notifier.alert_calls >= 1:
                break
            time.sleep(0.05)
        assert notifier.alert_calls == 1
        assert int(dispatcher.metrics_snapshot().get("notifications_throttled", 0)) >= 1

        approval = dispatcher.create_action_request(
            risk_event=_risk_event(session_id="sess-expire"),
            required_action="REQUIRE_APPROVAL",
            timeout_sec=1,
        )
        # Force expiry deterministically instead of waiting for timeout floor.
        row = dispatcher.store.get(approval.approval_id)
        assert row is not None
        if isinstance(dispatcher.store, InMemoryApprovalStore):
            dispatcher.store._rows[str(approval.approval_id)] = replace(row, expires_at="2000-01-01T00:00:00Z")  # type: ignore[attr-defined]
        expired = dispatcher.expire_timeouts()
        assert any(row.approval_id == approval.approval_id for row in expired)
        rec = dispatcher.get_approval(approval.approval_id)
        assert rec is not None
        assert rec.status == "expired"
    finally:
        dispatcher.close()
