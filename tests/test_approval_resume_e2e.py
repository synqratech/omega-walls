from __future__ import annotations

from omega.notifications.dispatcher import NotificationDispatcher
from omega.notifications.models import ApprovalDecision, RiskEvent, new_event_id, utc_now_iso
from omega.notifications.store import InMemoryApprovalStore


def _risk_event(*, session_id: str, actor_id: str = "actor-1") -> RiskEvent:
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=utc_now_iso(),
        surface="framework_matrix",
        control_outcome="HUMAN_ESCALATE",
        triggers=["BLOCK", "HUMAN_ESCALATE", "REQUIRE_APPROVAL"],
        reasons=["reason_spike"],
        action_types=["HUMAN_ESCALATE", "REQUIRE_APPROVAL"],
        trace_id="trc-test",
        decision_id="dec-test",
        tenant_id="tenant-test",
        session_id=session_id,
        actor_id=actor_id,
        step=1,
        severity="L3",
        risk_score=0.91,
        payload_redacted={"sample": "x"},
    )


def test_approval_lifecycle_pending_approve_deny_timeout_and_resume() -> None:
    dispatcher = NotificationDispatcher(
        config={"notifications": {}},
        store=InMemoryApprovalStore(),
        providers={},
    )
    try:
        event = _risk_event(session_id="sess-a")
        pending = dispatcher.create_action_request(risk_event=event, required_action="REQUIRE_APPROVAL", timeout_sec=120)
        assert pending.status == "pending"

        approved = dispatcher.resolve_approval(
            approval_id=pending.approval_id,
            decision=ApprovalDecision(decision="approved", source="ops", actor_id="human-1"),
        )
        assert approved is not None
        assert approved.status == "approved"

        # Resume flow: after resolution, a new request for same session can be created again.
        resumed = dispatcher.create_action_request(risk_event=event, required_action="REQUIRE_APPROVAL", timeout_sec=120)
        assert resumed.status == "pending"
        assert resumed.approval_id != pending.approval_id

        denied = dispatcher.resolve_approval(
            approval_id=resumed.approval_id,
            decision=ApprovalDecision(decision="denied", source="ops", actor_id="human-2"),
        )
        assert denied is not None
        assert denied.status == "denied"

        to_expire = dispatcher.create_action_request(risk_event=event, required_action="HUMAN_ESCALATE", timeout_sec=120)
        assert to_expire.status == "pending"
        expired = dispatcher.store.expire_pending(now_iso="2099-01-01T00:00:00Z")
        assert expired
        assert any(row.approval_id == to_expire.approval_id and row.status == "expired" for row in expired)

        after_timeout = dispatcher.create_action_request(risk_event=event, required_action="HUMAN_ESCALATE", timeout_sec=120)
        assert after_timeout.status == "pending"
        assert after_timeout.approval_id != to_expire.approval_id
    finally:
        dispatcher.close()

