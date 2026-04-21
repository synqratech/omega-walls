"""Typed models for notification and approval workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


def new_approval_id() -> str:
    return f"apr_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class RiskEvent:
    event_id: str
    timestamp: str
    surface: str
    control_outcome: str
    triggers: List[str]
    reasons: List[str]
    action_types: List[str]
    trace_id: str
    decision_id: str
    incident_artifact_id: str = ""
    tenant_id: str = ""
    session_id: str = ""
    actor_id: str = ""
    step: int = 0
    severity: str = ""
    risk_score: Optional[float] = None
    payload_redacted: Dict[str, Any] = field(default_factory=dict)
    event_kind: str = "risk_event"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRequestEvent:
    approval_id: str
    risk_event: RiskEvent
    required_action: str
    timeout_sec: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApprovalDecision:
    decision: str
    actor_id: str = ""
    source: str = ""
    reason: str = ""
    resolved_at: str = ""

    def normalized(self) -> "ApprovalDecision":
        status = str(self.decision or "").strip().lower()
        if status not in {"approved", "denied", "expired"}:
            raise ValueError("approval decision must be approved|denied|expired")
        resolved = str(self.resolved_at).strip() or utc_now_iso()
        return ApprovalDecision(
            decision=status,
            actor_id=str(self.actor_id or "").strip(),
            source=str(self.source or "").strip(),
            reason=str(self.reason or "").strip(),
            resolved_at=resolved,
        )


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    status: str
    created_at: str
    updated_at: str
    expires_at: str
    required_action: str
    tenant_id: str
    session_id: str
    actor_id: str
    trace_id: str
    decision_id: str
    control_outcome: str
    channels: List[str] = field(default_factory=list)
    callback_ids: Dict[str, str] = field(default_factory=dict)
    resolution: Optional[ApprovalDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.resolution is not None:
            out["resolution"] = asdict(self.resolution)
        return out
