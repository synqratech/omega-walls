"""Typed monitor event model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class MonitorEvent:
    ts: str
    surface: str
    session_id: str
    actor_id: str
    mode: str
    risk_score: float
    intended_action: str
    actual_action: str
    triggered_rules: List[str]
    attribution: List[Dict[str, Any]]
    reason_codes: List[str]
    trace_id: str
    decision_id: str
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    downstream: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, Any] = field(default_factory=dict)
    false_positive_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
