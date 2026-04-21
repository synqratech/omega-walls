"""Typed SDK result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GuardAction:
    type: str
    target: str
    doc_ids: Optional[List[str]] = None
    source_ids: Optional[List[str]] = None
    tool_mode: Optional[str] = None
    allowlist: Optional[List[str]] = None
    horizon_steps: Optional[int] = None
    incident_packet: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardDecision:
    off: bool
    severity: str
    control_outcome: str
    reason_codes: List[str]
    top_docs: List[str]
    actions: List[GuardAction]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [a.to_dict() for a in self.actions]
        return payload


@dataclass(frozen=True)
class DetectionResult:
    session_id: str
    step: int
    walls_triggered: List[str]
    wall_scores: Dict[str, float]
    memory_scores: Dict[str, float]
    decision: GuardDecision
    monitor: Optional[Dict[str, Any]] = None

    @property
    def off(self) -> bool:
        return bool(self.decision.off)

    @property
    def severity(self) -> str:
        return str(self.decision.severity)

    @property
    def control_outcome(self) -> str:
        return str(self.decision.control_outcome)

    @property
    def reason_codes(self) -> List[str]:
        return list(self.decision.reason_codes)

    @property
    def top_docs(self) -> List[str]:
        return list(self.decision.top_docs)

    @property
    def actions(self) -> List[GuardAction]:
        return list(self.decision.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable payload with nested typed decision + flat compatibility keys."""
        out = {
            "session_id": self.session_id,
            "step": int(self.step),
            "walls_triggered": list(self.walls_triggered),
            "wall_scores": dict(self.wall_scores),
            "memory_scores": dict(self.memory_scores),
            "decision": self.decision.to_dict(),
            # Flat compatibility layer for existing integrations.
            "off": bool(self.decision.off),
            "severity": str(self.decision.severity),
            "control_outcome": str(self.decision.control_outcome),
            "reason_codes": list(self.decision.reason_codes),
            "top_docs": list(self.decision.top_docs),
            "actions": [a.to_dict() for a in self.decision.actions],
        }
        if isinstance(self.monitor, dict):
            out["monitor"] = dict(self.monitor)
        return out


# Backward-compatible alias for existing code.
OmegaDetectionResult = DetectionResult
