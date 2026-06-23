"""Schemas for typed harmful effect shadow diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple


CORE_EFFECTS: Tuple[str, ...] = (
    "install_untrusted_skill",
    "modify_skill_or_tool",
    "write_persistent_memory",
    "memory_poisoning",
    "privilege_escalation",
    "resource_exhaustion",
)

EFFECT_DOMAINS: Tuple[str, ...] = (
    "skill_integrity",
    "memory_integrity",
    "privilege_integrity",
    "resource_integrity",
)

EFFECT_POLICY_GATE_STATUSES: Tuple[str, ...] = (
    "not_applicable",
    "passed",
    "review",
    "suppressed",
    "authorized",
)

FORECAST_STATUSES: Tuple[str, ...] = (
    "disabled",
    "skipped",
    "provider_unavailable",
    "provider_error",
    "invalid_response",
    "no_effect",
    "below_threshold",
    "candidate",
)


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return bool(value)


def _strict_confidence(value: Any, *, field_name: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return out


@dataclass(frozen=True)
class TypedEffectForecast:
    effect: str
    harmful: bool
    confidence: float
    status: str = "candidate"
    rationale: Optional[str] = None

    def __post_init__(self) -> None:
        effect = str(self.effect or "").strip().lower()
        status = str(self.status or "").strip().lower()
        if not effect:
            raise ValueError("effect must be non-empty")
        if status not in FORECAST_STATUSES:
            raise ValueError("status must be one of FORECAST_STATUSES")
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "harmful", _strict_bool(self.harmful, field_name="harmful"))
        object.__setattr__(self, "confidence", _strict_confidence(self.confidence))
        object.__setattr__(self, "status", status)
        rationale = None if self.rationale is None else str(self.rationale).strip()
        object.__setattr__(self, "rationale", rationale or None)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "effect": str(self.effect),
            "harmful": bool(self.harmful),
            "confidence": float(self.confidence),
            "status": str(self.status),
        }
        if self.rationale:
            out["rationale"] = str(self.rationale)
        return out

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TypedEffectForecast":
        effect = str(payload.get("effect", "") or "").strip().lower()
        if not effect or effect in {"none", "no_effect", "benign"}:
            return cls(effect="none", harmful=False, confidence=0.0, status="no_effect")
        return cls(
            effect=effect,
            harmful=_strict_bool(payload.get("harmful", False), field_name="harmful"),
            confidence=_strict_confidence(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "candidate") or "candidate"),
            rationale=(str(payload.get("rationale")).strip() if payload.get("rationale") is not None else None),
        )


@dataclass(frozen=True)
class EffectWallCandidate:
    effect: str
    effect_domain: str
    confidence: float
    reason_code: str
    action_types: Tuple[str, ...]
    would_block: bool = True
    shadow_only: bool = True

    def __post_init__(self) -> None:
        effect = str(self.effect or "").strip().lower()
        domain = str(self.effect_domain or "").strip().lower()
        reason_code = str(self.reason_code or "").strip().lower()
        if effect not in CORE_EFFECTS:
            raise ValueError("EffectWallCandidate.effect must be a core effect")
        if domain not in EFFECT_DOMAINS:
            raise ValueError("EffectWallCandidate.effect_domain must be a known effect domain")
        if not reason_code:
            raise ValueError("EffectWallCandidate.reason_code must be non-empty")
        actions = tuple(str(x).strip().upper() for x in self.action_types if str(x).strip())
        if not actions:
            raise ValueError("EffectWallCandidate.action_types must be non-empty")
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "effect_domain", domain)
        object.__setattr__(self, "confidence", _strict_confidence(self.confidence))
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "action_types", actions)
        object.__setattr__(self, "would_block", bool(self.would_block))
        object.__setattr__(self, "shadow_only", bool(self.shadow_only))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect": str(self.effect),
            "effect_domain": str(self.effect_domain),
            "confidence": float(self.confidence),
            "reason_code": str(self.reason_code),
            "action_types": list(self.action_types),
            "would_block": bool(self.would_block),
            "shadow_only": bool(self.shadow_only),
        }


@dataclass(frozen=True)
class EffectPolicyGate:
    gate_id: str
    effect: str
    effect_domain: str
    status: str
    reason_code: str
    would_enforce: bool
    shadow_only: bool = True
    confidence: float = 0.0
    source_trusts: Tuple[str, ...] = ()
    source_types: Tuple[str, ...] = ()
    rationale: Optional[str] = None

    def __post_init__(self) -> None:
        gate_id = str(self.gate_id or "").strip().lower()
        effect = str(self.effect or "").strip().lower()
        domain = str(self.effect_domain or "").strip().lower()
        status = str(self.status or "").strip().lower()
        reason_code = str(self.reason_code or "").strip().lower()
        if not gate_id:
            raise ValueError("EffectPolicyGate.gate_id must be non-empty")
        if effect not in CORE_EFFECTS:
            raise ValueError("EffectPolicyGate.effect must be a core effect")
        if domain not in EFFECT_DOMAINS:
            raise ValueError("EffectPolicyGate.effect_domain must be a known effect domain")
        if status not in EFFECT_POLICY_GATE_STATUSES:
            raise ValueError("EffectPolicyGate.status must be a known gate status")
        if not reason_code:
            raise ValueError("EffectPolicyGate.reason_code must be non-empty")
        trusts = tuple(sorted({str(x).strip().lower() for x in self.source_trusts if str(x).strip()}))
        types = tuple(sorted({str(x).strip().lower() for x in self.source_types if str(x).strip()}))
        rationale = None if self.rationale is None else str(self.rationale).strip()
        object.__setattr__(self, "gate_id", gate_id)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "effect_domain", domain)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "would_enforce", bool(self.would_enforce))
        object.__setattr__(self, "shadow_only", bool(self.shadow_only))
        object.__setattr__(self, "confidence", _strict_confidence(self.confidence))
        object.__setattr__(self, "source_trusts", trusts)
        object.__setattr__(self, "source_types", types)
        object.__setattr__(self, "rationale", rationale or None)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "gate_id": str(self.gate_id),
            "effect": str(self.effect),
            "effect_domain": str(self.effect_domain),
            "status": str(self.status),
            "reason_code": str(self.reason_code),
            "would_enforce": bool(self.would_enforce),
            "shadow_only": bool(self.shadow_only),
            "confidence": float(self.confidence),
            "source_trusts": list(self.source_trusts),
            "source_types": list(self.source_types),
        }
        if self.rationale:
            out["rationale"] = str(self.rationale)
        return out
