"""Conservative typed-effect to diagnostic-domain mapping."""

from __future__ import annotations

from typing import Iterable, Optional, Set

from omega.effects.schema import CORE_EFFECTS, EffectWallCandidate, TypedEffectForecast


EFFECT_TO_CANDIDATE = {
    "install_untrusted_skill": {
        "effect_domain": "skill_integrity",
        "reason_code": "effect_skill_install",
        "action_types": ("SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE"),
    },
    "modify_skill_or_tool": {
        "effect_domain": "skill_integrity",
        "reason_code": "effect_skill_or_tool_mutation",
        "action_types": ("SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE"),
    },
    "write_persistent_memory": {
        "effect_domain": "memory_integrity",
        "reason_code": "effect_persistent_memory_write",
        "action_types": ("SOFT_BLOCK", "MEMORY_WRITE_DENY", "HUMAN_ESCALATE"),
    },
    "memory_poisoning": {
        "effect_domain": "memory_integrity",
        "reason_code": "effect_memory_poisoning",
        "action_types": ("SOFT_BLOCK", "MEMORY_WRITE_DENY", "HUMAN_ESCALATE"),
    },
    "privilege_escalation": {
        "effect_domain": "privilege_integrity",
        "reason_code": "effect_privilege_escalation",
        "action_types": ("SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE"),
    },
    "resource_exhaustion": {
        "effect_domain": "resource_integrity",
        "reason_code": "effect_resource_exhaustion",
        "action_types": ("SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE"),
    },
}


def _allowed_effect_set(value: Iterable[str] | None) -> Set[str]:
    if value is None:
        return set(CORE_EFFECTS)
    allowed = {str(x).strip().lower() for x in value if str(x).strip()}
    return allowed if allowed else set(CORE_EFFECTS)


def build_effect_candidate(
    forecast: TypedEffectForecast,
    *,
    min_confidence: float = 0.70,
    allowed_effects: Iterable[str] | None = None,
) -> Optional[EffectWallCandidate]:
    if not bool(forecast.harmful):
        return None
    if str(forecast.status) not in {"candidate"}:
        return None
    effect = str(forecast.effect).strip().lower()
    if effect not in _allowed_effect_set(allowed_effects):
        return None
    if effect not in EFFECT_TO_CANDIDATE:
        return None
    if float(forecast.confidence) < float(min_confidence):
        return None
    spec = EFFECT_TO_CANDIDATE[effect]
    return EffectWallCandidate(
        effect=effect,
        effect_domain=str(spec["effect_domain"]),
        confidence=float(forecast.confidence),
        reason_code=str(spec["reason_code"]),
        action_types=tuple(spec["action_types"]),
        would_block=True,
        shadow_only=True,
    )
