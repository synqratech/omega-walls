"""Hybrid projector policy helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1

from . import normalization as norm

WALLS = list(WALLS_V1)


def extract_pi0_rule_tier(*, matches: Mapping[str, Any]) -> Dict[str, Any]:
    raw_tier = matches.get("pi0_rule_tier", {})
    walls_raw = raw_tier.get("walls", {}) if isinstance(raw_tier, Mapping) else {}
    walls: Dict[str, Dict[str, Any]] = {}
    hard_any = False
    soft_any = False
    for idx, wall in enumerate(WALLS):
        wall_raw = walls_raw.get(wall, {}) if isinstance(walls_raw, Mapping) else {}
        hard_hit = bool((wall_raw or {}).get("hard_hit", False))
        soft_hit = bool((wall_raw or {}).get("soft_hit", False))
        hard_signals = list((wall_raw or {}).get("hard_signals", [])) if isinstance(wall_raw, Mapping) else []
        soft_signals = list((wall_raw or {}).get("soft_signals", [])) if isinstance(wall_raw, Mapping) else []
        try:
            raw_score = float((wall_raw or {}).get("raw_score", 0.0))
        except Exception:  # noqa: BLE001
            raw_score = 0.0
        tier = str((wall_raw or {}).get("tier", "none") or "none")
        if tier == "none":
            if hard_hit:
                # Preserve legacy hybrid behavior for older rule-tier payloads that
                # only exposed hard/soft hits during the staged rollout.
                tier = "hard_live"
            elif soft_hit:
                tier = "soft"
        walls[wall] = {
            "hard_hit": hard_hit,
            "soft_hit": soft_hit,
            "hard_signals": [str(x) for x in hard_signals if str(x)],
            "soft_signals": [str(x) for x in soft_signals if str(x)],
            "raw_score": raw_score,
            "tier": tier,
            "local_ambiguity_evidence": bool((wall_raw or {}).get("local_ambiguity_evidence", False)),
            "action_surface_present": bool((wall_raw or {}).get("action_surface_present", False)),
            "hidden_surface_present": bool((wall_raw or {}).get("hidden_surface_present", False)),
            "exfil_surface_present": bool((wall_raw or {}).get("exfil_surface_present", False)),
            "live_directive_present": bool((wall_raw or {}).get("live_directive_present", False)),
            "wall_index": idx,
        }
        hard_any = hard_any or hard_hit
        soft_any = soft_any or soft_hit
    hard_any = bool(raw_tier.get("hard_any", hard_any)) if isinstance(raw_tier, Mapping) else hard_any
    soft_any = bool(raw_tier.get("soft_any", soft_any)) if isinstance(raw_tier, Mapping) else soft_any
    return {"walls": walls, "hard_any": bool(hard_any), "soft_any": bool(soft_any)}


def semantic_benign_override_eligible(
    *,
    tier_payload: Mapping[str, Any],
    defensive_context: bool,
    directive_intent: bool,
    api_positive_pressure_for_wall: float,
    confidence: float,
    confidence_min: float,
    api_positive_max_for_override: float,
    api_confirmation_available: bool,
) -> Tuple[bool, list[str]]:
    reasons: list[str] = []
    tier = str((tier_payload or {}).get("tier", "none") or "none")
    if tier != "semantic_dependent_hard":
        reasons.append("tier_not_eligible")
    if not bool((tier_payload or {}).get("local_ambiguity_evidence", False)):
        reasons.append("missing_local_ambiguity_evidence")
    if not bool(defensive_context):
        reasons.append("not_defensive_context")
    if bool(directive_intent):
        reasons.append("directive_intent_present")
    if float(api_positive_pressure_for_wall) > float(api_positive_max_for_override):
        reasons.append("api_positive_pressure_too_high")
    if float(confidence) < float(confidence_min):
        reasons.append("confidence_below_threshold")
    if bool((tier_payload or {}).get("action_surface_present", False)):
        reasons.append("action_surface_present")
    if bool((tier_payload or {}).get("hidden_surface_present", False)):
        reasons.append("hidden_surface_present")
    if bool((tier_payload or {}).get("exfil_surface_present", False)):
        reasons.append("exfil_surface_present")
    if not bool(api_confirmation_available):
        reasons.append("semantic_confirmation_unavailable")
    return (len(reasons) == 0, reasons)


def short_fast_path_decision(
    *,
    item: ContentItem,
    p0_v: np.ndarray,
    pi0_rule_tier: Mapping[str, Any],
    enabled: bool,
    threshold_chars: int,
    skip_on_pi0_hard: bool,
    skip_on_pi0_clean: bool,
    hard_min_score: float,
    clean_max_score: float,
) -> Tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    text_len = len(norm.normalize_text(getattr(item, "text", "")))
    if text_len > int(threshold_chars):
        return False, "not_short"
    hard_any = bool(pi0_rule_tier.get("hard_any", False))
    soft_any = bool(pi0_rule_tier.get("soft_any", False))
    max_p0 = float(np.max(np.asarray(p0_v, dtype=float)))
    if skip_on_pi0_hard and hard_any and max_p0 >= float(hard_min_score):
        walls_payload = (pi0_rule_tier.get("walls", {}) or {}) if isinstance(pi0_rule_tier, Mapping) else {}
        p0_arr = np.asarray(p0_v, dtype=float)
        for idx, wall in enumerate(WALLS):
            tier_payload = walls_payload.get(wall, {}) if isinstance(walls_payload, Mapping) else {}
            tier = str((tier_payload or {}).get("tier", "none") or "none")
            local_ambiguity = bool((tier_payload or {}).get("local_ambiguity_evidence", False))
            wall_score = float(p0_arr[idx]) if idx < len(p0_arr) else 0.0
            if wall_score < float(hard_min_score):
                continue
            if tier in {"non_overridable_hard", "hard_live"}:
                return True, "pi0_hard_high_confidence"
            if tier == "semantic_dependent_hard" and not local_ambiguity:
                return True, "pi0_hard_high_confidence"
    if skip_on_pi0_clean and (not hard_any) and (not soft_any) and max_p0 <= float(clean_max_score):
        return True, "pi0_clean_high_confidence"
    return False, "ambiguous"
