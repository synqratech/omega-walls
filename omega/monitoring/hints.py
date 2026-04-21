"""Deterministic false-positive hint heuristics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_HIGH_ACTIONS = {
    "SOFT_BLOCK",
    "SOURCE_QUARANTINE",
    "TOOL_FREEZE",
    "HUMAN_ESCALATE",
    "REQUIRE_APPROVAL",
}

_TRUSTED_LEVELS = {"trusted", "semi", "semi_trusted"}


def _cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    mon = (config.get("monitoring", {}) or {}) if isinstance(config.get("monitoring", {}), Mapping) else {}
    hints = (mon.get("false_positive_hints", {}) or {}) if isinstance(mon.get("false_positive_hints", {}), Mapping) else {}
    return hints


def _is_high_action(intended_action: str) -> bool:
    return str(intended_action or "").strip().upper() in _HIGH_ACTIONS


def _hint_low_confidence_near_threshold(
    *,
    risk_score: float,
    intended_action: str,
    reason_codes: Sequence[str],
    triggered_rules: Sequence[str],
    config: Mapping[str, Any],
    **_: Any,
) -> Optional[str]:
    if not _is_high_action(intended_action):
        return None
    cfg = _cfg(config).get("low_confidence_near_threshold", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    min_risk = float(cfg.get("min_risk", 0.65))
    max_risk = float(cfg.get("max_risk", 0.82))
    max_rules = int(cfg.get("max_triggered_rules", 2))
    allowed_reason_codes = {str(x) for x in (cfg.get("allowed_reason_codes", ["reason_spike"]) or [])}
    if risk_score < min_risk or risk_score > max_risk:
        return None
    if len(list(triggered_rules)) > max_rules:
        return None
    reasons = {str(x) for x in list(reason_codes)}
    if reasons and not reasons.issubset(allowed_reason_codes):
        return None
    return (
        "Possible FP: triggered by low-confidence near-threshold pattern. "
        "Consider raising threshold or allowlisting the benign source."
    )


def _hint_trusted_source_mismatch(
    *,
    intended_action: str,
    attribution: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    **_: Any,
) -> Optional[str]:
    if not _is_high_action(intended_action):
        return None
    cfg = _cfg(config).get("trusted_source_mismatch", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    trusted_levels = {
        str(x).strip().lower()
        for x in list(cfg.get("trusted_levels", list(_TRUSTED_LEVELS)) or [])
        if str(x).strip()
    }
    for row in list(attribution):
        trust = str((row or {}).get("trust", "")).strip().lower()
        if trust and trust in trusted_levels:
            return (
                "Possible FP: high-risk action on trusted source attribution. "
                "Review source trust mapping or source-specific policy."
            )
    return None


def _hint_transient_spike(
    *,
    intended_action: str,
    reason_codes: Sequence[str],
    config: Mapping[str, Any],
    **_: Any,
) -> Optional[str]:
    if not _is_high_action(intended_action):
        return None
    cfg = _cfg(config).get("transient_spike", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    spike_codes = {str(x) for x in list(cfg.get("spike_only_reason_codes", ["reason_spike"]) or [])}
    reasons = {str(x) for x in list(reason_codes)}
    if not reasons:
        return None
    if reasons.issubset(spike_codes):
        return (
            "Possible FP: transient context spike without sustained multi-wall pressure. "
            "Check benign context for policy-like keywords."
        )
    return None


def infer_false_positive_hint(
    *,
    risk_score: float,
    intended_action: str,
    reason_codes: Sequence[str],
    triggered_rules: Sequence[str],
    attribution: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Optional[str]:
    for resolver in (
        _hint_low_confidence_near_threshold,
        _hint_trusted_source_mismatch,
        _hint_transient_spike,
    ):
        hint = resolver(  # type: ignore[misc]
            risk_score=risk_score,
            intended_action=intended_action,
            reason_codes=reason_codes,
            triggered_rules=triggered_rules,
            attribution=attribution,
            config=config,
        )
        if hint:
            return str(hint)
    return None
