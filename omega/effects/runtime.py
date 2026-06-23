"""Runtime helper for typed effect shadow diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from omega.effects.forecast import TypedEffectForecaster
from omega.effects.mapping import build_effect_candidate
from omega.effects.policy_gate import evaluate_effect_policy_gate
from omega.effects.schema import CORE_EFFECTS, EffectWallCandidate, TypedEffectForecast
from omega.interfaces.contracts_v1 import ContentItem
from omega.runtime.skill_invocation import detect_named_skill_invocation
from omega.runtime.skillbox import SkillBox, skillbox_enabled


def _effects_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("effects", {}) if isinstance(config, Mapping) else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _joined_operational_effect_text(items: Sequence[ContentItem], *, user_query: str = "") -> str:
    parts = []
    for item in list(items or []):
        trust = str(getattr(item, "trust", "untrusted") or "untrusted").strip().lower()
        if trust in {"trusted", "trusted_control"}:
            continue
        text = str(getattr(item, "text", "") or "").strip()
        if text:
            parts.append(text)
    if not parts and not list(items or []):
        parts.append(str(user_query or "").strip())
    return "\n\n".join(part for part in parts if part)


def evaluate_typed_effect_shadow(
    *,
    config: Mapping[str, Any],
    projector: Any,
    items: Sequence[ContentItem],
    user_query: str = "",
    source_meta: Optional[Mapping[str, Any]] = None,
    forecaster: Optional[Any] = None,
    skillbox: Optional[SkillBox] = None,
) -> Dict[str, Any]:
    cfg = _effects_cfg(config)
    if not bool(cfg.get("enabled", False)):
        return {
            "effect_forecast_status": "disabled",
            "effect_wall_candidate": None,
            "effect_policy_gate": None,
            "effect_policy_gate_status": "disabled",
            "named_skill_invocation": None,
            "skill_provenance_assessment": None,
            "skillbox_status": "disabled",
            "skillbox_verification": None,
            "skillbox_ledger_hit": False,
            "skillbox_content_sha256": None,
            "skillbox_capabilities": [],
            "skillbox_gate_decision": "disabled",
            "effect_text_analysis": {"path": "operational_effects", "missing_effect_text": False},
        }
    mode = str(cfg.get("mode", "shadow")).strip().lower()
    if mode != "shadow":
        return {
            "effect_forecast_status": "skipped",
            "effect_wall_candidate": None,
            "effect_policy_gate": None,
            "effect_policy_gate_status": "skipped",
            "named_skill_invocation": None,
            "skill_provenance_assessment": None,
            "skillbox_status": "skipped",
            "skillbox_verification": None,
            "skillbox_ledger_hit": False,
            "skillbox_content_sha256": None,
            "skillbox_capabilities": [],
            "skillbox_gate_decision": "skipped",
            "effect_text_analysis": {"path": "operational_effects", "missing_effect_text": False},
        }
    named_skill_signal = detect_named_skill_invocation(items, user_query=user_query)
    active_skillbox = None
    if skillbox is not None:
        active_skillbox = skillbox
    elif skillbox_enabled(config):
        active_skillbox = SkillBox.from_config(config)
    text = _joined_operational_effect_text(items, user_query=user_query)
    if not text:
        skillbox_check = None
        if active_skillbox is not None and bool(named_skill_signal.detected):
            skillbox_check = active_skillbox.check_invocation(
                items=items,
                source_meta=source_meta,
                named_signal=named_skill_signal,
            )
        return {
            "effect_forecast_status": "skipped",
            "effect_wall_candidate": None,
            "effect_policy_gate": None,
            "effect_policy_gate_status": "skipped",
            "named_skill_invocation": named_skill_signal.to_dict(),
            "skill_provenance_assessment": (
                {
                    "status": str(skillbox_check.verification.verification_status),
                    "skill_name": skillbox_check.skill_name,
                    "invocation_type": str(skillbox_check.invocation_type),
                    "reason_code": str(skillbox_check.reason_code),
                    "simulated_block": bool(skillbox_check.would_block),
                    "requires_approval": bool(skillbox_check.requires_approval),
                    "source_trusts": list(named_skill_signal.source_trusts),
                }
                if skillbox_check is not None and skillbox_check.verification is not None
                else None
            ),
            "skillbox_status": (
                str(skillbox_check.status)
                if skillbox_check is not None
                else ("disabled" if active_skillbox is None else "not_applicable")
            ),
            "skillbox_verification": (
                skillbox_check.verification.to_dict()
                if skillbox_check is not None and skillbox_check.verification is not None
                else None
            ),
            "skillbox_ledger_hit": bool(skillbox_check.ledger_hit) if skillbox_check is not None else False,
            "skillbox_content_sha256": (
                skillbox_check.verification.content_sha256
                if skillbox_check is not None and skillbox_check.verification is not None
                else None
            ),
            "skillbox_capabilities": (
                list(skillbox_check.verification.capabilities)
                if skillbox_check is not None and skillbox_check.verification is not None
                else []
            ),
            "skillbox_gate_decision": (
                str(skillbox_check.gate_decision)
                if skillbox_check is not None
                else ("disabled" if active_skillbox is None else "not_applicable")
            ),
            "effect_text_analysis": {"path": "operational_effects", "missing_effect_text": True},
        }
    allowed = cfg.get("allowed_effects", list(CORE_EFFECTS))
    min_confidence = float(cfg.get("min_confidence", 0.70))
    active_forecaster = forecaster or TypedEffectForecaster(projector=projector, config=cfg)
    source_meta_effective = dict(source_meta or {})
    if bool(named_skill_signal.detected):
        source_meta_effective["named_skill_invocation_detected"] = True
        source_meta_effective["named_skill_invocation_type"] = str(named_skill_signal.invocation_type)
        source_meta_effective["named_skill_name"] = (
            str(named_skill_signal.skill_name) if named_skill_signal.skill_name else None
        )
    if bool(named_skill_signal.detected) and str(named_skill_signal.invocation_type) == "installed_skill_use":
        source_meta_effective["claimed_preinstalled_skill_use"] = True
    forecast = active_forecaster.forecast_text(text, source_meta=source_meta_effective)
    if not isinstance(forecast, TypedEffectForecast):
        forecast = TypedEffectForecast.from_payload(dict(forecast or {}))
    candidate = build_effect_candidate(
        forecast,
        min_confidence=min_confidence,
        allowed_effects=[str(x) for x in list(allowed or [])],
    )
    if (
        candidate is None
        and bool(named_skill_signal.detected)
        and str(forecast.status) not in {"provider_error", "provider_unavailable", "invalid_response"}
        and (not bool(forecast.harmful) or str(forecast.status) == "skipped")
    ):
        candidate = EffectWallCandidate(
            effect="install_untrusted_skill",
            effect_domain="skill_integrity",
            confidence=max(float(named_skill_signal.confidence), min_confidence),
            reason_code="named_installed_skill_invocation",
            action_types=("SOFT_BLOCK", "TOOL_FREEZE", "HUMAN_ESCALATE"),
            would_block=True,
            shadow_only=True,
        )
    policy_gate = evaluate_effect_policy_gate(
        candidate=candidate,
        text=text,
        items=items,
    )
    skillbox_check = None
    if active_skillbox is not None and bool(named_skill_signal.detected):
        skillbox_check = active_skillbox.check_invocation(
            items=items,
            source_meta=source_meta,
            named_signal=named_skill_signal,
        )
    skill_provenance_assessment = None
    if skillbox_check is not None and skillbox_check.verification is not None:
        skill_provenance_assessment = {
            "status": str(skillbox_check.verification.verification_status),
            "skill_name": str(skillbox_check.skill_name) if skillbox_check.skill_name else None,
            "invocation_type": str(skillbox_check.invocation_type),
            "reason_code": str(skillbox_check.reason_code),
            "simulated_block": bool(skillbox_check.would_block),
            "requires_approval": bool(skillbox_check.requires_approval),
            "source_trusts": list(named_skill_signal.source_trusts),
        }
    elif bool(named_skill_signal.detected):
        assessment_status = "requires_verification"
        gate_reason = (
            str(policy_gate.reason_code)
            if policy_gate is not None and str(policy_gate.reason_code or "").strip()
            else ""
        )
        if gate_reason == "effect_gate_skill_install_source_mismatch":
            assessment_status = "skill_source_mismatch"
        elif any(
            trust in {"untrusted", "semi", "semi_trusted", "tainted_internal", "mixed", "unknown"}
            for trust in list(named_skill_signal.source_trusts or [])
        ):
            assessment_status = "untrusted_skill_requires_approval"
        skill_provenance_assessment = {
            "status": assessment_status,
            "skill_name": str(named_skill_signal.skill_name) if named_skill_signal.skill_name else None,
            "invocation_type": str(named_skill_signal.invocation_type),
            "reason_code": (
                "effect_gate_skill_install_source_mismatch"
                if assessment_status == "skill_source_mismatch"
                else assessment_status
            ),
            "simulated_block": bool(assessment_status == "skill_source_mismatch"),
            "requires_approval": bool(
                assessment_status in {"requires_verification", "untrusted_skill_requires_approval"}
            ),
            "source_trusts": list(named_skill_signal.source_trusts),
        }
    status = str(forecast.status)
    if candidate is not None:
        status = "candidate"
    elif status in {"provider_error", "provider_unavailable", "invalid_response"}:
        status = status
    elif not bool(forecast.harmful):
        status = "no_effect"
    elif bool(forecast.harmful) and float(forecast.confidence) < min_confidence:
        status = "below_threshold"
    return {
        "effect_forecast": forecast.to_dict(),
        "effect_forecast_status": status,
        "effect_wall_candidate": candidate.to_dict() if candidate is not None else None,
        "effect_policy_gate": policy_gate.to_dict() if policy_gate is not None else None,
        "effect_policy_gate_status": (
            str(policy_gate.status) if policy_gate is not None else "not_applicable"
        ),
        "named_skill_invocation": named_skill_signal.to_dict(),
        "skill_provenance_assessment": skill_provenance_assessment,
        "skillbox_status": (
            str(skillbox_check.status)
            if skillbox_check is not None
            else ("disabled" if active_skillbox is None else "not_applicable")
        ),
        "skillbox_verification": (
            skillbox_check.verification.to_dict()
            if skillbox_check is not None and skillbox_check.verification is not None
            else None
        ),
        "skillbox_ledger_hit": bool(skillbox_check.ledger_hit) if skillbox_check is not None else False,
        "skillbox_content_sha256": (
            skillbox_check.verification.content_sha256
            if skillbox_check is not None and skillbox_check.verification is not None
            else None
        ),
        "skillbox_capabilities": (
            list(skillbox_check.verification.capabilities)
            if skillbox_check is not None and skillbox_check.verification is not None
            else []
        ),
        "skillbox_gate_decision": (
            str(skillbox_check.gate_decision)
            if skillbox_check is not None
            else ("disabled" if active_skillbox is None else "not_applicable")
        ),
        "effect_text_analysis": {"path": "operational_effects", "missing_effect_text": False},
    }
