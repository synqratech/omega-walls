"""Event builders for omega telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaStepResult, WALLS_V1
from omega.telemetry.redaction import redact_text


REASON_TO_NAME = {
    "reason_spike": "reason_spike",
    "reason_wall": "reason_wall",
    "reason_sum": "reason_sum",
    "reason_multi": "reason_multi",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reason_list(step_result: OmegaStepResult) -> List[str]:
    reasons = []
    r = step_result.reasons
    for attr in ("reason_spike", "reason_wall", "reason_sum", "reason_multi"):
        if getattr(r, attr):
            reasons.append(REASON_TO_NAME[attr])
    return reasons


def _walls_triggered(step_result: OmegaStepResult, wall_names: List[str]) -> List[str]:
    out: List[str] = []
    for idx, wall in enumerate(wall_names):
        if step_result.p[idx] > 0 or step_result.m_next[idx] > 0:
            out.append(wall)
    return out


def _actions_to_dict(actions: List[OffAction]) -> List[Dict[str, Any]]:
    result = []
    for action in actions:
        d = {"type": action.type, "target": action.target}
        if action.doc_ids is not None:
            d["doc_ids"] = action.doc_ids
        if action.source_ids is not None:
            d["source_ids"] = action.source_ids
        if action.tool_mode is not None:
            d["tool_mode"] = action.tool_mode
        if action.allowlist is not None:
            d["allowlist"] = action.allowlist
        if action.horizon_steps is not None:
            d["horizon_steps"] = action.horizon_steps
        if action.incident_packet is not None:
            d["incident_packet"] = action.incident_packet
        result.append(d)
    return result


def build_step_event(
    step_result: OmegaStepResult,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "event": "omega_step_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": step_result.session_id,
        "step": step_result.step,
        "v_total": step_result.v_total.tolist(),
        "p": step_result.p.tolist(),
        "m_prev": step_result.m_prev.tolist(),
        "m_next": step_result.m_next.tolist(),
        "off": step_result.off,
        "trace_id": trace_id,
        "decision_id": decision_id,
    }


def build_enforcement_step_event(
    session_id: str,
    step: int,
    enforcement_snapshot: Dict[str, Any],
    active_actions: List[OffAction],
    control_outcome: str = "ALLOW",
    cross_session: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    event = {
        "event": "enforcement_step_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": session_id,
        "step": step,
        "freeze": enforcement_snapshot.get("freeze", {}),
        "quarantine": enforcement_snapshot.get("quarantine", {}),
        "active_actions": _actions_to_dict(active_actions),
        "control_outcome": str(control_outcome),
        "trace_id": trace_id,
        "decision_id": decision_id,
    }
    if cross_session is not None:
        event["cross_session"] = cross_session
    return event


def build_tool_gateway_step_event(
    session_id: str,
    step: int,
    tool_name: str,
    request_origin: str,
    intent_id: Optional[int],
    decision: Dict[str, Any],
    capability: Dict[str, Any],
    human_approved: bool,
    executed: bool,
    adapter_present: bool,
    execution_mode: str,
    actor_hash: str,
    source_ids_seen: List[str],
    control_outcome: str = "ALLOW",
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "event": "tool_gateway_step_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": session_id,
        "step": step,
        "request": {
            "tool_name": tool_name,
            "request_origin": request_origin,
            "intent_id": intent_id,
        },
        "decision": {
            "allowed": bool(decision.get("allowed", False)),
            "reason": str(decision.get("reason", "")),
            "mode": str(decision.get("mode", "")),
            "off_state": bool(decision.get("off_state", False)),
            "freeze_active": bool(decision.get("freeze_active", False)),
            "control_outcome": str(control_outcome),
            "validation_status": str(decision.get("validation_status", "not_checked")),
            "validation_reason": decision.get("validation_reason"),
        },
        "capability": {
            "mode": str(capability.get("mode", "unknown")),
            "requires_human_approval": bool(capability.get("requires_human_approval", False)),
        },
        "approval": {
            "human_approved": bool(human_approved),
        },
        "execution": {
            "executed": bool(executed),
            "adapter_present": bool(adapter_present),
            "execution_mode": execution_mode,
        },
        "trace": {
            "actor_hash": actor_hash,
            "source_ids_seen": list(source_ids_seen),
        },
        "trace_id": trace_id,
        "decision_id": decision_id,
    }


def build_off_event(
    step_result: OmegaStepResult,
    decision: OffDecision,
    items: List[ContentItem],
    config_refs: Dict[str, str],
    thresholds: Dict[str, Any],
    capture_text: str = "NEVER",
    max_text_chars: int = 800,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    item_by_id = {item.doc_id: item for item in items}
    top_docs: List[Dict[str, Any]] = []
    top_ids = set(step_result.top_docs)

    for contrib in step_result.contribs:
        if contrib.doc_id not in top_ids:
            continue
        item = item_by_id[contrib.doc_id]
        content_ref: Dict[str, Any] = {"text_capture": capture_text}
        red = redact_text(item.text, max_chars=max_text_chars)
        content_ref["text_sha256"] = red.text_sha256
        if capture_text in {"REDACTED", "ALLOWLISTED"}:
            content_ref["excerpt"] = red.redacted
            content_ref["spans"] = [{"start": 0, "end": len(red.redacted)}]

        top_docs.append(
            {
                "doc_id": contrib.doc_id,
                "source_id": item.source_id,
                "source_type": item.source_type,
                "trust": item.trust,
                "contrib_c": contrib.c,
                "v": contrib.v.tolist(),
                "e": contrib.e.tolist(),
                "projector": {
                    "name": "pi0_intent_v2",
                    "polarity": contrib.evidence.polarity,
                    "debug_scores_raw": contrib.evidence.debug_scores_raw,
                    "matches": contrib.evidence.matches,
                },
                "content_ref": content_ref,
            }
        )

    return {
        "event": "omega_off_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": step_result.session_id,
        "trace_id": trace_id,
        "decision_id": decision_id,
        "step": step_result.step,
        "config_refs": config_refs,
        "walls": WALLS_V1,
        "reasons": _reason_list(step_result),
        "walls_triggered": _walls_triggered(step_result, WALLS_V1),
        "v_total": step_result.v_total.tolist(),
        "p": step_result.p.tolist(),
        "m_prev": step_result.m_prev.tolist(),
        "m_next": step_result.m_next.tolist(),
        "thresholds": thresholds,
        "top_docs": top_docs,
        "actions": _actions_to_dict(decision.actions),
        "control_outcome": str(decision.control_outcome),
        "tool_gateway": {
            "mode": next((a.tool_mode for a in decision.actions if a.type == "TOOL_FREEZE"), "TOOLS_DISABLED"),
            "active_until_step": next((step_result.step + (a.horizon_steps or 0) for a in decision.actions if a.type == "TOOL_FREEZE"), step_result.step),
        },
    }


def build_policy_decision_event(
    *,
    session_id: str,
    step: int,
    trace_id: str,
    decision_id: str,
    control_outcome: str,
    off: bool,
    severity: str,
    action_types: List[str],
    actions: List[OffAction],
    refs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event": "policy_decision_event_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": str(session_id),
        "step": int(step),
        "trace_id": str(trace_id),
        "decision_id": str(decision_id),
        "control_outcome": str(control_outcome),
        "off": bool(off),
        "severity": str(severity),
        "action_types": sorted({str(x) for x in list(action_types) if str(x).strip()}),
        "actions": _actions_to_dict(list(actions)),
        "refs": dict(refs or {}),
    }


def build_evidence_debug_event(
    *,
    session_id: str,
    step: int,
    trace_id: str,
    decision_id: str,
    walls: List[str],
    reasons: List[str],
    walls_triggered: List[str],
    top_docs_summary: List[Dict[str, Any]],
    projector_signal_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "event": "evidence_debug_event_v1",
        "schema_version": "1.0",
        "timestamp": _utc_now_iso(),
        "session_id": str(session_id),
        "step": int(step),
        "trace_id": str(trace_id),
        "decision_id": str(decision_id),
        "walls": list(walls),
        "reasons": sorted({str(x) for x in reasons if str(x).strip()}),
        "walls_triggered": sorted({str(x) for x in walls_triggered if str(x).strip()}),
        "top_docs_summary": list(top_docs_summary),
        "projector_signal_summary": dict(projector_signal_summary),
    }
