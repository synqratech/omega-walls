"""Canonical incident artifact builder (evidence-first, additive)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import uuid


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_sorted_unique(items: Iterable[Any]) -> list[str]:
    return sorted({str(x) for x in items if str(x).strip()})


def _normalize_action(action: Any) -> Dict[str, Any]:
    if hasattr(action, "__dict__"):
        raw = dict(action.__dict__)
    elif isinstance(action, dict):
        raw = dict(action)
    else:
        raw = {}
    out: Dict[str, Any] = {
        "type": str(raw.get("type", "")).strip(),
        "target": str(raw.get("target", "")).strip(),
    }
    for key in ("doc_ids", "source_ids", "allowlist"):
        values = raw.get(key)
        if values is not None:
            out[key] = _to_sorted_unique(values)
    for key in ("tool_mode", "horizon_steps"):
        value = raw.get(key)
        if value is not None:
            out[key] = value
    return out


def _extract_text_sha256(item: Mapping[str, Any]) -> str:
    content_ref = item.get("content_ref")
    if isinstance(content_ref, Mapping):
        text_hash = str(content_ref.get("text_sha256", "")).strip()
        if text_hash:
            return text_hash
    if str(item.get("text_sha256", "")).strip():
        return str(item.get("text_sha256", "")).strip()
    text = str(item.get("text", ""))
    if text:
        return _sha256_hex(text)
    return ""


def _normalize_top_doc(item: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "doc_id": str(item.get("doc_id", "")),
        "source_id": str(item.get("source_id", "")),
        "source_type": str(item.get("source_type", "")),
        "trust": str(item.get("trust", "")),
        "text_sha256": _extract_text_sha256(item),
    }
    contrib_c = item.get("contrib_c")
    if contrib_c is not None:
        out["contrib_c"] = float(contrib_c)
    return out


def _tool_prevention(tool_gateway_events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prevented: list[Dict[str, str]] = []
    reasons: list[str] = []
    for row in tool_gateway_events:
        req = row.get("request", {}) if isinstance(row.get("request", {}), Mapping) else {}
        decision = row.get("decision", {}) if isinstance(row.get("decision", {}), Mapping) else {}
        execution = row.get("execution", {}) if isinstance(row.get("execution", {}), Mapping) else {}
        allowed = bool(decision.get("allowed", False))
        executed = bool(execution.get("executed", False))
        if not allowed and not executed:
            tool_name = str(req.get("tool_name", "")).strip()
            reason = str(decision.get("reason", "")).strip()
            prevented.append(
                {
                    "tool_name": tool_name,
                    "reason": reason,
                    "request_origin": str(req.get("request_origin", "")).strip(),
                }
            )
            if reason:
                reasons.append(reason)
    return {
        "prevented_tools": prevented,
        "decision_reasons": _to_sorted_unique(reasons),
    }


def should_emit_incident_artifact(*, config: Mapping[str, Any], control_outcome: str) -> bool:
    off_policy = config.get("off_policy", {}) if isinstance(config.get("off_policy", {}), Mapping) else {}
    incident_cfg = off_policy.get("incident_artifact", {}) if isinstance(off_policy.get("incident_artifact", {}), Mapping) else {}
    if not bool(incident_cfg.get("enabled", False)):
        return False
    outcome = str(control_outcome).strip().upper() or "ALLOW"
    emit_cfg = incident_cfg.get("emit_for_outcomes")
    if isinstance(emit_cfg, Sequence) and not isinstance(emit_cfg, (str, bytes)):
        emit_set = {str(x).strip().upper() for x in emit_cfg if str(x).strip()}
        if emit_set:
            return outcome in emit_set
    return outcome != "ALLOW"


def build_incident_artifact(
    *,
    config: Mapping[str, Any],
    surface: str,
    session_id: str,
    step: int,
    control_outcome: str,
    off: bool,
    severity: str,
    actions: Sequence[Any],
    reason_flags: Sequence[str],
    contributing_signals: Optional[Mapping[str, Any]] = None,
    top_docs: Optional[Sequence[Mapping[str, Any]]] = None,
    blocked_doc_ids: Optional[Sequence[str]] = None,
    quarantined_source_ids: Optional[Sequence[str]] = None,
    tool_gateway_events: Optional[Sequence[Mapping[str, Any]]] = None,
    context_total_docs: Optional[int] = None,
    context_allowed_docs: Optional[int] = None,
    request_id: Optional[str] = None,
    verdict: Optional[str] = None,
    evidence_id: Optional[str] = None,
    config_refs: Optional[Mapping[str, Any]] = None,
    refs: Optional[Mapping[str, Any]] = None,
    incident_artifact_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_actions = [_normalize_action(a) for a in actions]
    action_types = _to_sorted_unique(a.get("type", "") for a in normalized_actions)
    reason_codes = _to_sorted_unique(reason_flags)
    blocked_ids = _to_sorted_unique(blocked_doc_ids or [])
    quarantined_ids = _to_sorted_unique(quarantined_source_ids or [])
    normalized_docs = [_normalize_top_doc(item) for item in (top_docs or [])]
    tool_summary = _tool_prevention(tool_gateway_events or [])
    prevented_tools = list(tool_summary["prevented_tools"])
    prevented_tool_names = _to_sorted_unique(row.get("tool_name", "") for row in prevented_tools)
    policy_version = str((config.get("off_policy", {}) or {}).get("policy_version", "")).strip()
    profile = str((config.get("profiles", {}) or {}).get("env", "")).strip()
    runtime_mode = str((config.get("runtime", {}) or {}).get("mode", "")).strip()
    ts = str(timestamp or _utc_now_iso())
    artifact_id = str(incident_artifact_id or uuid.uuid4())
    context_prevented = bool(blocked_ids or quarantined_ids)
    tool_execution_prevented = bool(prevented_tools)
    signals = dict(contributing_signals or {})
    include_timeline = bool(
        (
            (config.get("off_policy", {}) or {}).get("incident_artifact", {}) or {}
        ).get("include_timeline", True)
    )

    timeline = {}
    if include_timeline:
        timeline = {
            "context_admission": {
                "total_docs": int(context_total_docs) if context_total_docs is not None else None,
                "allowed_docs": int(context_allowed_docs) if context_allowed_docs is not None else None,
                "blocked_doc_ids": blocked_ids,
                "quarantined_source_ids": quarantined_ids,
                "filtered_before_prompt": bool(context_prevented),
            },
            "policy_decision": {
                "step": int(step),
                "off": bool(off),
                "severity": str(severity),
                "control_outcome": str(control_outcome),
                "reason_flags": reason_codes,
                "action_types": action_types,
            },
            "enforcement_state": {
                "actions_count": int(len(normalized_actions)),
                "quarantined_sources_count": int(len(quarantined_ids)),
            },
            "tool_chokepoint": {
                "requests_seen": int(len(tool_gateway_events or [])),
                "prevented_before_execution": bool(tool_execution_prevented),
                "prevented_tools_count": int(len(prevented_tools)),
                "decision_reasons": list(tool_summary["decision_reasons"]),
            },
        }

    artifact: Dict[str, Any] = {
        "event": "omega_incident_artifact_v1",
        "schema_version": "1.0",
        "incident_artifact_id": artifact_id,
        "timestamp": ts,
        "trace_id": str(trace_id) if trace_id is not None else None,
        "decision_id": str(decision_id) if decision_id is not None else None,
        "surface": str(surface),
        "context": {
            "session_id": str(session_id),
            "step": int(step),
            "request_id": str(request_id) if request_id else None,
            "policy_version": policy_version,
            "profile": profile,
            "runtime_mode": runtime_mode,
        },
        "decision": {
            "control_outcome": str(control_outcome),
            "off": bool(off),
            "severity": str(severity),
            "verdict": str(verdict) if verdict is not None else None,
            "action_types": action_types,
            "actions": normalized_actions,
        },
        "reasons": {
            "reason_flags": reason_codes,
            "contributing_signals": signals,
        },
        "sources": {
            "top_docs": normalized_docs,
            "blocked_doc_ids": blocked_ids,
            "quarantined_source_ids": quarantined_ids,
        },
        "prevention": {
            "context_prevented": bool(context_prevented),
            "tool_execution_prevented": bool(tool_execution_prevented),
            "prevented_tools": prevented_tools,
            "prevented_docs": blocked_ids,
        },
        "refs": {
            "config_refs": dict(config_refs or {}),
            "evidence_id": str(evidence_id) if evidence_id else None,
            "extra": dict(refs or {}),
        },
        "incident_30s": {
            "why": {
                "control_outcome": str(control_outcome),
                "reason_flags": reason_codes,
                "action_types": action_types,
                "signals": signals,
            },
            "when": {
                "timestamp": ts,
                "step": int(step),
                "surface": str(surface),
            },
            "what_prevented": {
                "context_prevented": bool(context_prevented),
                "tool_execution_prevented": bool(tool_execution_prevented),
                "prevented_tools": prevented_tool_names,
                "prevented_docs": blocked_ids,
            },
        },
    }
    if include_timeline:
        artifact["timeline"] = timeline
    return artifact
