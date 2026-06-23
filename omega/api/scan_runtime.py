"""Shared scan-runtime helpers extracted from api.server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.notifications.dispatcher import infer_major_triggers
from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso
from omega.policy.control_outcome import control_outcome_from_action_types


def infer_format(filename: str | None, mime: str | None) -> str:
    mime_l = str(mime or "").strip().lower()
    if mime_l == "application/pdf":
        return "pdf"
    if mime_l == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    if mime_l == "text/html":
        return "html"
    if mime_l in {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}:
        return "image"
    name = str(filename or "").strip().lower()
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if ext == ".zip":
        return "zip"
    return "text"


def source_type_for_format(fmt: str) -> str:
    if fmt in {"pdf", "docx", "html", "image"}:
        return fmt
    return "other"


def monitor_attribution_rows(*, items: Sequence[ContentItem], top_chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    item_by_id = {str(item.doc_id): item for item in list(items)}
    rows: List[Dict[str, Any]] = []
    for chunk in list(top_chunks):
        doc_id = str((chunk or {}).get("doc_id", "")).strip()
        if not doc_id:
            continue
        item = item_by_id.get(doc_id)
        if item is None:
            continue
        rows.append(
            {
                "doc_id": str(item.doc_id),
                "source_id": str(item.source_id),
                "trust": str(item.trust),
                "contribution": float((chunk or {}).get("score", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", ""))))
    return rows[:8]


def normalize_trust_band(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"trusted", "semi", "semi_trusted"}:
        return "trusted"
    if raw == "mixed":
        return "mixed"
    return "untrusted"


def source_risk_band(items: Sequence[ContentItem]) -> str:
    bands = {normalize_trust_band(getattr(item, "trust", "untrusted")) for item in list(items)}
    if not bands:
        return "untrusted"
    if len(bands) > 1:
        return "mixed"
    return next(iter(bands))


def resolve_control_outcome(*, action_types: Sequence[str], verdict: str) -> str:
    outcome = control_outcome_from_action_types(action_types)
    if outcome != "ALLOW":
        return outcome
    v = str(verdict).strip().lower()
    if v == "block":
        return "SOFT_BLOCK"
    if v == "quarantine":
        return "WARN"
    return "ALLOW"


def build_document_scan_report(
    *,
    chunk_agg: Any,
    fmt: str,
    ingestion_flags: Sequence[str],
    max_chunks: int,
) -> Dict[str, Any]:
    per_chunk: List[Dict[str, Any]] = []
    chunk_scores = list(getattr(chunk_agg, "chunk_scores", []) or [])
    for row in chunk_scores[: max(1, int(max_chunks))]:
        per_chunk.append(
            {
                "chunk_id": str(getattr(row, "doc_id", "")),
                "score_max": float(getattr(row, "score_max", 0.0)),
                "active_walls": list(getattr(row, "active_walls", []) or []),
                "wall_scores": dict(getattr(row, "wall_scores", {}) or {}),
                "pattern_signals": list(getattr(row, "pattern_signals", []) or []),
                "rule_ids": list(getattr(row, "matched_rule_ids", []) or []),
            }
        )
    return {
        "format": str(fmt),
        "chunks_total": int(len(chunk_scores)),
        "chunks_reported": int(len(per_chunk)),
        "ingestion_flags": sorted(set(str(x) for x in ingestion_flags if str(x).strip())),
        "wall_max": dict(getattr(chunk_agg, "wall_max", {}) or {}),
        "worst_chunk_score": float(getattr(chunk_agg, "worst_chunk_score", 0.0)),
        "pattern_synergy": float(getattr(chunk_agg, "pattern_synergy", 0.0)),
        "confidence": float(getattr(chunk_agg, "confidence", 0.0)),
        "doc_score": float(getattr(chunk_agg, "doc_score", 0.0)),
        "pair_hits": list(getattr(chunk_agg, "pair_hits", []) or []),
        "triggered_chunk_ids": list(getattr(chunk_agg, "triggered_chunk_ids", []) or []),
        "rule_ids": list(getattr(chunk_agg, "rule_ids", []) or []),
        "per_chunk": per_chunk,
        "text_included": False,
    }


def build_api_risk_event(
    *,
    payload: Mapping[str, Any],
    parsed: Mapping[str, Any],
    fallback_active: bool,
) -> RiskEvent:
    action_types = [str(x) for x in list((((payload.get("policy_trace", {}) or {}).get("action_types", [])) or []))]
    control_outcome = str(payload.get("control_outcome", "ALLOW"))
    risk_score_raw = payload.get("risk_score", None)
    risk_float = None
    if risk_score_raw is not None:
        try:
            risk_float = max(0.0, min(1.0, float(risk_score_raw) / 100.0))
        except (TypeError, ValueError):
            risk_float = None
    reasons = [str(x) for x in list(payload.get("reasons", []) or [])]
    return RiskEvent(
        event_id=new_event_id(),
        timestamp=utc_now_iso(),
        surface="api",
        control_outcome=control_outcome,
        triggers=infer_major_triggers(
            control_outcome=control_outcome,
            action_types=action_types,
            fallback_active=bool(fallback_active),
        ),
        reasons=reasons,
        action_types=action_types,
        trace_id=str(payload.get("trace_id", "")),
        decision_id=str(payload.get("decision_id", "")),
        incident_artifact_id=str(payload.get("incident_artifact_id", "")),
        tenant_id=str(parsed.get("tenant_id", "")),
        session_id=str(parsed.get("session_id") or payload.get("request_id", "")),
        actor_id=str(parsed.get("actor_id") or parsed.get("session_id") or ""),
        step=int((((payload.get("policy_trace", {}) or {}).get("state_step_next", 0)) or 0)),
        severity=str((((payload.get("policy_trace", {}) or {}).get("severity", "")) or "")),
        risk_score=risk_float,
        payload_redacted={
            "control_outcome": control_outcome,
            "reasons": reasons,
            "action_types": action_types,
            "trace_id": str(payload.get("trace_id", "")),
            "decision_id": str(payload.get("decision_id", "")),
            "incident_artifact_id": str(payload.get("incident_artifact_id", "")),
            "tenant_id": str(parsed.get("tenant_id", "")),
            "session_id": str(parsed.get("session_id") or payload.get("request_id", "")),
            "actor_id": str(parsed.get("actor_id") or parsed.get("session_id") or ""),
            "risk_score": int(payload.get("risk_score", 0) or 0),
        },
    )
