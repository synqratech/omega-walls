"""FW-005 structured log contract models and mapping helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now_iso_millis() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class AttributionItem(BaseModel):
    source: str = Field(min_length=1)
    chunk_hash: str = Field(min_length=1)
    score_contrib: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ErrorInfo(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


class OmegaLogEvent(BaseModel):
    ts: str = Field(min_length=1)
    level: str = Field(min_length=1)
    event: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=1.0)
    intended_action: str = Field(min_length=1)
    actual_action: str = Field(min_length=1)
    triggered_rules: List[str] = Field(default_factory=list)
    attribution: List[AttributionItem] = Field(default_factory=list)
    fp_hint: Optional[str] = None
    error: Optional[ErrorInfo] = None
    intended_action_native: Optional[str] = None
    actual_action_native: Optional[str] = None
    risk_score_native: Optional[float] = None
    trace_id: Optional[str] = None
    decision_id: Optional[str] = None
    surface: Optional[str] = None
    input_type: Optional[str] = None
    input_length: Optional[int] = Field(default=None, ge=0)
    source_type: Optional[str] = None
    chunk_hash: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("level", mode="before")
    @classmethod
    def _norm_level(cls, value: Any) -> str:
        raw = str(value or "").strip().upper()
        allowed = {"DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"}
        if raw not in allowed:
            raise ValueError("level must be DEBUG|INFO|WARN|ERROR|CRITICAL")
        return raw

    @field_validator("mode", mode="before")
    @classmethod
    def _norm_mode(cls, value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw not in {"monitor", "enforce"}:
            raise ValueError("mode must be monitor|enforce")
        return raw

    @field_validator("session_id", mode="before")
    @classmethod
    def _norm_session_id(cls, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("session_id must be non-empty")
        return raw

    @field_validator("intended_action", "actual_action", mode="before")
    @classmethod
    def _norm_action(cls, value: Any) -> str:
        raw = str(value or "").strip().upper()
        allowed = {"ALLOW", "BLOCK", "QUARANTINE", "ESCALATE"}
        if raw not in allowed:
            raise ValueError("action must be ALLOW|BLOCK|QUARANTINE|ESCALATE")
        return raw

    @model_validator(mode="after")
    def _monitor_actual_allow(self) -> "OmegaLogEvent":
        if self.mode == "monitor" and self.actual_action != "ALLOW":
            raise ValueError("actual_action must be ALLOW in monitor mode")
        return self


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_api_risk_score(value: Any) -> tuple[float, Optional[float]]:
    if value is None:
        return 0.0, None
    raw = float(value)
    if raw > 1.0:
        return max(0.0, min(1.0, raw / 100.0)), raw
    return max(0.0, min(1.0, raw)), raw


def canonical_action(
    *,
    action_native: str,
    action_types: Optional[Sequence[str]] = None,
) -> str:
    items = {str(action_native or "").strip().upper()}
    for action in list(action_types or []):
        item = str(action or "").strip().upper()
        if item:
            items.add(item)
    if {"HUMAN_ESCALATE", "REQUIRE_APPROVAL", "ESCALATE"} & items:
        return "ESCALATE"
    if {"SOFT_BLOCK", "TOOL_FREEZE", "BLOCK"} & items:
        return "BLOCK"
    if {"SOURCE_QUARANTINE", "WARN", "QUARANTINE"} & items:
        return "QUARANTINE"
    return "ALLOW"


def map_attribution(
    *,
    rows: Sequence[Mapping[str, Any]],
    fragments: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[AttributionItem]:
    frag_hash: Dict[tuple[str, str], str] = {}
    for fragment in list(fragments or []):
        source = str((fragment or {}).get("source_id", "")).strip()
        doc = str((fragment or {}).get("doc_id", "")).strip()
        hashed = str((fragment or {}).get("excerpt_sha256", "")).strip()
        if source and doc and hashed:
            frag_hash[(doc, source)] = hashed

    out: List[AttributionItem] = []
    for row in list(rows):
        source = str((row or {}).get("source", "")).strip() or str((row or {}).get("source_id", "")).strip()
        doc = str((row or {}).get("doc_id", "")).strip()
        if not source and not doc:
            continue
        chunk_hash = frag_hash.get((doc, source), "")
        if not chunk_hash:
            chunk_hash = str((row or {}).get("chunk_hash", "")).strip()
        if not chunk_hash:
            chunk_hash = _stable_hash(f"{doc}|{source}")
        contrib_raw = (row or {}).get("score_contrib", None)
        if contrib_raw is None:
            contrib_raw = (row or {}).get("contribution", (row or {}).get("score", 0.0))
        out.append(
            AttributionItem(
                source=source or "unknown",
                chunk_hash=chunk_hash,
                score_contrib=max(0.0, float(contrib_raw or 0.0)),
            )
        )
    return out


def infer_level(*, risk_score: float, canonical_actual_action: str, error: Optional[ErrorInfo] = None) -> str:
    if error is not None:
        return "ERROR"
    if canonical_actual_action != "ALLOW":
        return "WARN"
    if float(risk_score) >= 0.7:
        return "WARN"
    return "INFO"


def make_log_event(
    *,
    event: str,
    session_id: str,
    mode: str,
    engine_version: str,
    risk_score: float,
    intended_action_native: str,
    actual_action_native: str,
    action_types: Optional[Sequence[str]] = None,
    triggered_rules: Optional[Sequence[str]] = None,
    attribution_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    fragments: Optional[Sequence[Mapping[str, Any]]] = None,
    fp_hint: Optional[str] = None,
    error: Optional[ErrorInfo] = None,
    ts: Optional[str] = None,
    trace_id: Optional[str] = None,
    decision_id: Optional[str] = None,
    surface: Optional[str] = None,
    input_type: Optional[str] = None,
    input_length: Optional[int] = None,
    source_type: Optional[str] = None,
    chunk_hash: Optional[str] = None,
    risk_score_native: Optional[float] = None,
) -> OmegaLogEvent:
    canonical_intended = canonical_action(action_native=intended_action_native, action_types=action_types)
    canonical_actual = canonical_action(action_native=actual_action_native, action_types=action_types)
    if str(mode).strip().lower() == "monitor":
        canonical_actual = "ALLOW"
    event_level = infer_level(risk_score=float(risk_score), canonical_actual_action=canonical_actual, error=error)
    return OmegaLogEvent(
        ts=str(ts or utc_now_iso_millis()),
        level=event_level,
        event=str(event),
        session_id=str(session_id),
        mode=str(mode),
        engine_version=str(engine_version),
        risk_score=max(0.0, min(1.0, float(risk_score))),
        intended_action=canonical_intended,
        actual_action=canonical_actual,
        triggered_rules=[str(x) for x in list(triggered_rules or []) if str(x).strip()],
        attribution=map_attribution(rows=list(attribution_rows or []), fragments=list(fragments or [])),
        fp_hint=(str(fp_hint) if fp_hint else None),
        error=error,
        intended_action_native=str(intended_action_native or ""),
        actual_action_native=str(actual_action_native or ""),
        risk_score_native=(float(risk_score_native) if risk_score_native is not None else None),
        trace_id=(str(trace_id) if trace_id else None),
        decision_id=(str(decision_id) if decision_id else None),
        surface=(str(surface) if surface else None),
        input_type=(str(input_type) if input_type else None),
        input_length=(int(input_length) if input_length is not None else None),
        source_type=(str(source_type) if source_type else None),
        chunk_hash=(str(chunk_hash) if chunk_hash else None),
    )
