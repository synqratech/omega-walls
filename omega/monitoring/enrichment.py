"""Helpers to enrich monitor events for explain timelines."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from omega.telemetry.redaction import redact_text


_BLOCK_LIKE_ACTIONS = {
    "SOFT_BLOCK",
    "SOURCE_QUARANTINE",
    "TOOL_FREEZE",
    "HUMAN_ESCALATE",
    "REQUIRE_APPROVAL",
    "WARN",
}


def build_redacted_fragments(
    *,
    attribution_rows: Sequence[Mapping[str, Any]],
    item_text_by_doc: Mapping[str, str],
    max_fragments: int = 4,
    max_chars: int = 240,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for row in list(attribution_rows):
        doc_id = str((row or {}).get("doc_id", "")).strip()
        if not doc_id:
            continue
        source_id = str((row or {}).get("source_id", "")).strip()
        trust = str((row or {}).get("trust", "untrusted")).strip() or "untrusted"
        contribution = float((row or {}).get("contribution", 0.0) or 0.0)
        text = str(item_text_by_doc.get(doc_id, ""))
        red = redact_text(text, max_chars=max_chars)
        rows.append(
            {
                "doc_id": doc_id,
                "source_id": source_id,
                "trust": trust,
                "excerpt_redacted": str(red.redacted),
                "excerpt_sha256": str(red.text_sha256),
                "contribution": contribution,
            }
        )
    rows.sort(key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", ""))))
    return rows[: max(1, int(max_fragments))]


def build_downstream_summary(
    *,
    intended_action: str,
    action_types: Sequence[str],
    blocked_doc_ids: Sequence[str],
    quarantined_source_ids: Sequence[str],
    prevented_tools: Sequence[str],
) -> Dict[str, Any]:
    action_set = {str(x).strip().upper() for x in list(action_types) if str(x).strip()}
    blocked_docs = sorted({str(x).strip() for x in list(blocked_doc_ids) if str(x).strip()})
    quarantined_sources = sorted({str(x).strip() for x in list(quarantined_source_ids) if str(x).strip()})
    tools = sorted({str(x).strip() for x in list(prevented_tools) if str(x).strip()})
    intended = str(intended_action or "").strip().upper() or "ALLOW"
    context_prevented = bool(
        intended in _BLOCK_LIKE_ACTIONS
        or blocked_docs
        or quarantined_sources
    )
    tool_execution_prevented = bool("TOOL_FREEZE" in action_set or tools)
    return {
        "context_prevented": bool(context_prevented),
        "blocked_doc_ids": blocked_docs,
        "quarantined_source_ids": quarantined_sources,
        "tool_execution_prevented": bool(tool_execution_prevented),
        "prevented_tools": tools,
    }
