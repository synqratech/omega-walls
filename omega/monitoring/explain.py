"""Session explain/timeline builder over monitor JSONL events."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_NON_ALLOW_OUTCOMES = {
    "SOFT_BLOCK",
    "SOURCE_QUARANTINE",
    "TOOL_FREEZE",
    "HUMAN_ESCALATE",
    "REQUIRE_APPROVAL",
    "WARN",
}


def _parse_ts(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_window(value: Optional[str]) -> Optional[timedelta]:
    if not value:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    m = re.fullmatch(r"(\d+)\s*([smhd])", raw)
    if not m:
        return None
    n = int(m.group(1))
    unit = str(m.group(2))
    if n <= 0:
        return None
    if unit == "s":
        return timedelta(seconds=n)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(days=n)


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, Mapping):
                rows.append(parsed)
    return rows


def _legacy_fragments_from_attribution(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for attr in list(row.get("attribution", []) or []):
        if not isinstance(attr, Mapping):
            continue
        doc_id = str(attr.get("doc_id", "")).strip()
        if not doc_id:
            continue
        out.append(
            {
                "doc_id": doc_id,
                "source_id": str(attr.get("source_id", "")).strip(),
                "trust": str(attr.get("trust", "untrusted")).strip() or "untrusted",
                "excerpt_redacted": "",
                "excerpt_sha256": "",
                "contribution": float(attr.get("contribution", 0.0) or 0.0),
            }
        )
    out.sort(key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", ""))))
    return out[:4]


def _normalize_fragments(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fr = row.get("fragments", [])
    out: List[Dict[str, Any]] = []
    if isinstance(fr, Sequence) and not isinstance(fr, (str, bytes)):
        for item in list(fr):
            if not isinstance(item, Mapping):
                continue
            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue
            out.append(
                {
                    "doc_id": doc_id,
                    "source_id": str(item.get("source_id", "")).strip(),
                    "trust": str(item.get("trust", "untrusted")).strip() or "untrusted",
                    "excerpt_redacted": str(item.get("excerpt_redacted", "")),
                    "excerpt_sha256": str(item.get("excerpt_sha256", "")),
                    "contribution": float(item.get("contribution", 0.0) or 0.0),
                }
            )
    if out:
        out.sort(key=lambda x: (-float(x.get("contribution", 0.0)), str(x.get("doc_id", ""))))
        return out[:4]
    return _legacy_fragments_from_attribution(row)


def _normalize_rules(row: Mapping[str, Any]) -> Dict[str, Any]:
    rules = row.get("rules", {})
    if isinstance(rules, Mapping):
        trig = [str(x) for x in list(rules.get("triggered_rules", []) or []) if str(x).strip()]
        reason = [str(x) for x in list(rules.get("reason_codes", []) or []) if str(x).strip()]
        if trig or reason:
            return {"triggered_rules": trig, "reason_codes": reason}
    return {
        "triggered_rules": [str(x) for x in list(row.get("triggered_rules", []) or []) if str(x).strip()],
        "reason_codes": [str(x) for x in list(row.get("reason_codes", []) or []) if str(x).strip()],
    }


def _default_downstream(row: Mapping[str, Any]) -> Dict[str, Any]:
    intended = str(row.get("intended_action", "ALLOW")).strip().upper() or "ALLOW"
    context_prevented = bool(intended in _NON_ALLOW_OUTCOMES)
    return {
        "context_prevented": context_prevented,
        "blocked_doc_ids": [],
        "quarantined_source_ids": [],
        "tool_execution_prevented": bool(intended == "TOOL_FREEZE"),
        "prevented_tools": [],
    }


def _normalize_downstream(row: Mapping[str, Any]) -> Dict[str, Any]:
    downstream = row.get("downstream", {})
    if not isinstance(downstream, Mapping):
        return _default_downstream(row)
    if downstream:
        return {
            "context_prevented": bool(downstream.get("context_prevented", False)),
            "blocked_doc_ids": [str(x) for x in list(downstream.get("blocked_doc_ids", []) or []) if str(x).strip()],
            "quarantined_source_ids": [
                str(x) for x in list(downstream.get("quarantined_source_ids", []) or []) if str(x).strip()
            ],
            "tool_execution_prevented": bool(downstream.get("tool_execution_prevented", False)),
            "prevented_tools": [str(x) for x in list(downstream.get("prevented_tools", []) or []) if str(x).strip()],
        }
    return _default_downstream(row)


def _timeline_sort_key(row: Mapping[str, Any]) -> tuple:
    ts = _parse_ts(str(row.get("ts", "")))
    ts_key = ts.timestamp() if ts is not None else 0.0
    metadata = row.get("metadata", {})
    step = 0
    if isinstance(metadata, Mapping):
        try:
            step = int(metadata.get("step", 0) or 0)
        except Exception:
            step = 0
    return (ts_key, step, str(row.get("decision_id", "")))


def _mttd(timeline: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not timeline:
        return {
            "first_non_allow_index": None,
            "first_non_allow_ts": None,
            "seconds_from_session_start": None,
        }
    first_ts = _parse_ts(str(timeline[0].get("ts", "")))
    for idx, row in enumerate(list(timeline), start=1):
        intended = str(row.get("intended_action", "ALLOW")).strip().upper() or "ALLOW"
        if intended == "ALLOW":
            continue
        hit_ts = _parse_ts(str(row.get("ts", "")))
        delta_sec: Optional[float] = None
        if first_ts is not None and hit_ts is not None:
            delta_sec = round(float((hit_ts - first_ts).total_seconds()), 3)
        return {
            "first_non_allow_index": int(idx),
            "first_non_allow_ts": str(row.get("ts", "")),
            "seconds_from_session_start": delta_sec,
        }
    return {
        "first_non_allow_index": None,
        "first_non_allow_ts": None,
        "seconds_from_session_start": None,
    }


def build_session_explain(
    *,
    events_path: Path,
    session_id: str,
    window: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    rows = list(_iter_jsonl(events_path))
    now = datetime.now(timezone.utc)
    window_delta = _parse_window(window)
    filtered: List[Mapping[str, Any]] = []
    for row in rows:
        if str(row.get("session_id", "")) != sid:
            continue
        if window_delta is not None:
            ts = _parse_ts(str(row.get("ts", "")))
            if ts is None or ts < (now - window_delta):
                continue
        filtered.append(row)
    if not filtered:
        raise ValueError(f"no monitor events found for session '{sid}'")

    filtered.sort(key=_timeline_sort_key)
    if int(limit) > 0:
        filtered = filtered[: int(limit)]

    timeline: List[Dict[str, Any]] = []
    missing_fields_count = 0
    legacy_rows_detected = 0
    for idx, row in enumerate(filtered, start=1):
        fragments = _normalize_fragments(row)
        rules = _normalize_rules(row)
        downstream = _normalize_downstream(row)
        has_fragments_field = "fragments" in row
        has_downstream_field = "downstream" in row
        has_rules_field = ("rules" in row) or ("triggered_rules" in row) or ("reason_codes" in row)
        is_legacy = not (has_fragments_field and has_downstream_field and has_rules_field)
        if is_legacy:
            legacy_rows_detected += 1
        if not has_fragments_field:
            missing_fields_count += 1
        if not has_rules_field:
            missing_fields_count += 1
        if not has_downstream_field:
            missing_fields_count += 1
        primary_fragment = fragments[0] if fragments else None
        timeline.append(
            {
                "index": int(idx),
                "ts": str(row.get("ts", "")),
                "surface": str(row.get("surface", "")),
                "risk_score": float(row.get("risk_score", 0.0) or 0.0),
                "intended_action": str(row.get("intended_action", "ALLOW")),
                "actual_action": str(row.get("actual_action", "ALLOW")),
                "rules": rules,
                "primary_fragment": primary_fragment,
                "fragments": fragments,
                "downstream": downstream,
                "trace_id": str(row.get("trace_id", "")),
                "decision_id": str(row.get("decision_id", "")),
            }
        )

    ts_values = [str(x.get("ts", "")) for x in timeline if str(x.get("ts", "")).strip()]
    surfaces = sorted({str(x.get("surface", "")) for x in timeline if str(x.get("surface", "")).strip()})
    intended_counts: Dict[str, int] = {}
    for row in timeline:
        key = str(row.get("intended_action", "ALLOW")).strip().upper() or "ALLOW"
        intended_counts[key] = int(intended_counts.get(key, 0)) + 1

    return {
        "session_id": sid,
        "summary": {
            "events_count": int(len(timeline)),
            "first_ts": ts_values[0] if ts_values else None,
            "last_ts": ts_values[-1] if ts_values else None,
            "surfaces": surfaces,
            "max_risk": float(max((float(x.get("risk_score", 0.0) or 0.0) for x in timeline), default=0.0)),
            "intended_outcomes_count": dict(sorted(intended_counts.items(), key=lambda kv: kv[0])),
        },
        "timeline": timeline,
        "mttd": _mttd(timeline),
        "data_quality": {
            "missing_fields_count": int(missing_fields_count),
            "legacy_rows_detected": int(legacy_rows_detected),
        },
    }


def explain_as_csv(payload: Mapping[str, Any]) -> str:
    timeline = list(payload.get("timeline", []) or [])
    cols = [
        "session_id",
        "index",
        "ts",
        "surface",
        "risk_score",
        "intended_action",
        "actual_action",
        "triggered_rules",
        "reason_codes",
        "excerpt_redacted",
        "excerpt_sha256",
        "context_prevented",
        "tool_execution_prevented",
        "blocked_doc_ids",
        "quarantined_source_ids",
        "prevented_tools",
        "trace_id",
        "decision_id",
    ]
    rows: List[Dict[str, Any]] = []
    for row in timeline:
        rules = row.get("rules", {}) if isinstance(row.get("rules", {}), Mapping) else {}
        downstream = row.get("downstream", {}) if isinstance(row.get("downstream", {}), Mapping) else {}
        fragment = row.get("primary_fragment", {}) if isinstance(row.get("primary_fragment", {}), Mapping) else {}
        rows.append(
            {
                "session_id": str(payload.get("session_id", "")),
                "index": int(row.get("index", 0) or 0),
                "ts": str(row.get("ts", "")),
                "surface": str(row.get("surface", "")),
                "risk_score": float(row.get("risk_score", 0.0) or 0.0),
                "intended_action": str(row.get("intended_action", "")),
                "actual_action": str(row.get("actual_action", "")),
                "triggered_rules": ";".join(str(x) for x in list(rules.get("triggered_rules", []) or [])),
                "reason_codes": ";".join(str(x) for x in list(rules.get("reason_codes", []) or [])),
                "excerpt_redacted": str(fragment.get("excerpt_redacted", "")),
                "excerpt_sha256": str(fragment.get("excerpt_sha256", "")),
                "context_prevented": bool(downstream.get("context_prevented", False)),
                "tool_execution_prevented": bool(downstream.get("tool_execution_prevented", False)),
                "blocked_doc_ids": ";".join(str(x) for x in list(downstream.get("blocked_doc_ids", []) or [])),
                "quarantined_source_ids": ";".join(
                    str(x) for x in list(downstream.get("quarantined_source_ids", []) or [])
                ),
                "prevented_tools": ";".join(str(x) for x in list(downstream.get("prevented_tools", []) or [])),
                "trace_id": str(row.get("trace_id", "")),
                "decision_id": str(row.get("decision_id", "")),
            }
        )

    class _Sink(list):
        def write(self, chunk: str) -> int:
            self.append(chunk)
            return len(chunk)

    sink: _Sink = _Sink()
    writer = csv.DictWriter(sink, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    return "".join(sink).rstrip("\n")
