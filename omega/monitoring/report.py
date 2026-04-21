"""Monitor report aggregation over local JSONL events."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_BLOCK_LIKE_ACTIONS = {
    "SOFT_BLOCK",
    "SOURCE_QUARANTINE",
    "TOOL_FREEZE",
}

_ESCALATE_ACTIONS = {
    "HUMAN_ESCALATE",
    "REQUIRE_APPROVAL",
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


def _risk_bucket(score: float) -> str:
    val = max(0.0, min(1.0, float(score)))
    if val < 0.3:
        return "0.0-0.3"
    if val < 0.7:
        return "0.3-0.7"
    return "0.7-1.0"


def build_monitor_report(
    *,
    events_path: Path,
    session_id: Optional[str] = None,
    window: Optional[str] = None,
) -> Dict[str, Any]:
    rows = list(_iter_jsonl(events_path))
    now = datetime.now(timezone.utc)
    window_delta = _parse_window(window)
    filtered: List[Mapping[str, Any]] = []
    for row in rows:
        if session_id and str(row.get("session_id", "")) != str(session_id):
            continue
        if window_delta is not None:
            ts = _parse_ts(str(row.get("ts", "")))
            if ts is None or ts < (now - window_delta):
                continue
        filtered.append(row)

    risk_distribution = {"0.0-0.3": 0, "0.3-0.7": 0, "0.7-1.0": 0}
    would_block = 0
    would_escalate = 0
    rule_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum_risk": 0.0})
    hint_counts: Dict[str, int] = defaultdict(int)
    for row in filtered:
        risk_score = float(row.get("risk_score", 0.0) or 0.0)
        risk_distribution[_risk_bucket(risk_score)] += 1
        intended = str(row.get("intended_action", "ALLOW")).strip().upper() or "ALLOW"
        if intended in _BLOCK_LIKE_ACTIONS:
            would_block += 1
        if intended in _ESCALATE_ACTIONS:
            would_escalate += 1
        for rule in list(row.get("triggered_rules", []) or []):
            key = str(rule).strip()
            if not key:
                continue
            rule_counts[key]["count"] += 1.0
            rule_counts[key]["sum_risk"] += float(risk_score)
        hint_raw = row.get("false_positive_hint", "")
        hint = "" if hint_raw is None else str(hint_raw).strip()
        if hint:
            hint_counts[hint] += 1

    top_rules = sorted(
        (
            {
                "rule": key,
                "count": int(val["count"]),
                "avg_score": round((float(val["sum_risk"]) / max(float(val["count"]), 1.0)), 4),
            }
            for key, val in rule_counts.items()
        ),
        key=lambda x: (-int(x["count"]), str(x["rule"])),
    )[:20]
    hints_summary = sorted(
        (
            {
                "hint": key,
                "count": int(count),
            }
            for key, count in hint_counts.items()
        ),
        key=lambda x: (-int(x["count"]), str(x["hint"])),
    )[:20]

    return {
        "total_checks": int(len(filtered)),
        "risk_distribution": risk_distribution,
        "would_block": int(would_block),
        "would_escalate": int(would_escalate),
        "top_rules_triggered": top_rules,
        "false_positive_hints": hints_summary,
    }
