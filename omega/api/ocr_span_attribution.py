from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _append_unique(target: List[str], values: Iterable[Any]) -> None:
    seen = set(target)
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        target.append(text)
        seen.add(text)


def _tokens(text: Any) -> List[str]:
    return [tok for tok in _normalize_text(text).split(" ") if tok]


def _window_terms(raw_windows: Any) -> List[str]:
    out: List[str] = []
    for row in list(raw_windows or []):
        if not isinstance(row, Mapping):
            continue
        _append_unique(out, [row.get("a"), row.get("b")])
        _append_unique(out, row.get("markers", []) or [])
    return out


def _span_text(span: Any) -> str:
    return str(getattr(span, "text", "") or "")


def _span_window_matches(
    *,
    span_lookup: Mapping[str, Any],
    chunk_span_ids: Sequence[str],
    marker_norm: str,
) -> List[str]:
    marker_tokens = _tokens(marker_norm)
    if not marker_tokens:
        return []
    normalized_ids = [str(x) for x in list(chunk_span_ids or []) if str(x).strip()]
    best: List[str] = []
    best_score = 0.0
    max_window = min(6, max(1, len(normalized_ids)))
    marker_token_set = set(marker_tokens)
    for start in range(len(normalized_ids)):
        for end in range(start + 1, min(len(normalized_ids), start + max_window) + 1):
            window_ids = normalized_ids[start:end]
            window_text = " ".join(_span_text(span_lookup.get(span_id)) for span_id in window_ids)
            window_norm = _normalize_text(window_text)
            if not window_norm:
                continue
            if marker_norm in window_norm:
                return list(window_ids)
            window_tokens = _tokens(window_norm)
            if not window_tokens:
                continue
            overlap = marker_token_set.intersection(window_tokens)
            if not overlap:
                continue
            marker_cov = float(len(overlap)) / float(len(marker_token_set))
            window_cov = float(len(overlap)) / float(len(set(window_tokens)))
            score = (0.75 * marker_cov) + (0.25 * window_cov)
            if marker_cov >= 0.66 and (score > best_score or (score == best_score and len(window_ids) < len(best))):
                best = list(window_ids)
                best_score = score
    return best


def extract_wall_signal_hints(matches: Mapping[str, Any], *, wall: str) -> Dict[str, List[str]]:
    hints_root = matches.get("wall_signal_hints", {})
    if isinstance(hints_root, Mapping):
        hint_row = hints_root.get(str(wall), {})
        if isinstance(hint_row, Mapping) and hint_row:
            markers: List[str] = []
            _append_unique(markers, hint_row.get("markers", []) or [])
            _append_unique(markers, hint_row.get("phrases", []) or [])
            _append_unique(markers, _window_terms(hint_row.get("windows", []) or []))
            return {"markers": markers}

    markers: List[str] = []
    wall_name = str(wall)
    if wall_name == "override_instructions":
        _append_unique(markers, matches.get("anchors", []) or [])
        _append_unique(markers, matches.get("phrases", []) or [])
        _append_unique(markers, _window_terms(matches.get("windows", []) or []))
        _append_unique(markers, _window_terms(matches.get("real_override_intent_windows", []) or []))
    elif wall_name == "secret_exfiltration":
        _append_unique(markers, matches.get("secret_leak_markers", []) or [])
        _append_unique(markers, _window_terms(matches.get("windows", []) or []))
        _append_unique(markers, _window_terms(matches.get("promptshield_secret_emit_windows", []) or []))
        _append_unique(markers, _window_terms(matches.get("contact_exfil_windows", []) or []))
    elif wall_name == "tool_or_action_abuse":
        _append_unique(markers, matches.get("tool_tokens", []) or [])
        _append_unique(markers, matches.get("tool_context_gated_tokens", []) or [])
        _append_unique(markers, _window_terms(matches.get("sql_db_windows", []) or []))
        _append_unique(markers, _window_terms(matches.get("context_required_deceptive_correction_windows", []) or []))
    elif wall_name == "policy_evasion":
        _append_unique(markers, matches.get("evasion_markers", []) or [])
        _append_unique(markers, _window_terms(matches.get("windows", []) or []))
    return {"markers": markers}


def matched_ocr_span_ids_for_item(
    *,
    item_text: str,
    item_meta: Mapping[str, Any],
    ocr_span_lookup: Mapping[str, Any],
    matches: Mapping[str, Any],
    active_walls: Sequence[str],
) -> List[str]:
    chunk_span_ids = [str(x) for x in list(item_meta.get("ocr_span_ids", []) or []) if str(x).strip()]
    if not chunk_span_ids:
        return []
    item_text_norm = str(item_text or "")
    item_text_lower = item_text_norm.lower()
    if not item_text_lower.strip():
        return []
    chunk_start = int(item_meta.get("ocr_char_start", 0) or 0)
    matched_ids: List[str] = []
    for wall in list(active_walls or []):
        hint_row = extract_wall_signal_hints(matches, wall=str(wall))
        for marker in list(hint_row.get("markers", []) or []):
            marker_norm = _normalize_text(marker)
            if not marker_norm:
                continue
            start_at = 0
            while True:
                found_at = item_text_lower.find(marker_norm, start_at)
                if found_at < 0:
                    break
                local_start = int(found_at)
                local_end = int(found_at + len(marker_norm))
                global_start = int(chunk_start + local_start)
                global_end = int(chunk_start + local_end)
                for span_id in chunk_span_ids:
                    span = ocr_span_lookup.get(str(span_id))
                    if span is None:
                        continue
                    span_start = getattr(span, "char_start", None)
                    span_end = getattr(span, "char_end", None)
                    if span_start is None or span_end is None:
                        continue
                    if int(span_end) <= global_start or int(span_start) >= global_end:
                        continue
                    if str(span_id) not in matched_ids:
                        matched_ids.append(str(span_id))
                start_at = found_at + max(1, len(marker_norm))
            if matched_ids:
                continue
            for span_id in _span_window_matches(
                span_lookup=ocr_span_lookup,
                chunk_span_ids=chunk_span_ids,
                marker_norm=marker_norm,
            ):
                if str(span_id) not in matched_ids:
                    matched_ids.append(str(span_id))
    return matched_ids
