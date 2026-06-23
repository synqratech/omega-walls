from __future__ import annotations

import io
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, ProjectionResult


def polygon_to_rect_px(polygon_px: Any) -> Optional[Dict[str, float]]:
    if not isinstance(polygon_px, (list, tuple)) or not polygon_px:
        return None
    try:
        xs = [float(pt[0]) for pt in polygon_px if isinstance(pt, (list, tuple)) and len(pt) >= 2]
        ys = [float(pt[1]) for pt in polygon_px if isinstance(pt, (list, tuple)) and len(pt) >= 2]
    except (TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}


def _sha256_bytes(raw: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(bytes(raw))
    return h.hexdigest()


def _trace_sha256(text: str) -> str:
    return _sha256_bytes(str(text or "").encode("utf-8", errors="ignore"))


def rect_union(rects: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    rows = [dict(r) for r in list(rects or []) if isinstance(r, Mapping)]
    if not rows:
        return None
    try:
        x_min = min(float(r["x_min"]) for r in rows)
        y_min = min(float(r["y_min"]) for r in rows)
        x_max = max(float(r["x_max"]) for r in rows)
        y_max = max(float(r["y_max"]) for r in rows)
    except (KeyError, TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def expand_rect_px(
    rect_px: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    pad_ratio: float = 0.25,
) -> Optional[Dict[str, float]]:
    try:
        x_min = float(rect_px["x_min"])
        y_min = float(rect_px["y_min"])
        x_max = float(rect_px["x_max"])
        y_max = float(rect_px["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    pad_x = width * max(0.0, float(pad_ratio))
    pad_y = height * max(0.0, float(pad_ratio))
    out = {
        "x_min": max(0.0, x_min - pad_x),
        "y_min": max(0.0, y_min - pad_y),
        "x_max": min(float(max(0, int(image_width))), x_max + pad_x),
        "y_max": min(float(max(0, int(image_height))), y_max + pad_y),
    }
    if out["x_max"] <= out["x_min"] or out["y_max"] <= out["y_min"]:
        return None
    return out


def ensure_min_rect_px(
    rect_px: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    min_width_px: float,
    min_height_px: float,
) -> Optional[Dict[str, float]]:
    try:
        x_min = float(rect_px["x_min"])
        y_min = float(rect_px["y_min"])
        x_max = float(rect_px["x_max"])
        y_max = float(rect_px["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    width = x_max - x_min
    height = y_max - y_min
    need_w = max(0.0, float(min_width_px) - width)
    need_h = max(0.0, float(min_height_px) - height)
    out = {
        "x_min": max(0.0, x_min - (need_w / 2.0)),
        "y_min": max(0.0, y_min - (need_h / 2.0)),
        "x_max": min(float(max(0, int(image_width))), x_max + (need_w / 2.0)),
        "y_max": min(float(max(0, int(image_height))), y_max + (need_h / 2.0)),
    }
    if out["x_max"] <= out["x_min"] or out["y_max"] <= out["y_min"]:
        return None
    return out


def crop_area_ratio(rect_px: Mapping[str, Any], *, image_width: int, image_height: int) -> float:
    if int(image_width) <= 0 or int(image_height) <= 0:
        return 0.0
    try:
        width = max(0.0, float(rect_px["x_max"]) - float(rect_px["x_min"]))
        height = max(0.0, float(rect_px["y_max"]) - float(rect_px["y_min"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, (width * height) / float(max(1, image_width * image_height))))


def _ocr_span_text_by_id(span_lookup: Mapping[str, Any], span_ids: Sequence[str]) -> str:
    parts: List[str] = []
    for span_id in list(span_ids or []):
        span = span_lookup.get(str(span_id))
        if span is None:
            continue
        text = str(getattr(span, "text", "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _ordered_span_ids(span_lookup: Mapping[str, Any], span_ids: Sequence[str]) -> List[str]:
    rows = []
    for span_id in [str(x) for x in list(span_ids or []) if str(x).strip()]:
        span = span_lookup.get(span_id)
        if span is None:
            continue
        rows.append(
            (
                int(getattr(span, "provider_order", 10**9) if getattr(span, "provider_order", None) is not None else 10**9),
                span_id,
            )
        )
    return [span_id for _, span_id in sorted(rows)]


def _windowed_span_groups(
    *,
    span_lookup: Mapping[str, Any],
    triggered_span_ids: Sequence[str],
    max_spans_per_group: int,
    max_groups: int,
) -> List[List[str]]:
    ordered = _ordered_span_ids(span_lookup, triggered_span_ids)
    if not ordered:
        return []
    size = max(1, int(max_spans_per_group))
    groups: List[List[str]] = []
    for start in range(0, len(ordered), size):
        groups.append(list(ordered[start : start + size]))
        if len(groups) >= max(1, int(max_groups)):
            break
    return groups


def _contextual_span_groups(
    *,
    span_lookup: Mapping[str, Any],
    triggered_span_ids: Sequence[str],
    supporting_span_ids: Sequence[str],
    context_radius: int,
    max_context_spans: int,
    max_groups: int,
) -> List[List[str]]:
    support_ordered = _ordered_span_ids(span_lookup, supporting_span_ids)
    if not support_ordered:
        support_ordered = _ordered_span_ids(span_lookup, triggered_span_ids)
    if not support_ordered:
        return []
    pos_by_id = {span_id: idx for idx, span_id in enumerate(support_ordered)}
    groups: List[List[str]] = []
    seen: set[tuple[str, ...]] = set()
    for span_id in _ordered_span_ids(span_lookup, triggered_span_ids):
        if span_id not in pos_by_id:
            continue
        center = pos_by_id[span_id]
        left = max(0, int(center - max(0, int(context_radius))))
        right = min(len(support_ordered), int(center + max(0, int(context_radius)) + 1))
        group = list(support_ordered[left:right])
        if len(group) > int(max_context_spans):
            start = max(0, int(center - (int(max_context_spans) // 2)))
            end = min(len(support_ordered), start + int(max_context_spans))
            start = max(0, end - int(max_context_spans))
            group = list(support_ordered[start:end])
        key = tuple(group)
        if not key or key in seen:
            continue
        seen.add(key)
        groups.append(group)
        if len(groups) >= max(1, int(max_groups)):
            break
    return groups


def crop_image_bytes(*, raw_bytes: bytes, rect_px: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(raw_bytes)) as img:
            x_min = int(max(0, round(float(rect_px["x_min"]))))
            y_min = int(max(0, round(float(rect_px["y_min"]))))
            x_max = int(min(img.size[0], round(float(rect_px["x_max"]))))
            y_max = int(min(img.size[1], round(float(rect_px["y_max"]))))
            if x_max <= x_min or y_max <= y_min:
                return None
            cropped = img.crop((x_min, y_min, x_max, y_max))
            out = io.BytesIO()
            cropped.save(out, format="PNG")
            cropped_raw = out.getvalue()
            return {
                "mime": "image/png",
                "raw_bytes": cropped_raw,
                "size_bytes": int(len(cropped_raw)),
                "sha256": _sha256_bytes(cropped_raw),
                "width": int(cropped.size[0]),
                "height": int(cropped.size[1]),
            }
    except Exception:
        return None
    return None


def build_ocr_adjudication_items(
    *,
    request_id: str,
    source_id: str,
    source_type: str,
    trust: str,
    file_bytes: bytes,
    triggered_span_ids: Sequence[str],
    matched_span_ids: Sequence[str] | None = None,
    supporting_span_ids: Sequence[str] | None = None,
    source_walls: Sequence[str],
    span_lookup: Mapping[str, Any],
    source_image_meta: Mapping[str, Any],
    max_group_spans: int = 8,
    tile_span_limit: int = 1,
    crop_strategy: str = "contextual",
    context_span_radius: int = 2,
    max_context_spans: int = 5,
    max_tiles: int = 2,
    max_crop_area_ratio: float = 0.35,
    min_crop_width_px: float = 160.0,
    min_crop_height_px: float = 72.0,
    include_candidate_text: bool = True,
    variant_id: str = "contextual_image_text",
    register_image_payload: Callable[..., Mapping[str, Any]],
) -> Tuple[List[ContentItem], Dict[str, Any]]:
    strategy = str(crop_strategy or "contextual").strip().lower()
    exact_match_ids = [str(x) for x in list(matched_span_ids or []) if str(x).strip()]
    support_ids = [str(x) for x in list(supporting_span_ids or triggered_span_ids or []) if str(x).strip()]
    if strategy == "tiny":
        groups = _windowed_span_groups(
            span_lookup=span_lookup,
            triggered_span_ids=triggered_span_ids,
            max_spans_per_group=max_group_spans,
            max_groups=max_tiles,
        )
    else:
        groups = _contextual_span_groups(
            span_lookup=span_lookup,
            triggered_span_ids=triggered_span_ids,
            supporting_span_ids=support_ids,
            context_radius=context_span_radius,
            max_context_spans=max_context_spans,
            max_groups=max_tiles,
        )
    trace: Dict[str, Any] = {
        "triggered_span_ids": [str(x) for x in list(triggered_span_ids or []) if str(x).strip()],
        "matched_span_ids": list(exact_match_ids),
        "supporting_span_ids": support_ids,
        "source_walls": [str(x) for x in list(source_walls or []) if str(x).strip()],
        "variant_id": str(variant_id),
        "crop_strategy": str(strategy),
        "exact_attribution": bool(exact_match_ids),
        "tile_count": 0,
        "tiles": [],
        "reason": "ok",
    }
    if not groups:
        trace["reason"] = "no_triggered_spans"
        return [], trace
    image_width = max(int(source_image_meta.get("width", 0) or 0), 0)
    image_height = max(int(source_image_meta.get("height", 0) or 0), 0)
    items: List[ContentItem] = []
    for tile_idx, group in enumerate(groups):
        if strategy == "tiny":
            local_ids = list(group[: max(1, int(tile_span_limit))]) if len(group) > int(tile_span_limit) else list(group)
        else:
            local_ids = list(group[: max(1, int(max_context_spans))]) if len(group) > int(max_context_spans) else list(group)
        rects = []
        for span_id in local_ids:
            span = span_lookup.get(str(span_id))
            if span is None:
                continue
            rect_px = polygon_to_rect_px(getattr(span, "polygon_px", None))
            if rect_px is not None:
                rects.append(rect_px)
            image_width = max(image_width, int(getattr(span, "image_width", 0) or 0))
            image_height = max(image_height, int(getattr(span, "image_height", 0) or 0))
        union_rect = rect_union(rects)
        if union_rect is None:
            continue
        expanded_rect = expand_rect_px(
            union_rect,
            image_width=image_width,
            image_height=image_height,
            pad_ratio=0.25,
        )
        if expanded_rect is None:
            continue
        expanded_rect = ensure_min_rect_px(
            expanded_rect,
            image_width=image_width,
            image_height=image_height,
            min_width_px=float(min_crop_width_px),
            min_height_px=float(min_crop_height_px),
        )
        if expanded_rect is None:
            continue
        area_ratio = crop_area_ratio(expanded_rect, image_width=image_width, image_height=image_height)
        if area_ratio > float(max_crop_area_ratio):
            trace["tiles"].append(
                {
                    "tile_index": int(tile_idx),
                    "context_span_ids": list(local_ids),
                    "variant_id": str(variant_id),
                    "crop_rect_px": dict(expanded_rect),
                    "crop_area_ratio": float(area_ratio),
                    "skipped": "crop_too_large",
                }
            )
            continue
        crop_meta = crop_image_bytes(raw_bytes=bytes(file_bytes or b""), rect_px=expanded_rect)
        if not isinstance(crop_meta, Mapping):
            continue
        crop_raw = bytes(crop_meta.get("raw_bytes") or b"")
        if not crop_raw:
            continue
        registered_image = dict(
            register_image_payload(
                raw=crop_raw,
                mime=str(crop_meta.get("mime", "image/png")),
                sha256=str(crop_meta.get("sha256", "")),
                role="untrusted_visual_content",
                width=int(crop_meta.get("width", 0) or 0),
                height=int(crop_meta.get("height", 0) or 0),
            )
        )
        if not str(registered_image.get("bytes_ref", "")).strip():
            raise ValueError("OCR adjudication image registrar must return bytes_ref")
        forbidden_media_keys = {"bytes_b64", "raw_bytes", "file_bytes", "image_bytes"}
        if forbidden_media_keys.intersection(registered_image):
            raise ValueError("OCR adjudication image registrar returned raw media")
        local_candidate_ids = [span_id for span_id in list(local_ids) if span_id in set(exact_match_ids)]
        candidate_text = _ocr_span_text_by_id(span_lookup, local_candidate_ids)
        semantic_text = str(candidate_text) if bool(include_candidate_text) and bool(exact_match_ids) else ""
        meta = {
            "attachment_chunk_kind": "image_semantic",
            "attachment_modality": "image_semantic",
            "semantic_image": registered_image,
            "semantic_trace_hints": {
                "kind": "ocr_targeted_adjudication",
                "variant_id": str(variant_id),
                "crop_strategy": str(strategy),
                "triggered_span_ids": [str(x) for x in list(triggered_span_ids or []) if str(x).strip()],
                "context_span_ids": list(local_ids),
                "candidate_span_ids": list(local_candidate_ids),
                "source_walls": [str(x) for x in list(source_walls or []) if str(x).strip()],
                "exact_attribution": bool(exact_match_ids),
            },
            "ocr_adjudication_target": {
                "variant_id": str(variant_id),
                "crop_strategy": str(strategy),
                "triggered_span_ids": [str(x) for x in list(triggered_span_ids or []) if str(x).strip()],
                "context_span_ids": list(local_ids),
                "candidate_span_ids": list(local_candidate_ids),
                "source_walls": [str(x) for x in list(source_walls or []) if str(x).strip()],
                "exact_attribution": bool(exact_match_ids),
                "crop_rect_px": dict(expanded_rect),
                "crop_area_ratio": float(area_ratio),
                "crop_sha256": str(crop_meta.get("sha256", "")),
                "candidate_text_sha256": _trace_sha256(candidate_text),
                "tile_index": int(tile_idx),
            },
        }
        items.append(
            ContentItem(
                doc_id=f"{request_id}:ocr-adjudication:{tile_idx:02d}",
                source_id=str(source_id),
                source_type=str(source_type),
                trust=str(trust),
                text=str(semantic_text),
                meta=meta,
            )
        )
        trace["tiles"].append(
            {
                "tile_index": int(tile_idx),
                "context_span_ids": list(local_ids),
                "candidate_span_ids": list(local_candidate_ids),
                "variant_id": str(variant_id),
                "crop_rect_px": dict(expanded_rect),
                "crop_area_ratio": float(area_ratio),
                "crop_sha256": str(crop_meta.get("sha256", "")),
                "exact_attribution": bool(exact_match_ids),
            }
        )
    trace["tile_count"] = int(len(items))
    if not items:
        trace["reason"] = "no_local_tiles"
    return items, trace


def build_ocr_adjudication_matrix_items(
    *,
    request_id: str,
    source_id: str,
    source_type: str,
    trust: str,
    file_bytes: bytes,
    triggered_span_ids: Sequence[str],
    supporting_span_ids: Sequence[str] | None,
    matched_span_ids: Sequence[str] | None,
    source_walls: Sequence[str],
    span_lookup: Mapping[str, Any],
    source_image_meta: Mapping[str, Any],
    register_image_payload: Callable[..., Mapping[str, Any]],
) -> Tuple[List[ContentItem], Dict[str, Any]]:
    variants = [
        {
            "variant_id": "tiny_image_only",
            "crop_strategy": "tiny",
            "include_candidate_text": False,
            "tile_span_limit": 1,
            "context_span_radius": 0,
            "max_context_spans": 1,
        },
        {
            "variant_id": "tiny_image_text",
            "crop_strategy": "tiny",
            "include_candidate_text": True,
            "tile_span_limit": 1,
            "context_span_radius": 0,
            "max_context_spans": 1,
        },
        {
            "variant_id": "contextual_image_only",
            "crop_strategy": "contextual",
            "include_candidate_text": False,
            "tile_span_limit": 1,
            "context_span_radius": 2,
            "max_context_spans": 5,
        },
        {
            "variant_id": "contextual_image_text",
            "crop_strategy": "contextual",
            "include_candidate_text": True,
            "tile_span_limit": 1,
            "context_span_radius": 2,
            "max_context_spans": 5,
        },
    ]
    items: List[ContentItem] = []
    matrix_trace: Dict[str, Any] = {"variants": []}
    for variant in variants:
        sub_items, sub_trace = build_ocr_adjudication_items(
            request_id=request_id,
            source_id=source_id,
            source_type=source_type,
            trust=trust,
            file_bytes=file_bytes,
            triggered_span_ids=triggered_span_ids,
            matched_span_ids=matched_span_ids,
            supporting_span_ids=supporting_span_ids,
            source_walls=source_walls,
            span_lookup=span_lookup,
            source_image_meta=source_image_meta,
            crop_strategy=str(variant["crop_strategy"]),
            tile_span_limit=int(variant["tile_span_limit"]),
            context_span_radius=int(variant["context_span_radius"]),
            max_context_spans=int(variant["max_context_spans"]),
            include_candidate_text=bool(variant["include_candidate_text"]),
            variant_id=str(variant["variant_id"]),
            register_image_payload=register_image_payload,
        )
        items.extend(sub_items)
        matrix_trace["variants"].append(dict(sub_trace))
    return items, matrix_trace


def interpret_ocr_adjudication_projection(
    *,
    projection: ProjectionResult,
    source_walls: Sequence[str],
    threshold: float,
    benign_min_confidence: float = 0.75,
    malicious_min_confidence: float = 0.55,
) -> Dict[str, Any]:
    vec = np.asarray(projection.v, dtype=float)
    evidence = projection.evidence
    api_match = dict(evidence.matches.get("api_perception", {})) if isinstance(evidence.matches, Mapping) else {}
    wall_names = [
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    ]
    positive_walls = [
        wall_names[idx]
        for idx in range(min(len(wall_names), len(vec)))
        if float(vec[idx]) >= float(threshold)
    ]
    raw_scores = [float(x) for x in list(getattr(evidence, "debug_scores_raw", []) or [])]
    polarity = [int(x) for x in list(getattr(evidence, "polarity", []) or [])]
    negative_walls = [
        wall_names[idx]
        for idx in range(min(len(wall_names), len(raw_scores), len(polarity)))
        if float(raw_scores[idx]) < 0.0 or int(polarity[idx]) < 0
    ]
    source_wall_set = {str(x) for x in list(source_walls or []) if str(x).strip()}
    positive_source_walls = [wall for wall in positive_walls if wall in source_wall_set]
    negative_source_walls = [wall for wall in negative_walls if wall in source_wall_set]
    semantic_status = str(api_match.get("vision_semantic_status") or api_match.get("semantic_status") or "unknown")
    confidence = float(api_match.get("confidence", 0.0) or 0.0)
    defensive_context = bool(api_match.get("defensive_context", False))
    directive_intent_raw = api_match.get("directive_intent", {})
    directive_intent = dict(directive_intent_raw) if isinstance(directive_intent_raw, Mapping) else {}
    source_directive_intent = {
        wall: bool(directive_intent.get(wall, wall in positive_source_walls))
        for wall in list(source_wall_set)
    }
    if (
        positive_source_walls
        and any(bool(source_directive_intent.get(wall, False)) for wall in positive_source_walls)
        and confidence >= float(malicious_min_confidence)
    ):
        result = "live_attack"
    elif defensive_context and (negative_source_walls or negative_walls) and confidence >= float(benign_min_confidence):
        result = "quoted_or_defensive"
    elif (
        not positive_walls
        and not negative_walls
        and not positive_source_walls
        and not any(bool(source_directive_intent.get(wall, False)) for wall in list(source_wall_set))
        and not defensive_context
        and confidence >= float(benign_min_confidence)
    ):
        result = "benign_ui"
    else:
        result = "insufficient_context"
    return {
        "result": str(result),
        "confirmed_walls": list(positive_source_walls),
        "negative_walls": list(negative_source_walls),
        "confidence": float(confidence),
        "defensive_context": bool(defensive_context),
        "directive_intent": dict(source_directive_intent),
        "semantic_status": semantic_status,
    }
