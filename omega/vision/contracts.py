from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Optional, Sequence, Tuple



OCR_JOINED_SECURITY_TOKEN_RE = re.compile(
    r"\b(dump|reveal|show|print|send|upload|export)(secret|token|password|credentials|credential|key)\b",
    flags=re.IGNORECASE,
)


def repair_ocr_token_boundaries(text: str) -> str:
    """Repair only high-confidence security verb/object joins common in OCR output."""
    return OCR_JOINED_SECURITY_TOKEN_RE.sub(lambda match: f"{match.group(1)} {match.group(2)}", str(text or ""))

@dataclass(frozen=True)
class OCRSpan:
    span_id: str
    text: str
    confidence: Optional[float] = None
    polygon_px: Optional[Tuple[Tuple[float, float], ...]] = None
    image_width: int = 0
    image_height: int = 0
    provider_order: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


@dataclass(frozen=True)
class OCRQualityPolicy:
    min_confidence: float = 0.50
    max_spans: int = 256
    max_span_chars: int = 512
    require_geometry: bool = True
    min_polygon_area_px: float = 4.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.min_confidence)) or not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError("ocr min_confidence must be in [0,1]")
        if int(self.max_spans) <= 0:
            raise ValueError("ocr max_spans must be > 0")
        if int(self.max_span_chars) <= 0:
            raise ValueError("ocr max_span_chars must be > 0")
        if not math.isfinite(float(self.min_polygon_area_px)) or float(self.min_polygon_area_px) < 0.0:
            raise ValueError("ocr min_polygon_area_px must be finite and >= 0")


@dataclass(frozen=True)
class OCRQualitySummary:
    total_spans: int = 0
    kept_spans: int = 0
    dropped_empty: int = 0
    dropped_low_confidence: int = 0
    dropped_invalid_confidence: int = 0
    dropped_invalid_geometry: int = 0
    dropped_over_limit: int = 0
    clipped_span_texts: int = 0
    mean_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    geometry_coverage_ratio: float = 0.0
    status: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_spans": int(self.total_spans),
            "kept_spans": int(self.kept_spans),
            "dropped_empty": int(self.dropped_empty),
            "dropped_low_confidence": int(self.dropped_low_confidence),
            "dropped_invalid_confidence": int(self.dropped_invalid_confidence),
            "dropped_invalid_geometry": int(self.dropped_invalid_geometry),
            "dropped_over_limit": int(self.dropped_over_limit),
            "clipped_span_texts": int(self.clipped_span_texts),
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "geometry_coverage_ratio": float(self.geometry_coverage_ratio),
            "status": str(self.status),
        }


def _valid_confidence(value: Optional[float]) -> tuple[bool, Optional[float]]:
    if value is None:
        return True, None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False, None
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return False, None
    return True, score


def _polygon_rect(
    polygon: Optional[Sequence[Sequence[float]]],
    *,
    image_width: int,
    image_height: int,
    min_area: float,
) -> tuple[bool, Optional[Tuple[Tuple[float, float], ...]], float]:
    if not polygon:
        return False, None, 0.0
    points: list[tuple[float, float]] = []
    try:
        for point in polygon:
            if len(point) < 2:
                return False, None, 0.0
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return False, None, 0.0
            if x < 0.0 or y < 0.0:
                return False, None, 0.0
            if image_width > 0 and x > float(image_width):
                return False, None, 0.0
            if image_height > 0 and y > float(image_height):
                return False, None, 0.0
            points.append((x, y))
    except (TypeError, ValueError):
        return False, None, 0.0
    if len(points) < 3:
        return False, None, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    area = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    if area < float(min_area):
        return False, None, area
    return True, tuple(points), area


def normalize_ocr_spans(
    spans: Sequence[OCRSpan],
    *,
    max_chars: int,
    image_width: int,
    image_height: int,
    policy: OCRQualityPolicy,
) -> tuple[list[OCRSpan], OCRQualitySummary]:
    """Validate and bound OCR output before it becomes security evidence."""
    out: list[OCRSpan] = []
    total_chars = 0
    dropped_empty = dropped_low = dropped_bad_conf = dropped_geom = dropped_limit = clipped = 0
    kept_confidences: list[float] = []
    geometry_area = 0.0
    seen_ids: set[str] = set()
    rows = list(spans or [])
    for idx, span in enumerate(rows):
        if len(out) >= int(policy.max_spans) or total_chars >= int(max_chars):
            dropped_limit += 1
            continue
        text = repair_ocr_token_boundaries(" ".join(str(span.text or "").split()).strip())
        if not text:
            dropped_empty += 1
            continue
        valid_conf, confidence = _valid_confidence(span.confidence)
        if not valid_conf:
            dropped_bad_conf += 1
            continue
        if confidence is not None and confidence < float(policy.min_confidence):
            dropped_low += 1
            continue
        valid_geom, polygon, area = _polygon_rect(
            span.polygon_px,
            image_width=int(image_width),
            image_height=int(image_height),
            min_area=float(policy.min_polygon_area_px),
        )
        if bool(policy.require_geometry) and not valid_geom:
            dropped_geom += 1
            continue
        if len(text) > int(policy.max_span_chars):
            text = text[: int(policy.max_span_chars)].strip()
            clipped += 1
        remaining = int(max_chars) - total_chars
        if remaining <= 0:
            dropped_limit += 1
            continue
        if len(text) > remaining:
            text = text[:remaining].strip()
            clipped += 1
        if not text:
            dropped_limit += 1
            continue
        raw_id = str(span.span_id or f"ocr-span-{idx:04d}").strip() or f"ocr-span-{idx:04d}"
        span_id = raw_id
        suffix = 1
        while span_id in seen_ids:
            span_id = f"{raw_id}-{suffix}"
            suffix += 1
        seen_ids.add(span_id)
        char_start = total_chars if not out else total_chars + 1
        char_end = char_start + len(text)
        out.append(
            OCRSpan(
                span_id=span_id,
                text=text,
                confidence=confidence,
                polygon_px=polygon if valid_geom else None,
                image_width=max(0, int(image_width)),
                image_height=max(0, int(image_height)),
                provider_order=(int(span.provider_order) if span.provider_order is not None else idx),
                char_start=char_start,
                char_end=char_end,
            )
        )
        total_chars = char_end
        if confidence is not None:
            kept_confidences.append(confidence)
        if valid_geom:
            geometry_area += area
    image_area = float(max(1, int(image_width) * int(image_height)))
    coverage = min(1.0, max(0.0, geometry_area / image_area)) if out else 0.0
    if out:
        status = "usable"
    elif rows:
        status = "filtered_empty"
    else:
        status = "empty"
    summary = OCRQualitySummary(
        total_spans=len(rows),
        kept_spans=len(out),
        dropped_empty=dropped_empty,
        dropped_low_confidence=dropped_low,
        dropped_invalid_confidence=dropped_bad_conf,
        dropped_invalid_geometry=dropped_geom,
        dropped_over_limit=dropped_limit,
        clipped_span_texts=clipped,
        mean_confidence=(sum(kept_confidences) / len(kept_confidences) if kept_confidences else None),
        min_confidence=(min(kept_confidences) if kept_confidences else None),
        max_confidence=(max(kept_confidences) if kept_confidences else None),
        geometry_coverage_ratio=coverage,
        status=status,
    )
    return out, summary
