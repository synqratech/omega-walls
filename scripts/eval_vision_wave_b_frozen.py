#!/usr/bin/env python3
"""Deterministic OCR/spatial comparative gate for Vision Wave B.

This gate is network-free and uses recorded OCR/vision/adjudication outputs over a
fixed image corpus. It validates modality fusion, OCR quality filtering, geometry-
based crops, uncertainty-triggered region passes and benign adjudication. It is not
a live quality claim for RapidOCR, PaddleOCR or a cloud vision model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.api.ocr_adjudication import build_ocr_adjudication_items, interpret_ocr_adjudication_projection  # noqa: E402
from omega.interfaces.contracts_v1 import ProjectionEvidence, ProjectionResult, WALLS_V1  # noqa: E402
from omega.projector.api_hybrid.blob_store import ImageBlobStore  # noqa: E402
from omega.vision.contracts import OCRQualityPolicy, OCRSpan, normalize_ocr_spans  # noqa: E402
from omega.vision.spatial_policy import RegionPassPolicy, decide_region_pass  # noqa: E402

SUITE_VERSION = "vision_wave_b_frozen_v1"
KEY_IMPLEMENTATION_FILES = (
    "omega/vision/contracts.py",
    "omega/vision/spatial_policy.py",
    "omega/rag/attachment_ingestion.py",
    "omega/api/ocr_span_attribution.py",
    "omega/api/ocr_adjudication.py",
    "omega/api/scan_request_orchestration.py",
    "omega/projector/api_hybrid/semantic_contracts.py",
    "omega/projector/api_hybrid/blob_store.py",
    "omega/projector/api_hybrid_projector.py",
    "omega/config/loader.py",
    "omega/config/validators/projector.py",
    "config/profiles/pilot.yml",
    "config/profiles/prod_vision.yml",
)
MODES = ("vision_only", "ocr_only", "vision_plus_ocr", "region_plus_ocr")
THRESHOLD = 0.12

FORBIDDEN_RAW_MEDIA_KEYS = {
    "bytes_b64",
    "raw_bytes",
    "file_bytes",
    "image_bytes",
    "image_bytes_b64",
}


def _count_raw_media_boundary_findings(value: Any) -> int:
    """Count raw-media fields that escaped into a serializable core contract."""
    if isinstance(value, Mapping):
        findings = sum(1 for key in value if str(key) in FORBIDDEN_RAW_MEDIA_KEYS)
        return findings + sum(_count_raw_media_boundary_findings(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_raw_media_boundary_findings(item) for item in value)
    return 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate id at line {line_no}")
        seen.add(sample_id)
        rows.append(row)
    if not rows:
        raise ValueError("empty Wave B manifest")
    return rows


def _projection(payload: Mapping[str, Any], *, doc_id: str) -> ProjectionResult:
    pressure = dict(payload.get("pressure_signed", {}) or {})
    directive = dict(payload.get("directive_intent", {}) or {})
    raw = [float(pressure.get(wall, 0.0)) for wall in WALLS_V1]
    return ProjectionResult(
        doc_id=doc_id,
        v=np.asarray([max(0.0, value) for value in raw], dtype=float),
        evidence=ProjectionEvidence(
            polarity=[1 if value > 0 else (-1 if value < 0 else 0) for value in raw],
            debug_scores_raw=raw,
            matches={
                "api_perception": {
                    "confidence": float(payload.get("confidence", 0.0)),
                    "defensive_context": bool(payload.get("defensive_context", False)),
                    "directive_intent": {str(k): bool(v) for k, v in directive.items()},
                    "vision_semantic_status": "vision_semantic_active",
                }
            },
        ),
    )


def _positive(payload: Mapping[str, Any]) -> bool:
    pressure = dict(payload.get("pressure_signed", {}) or {})
    directive = dict(payload.get("directive_intent", {}) or {})
    return any(float(pressure.get(w, 0.0)) >= THRESHOLD and bool(directive.get(w, False)) for w in WALLS_V1)


def _metrics(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    attacks = [row for row in selected if row["label"] == "malicious"]
    benign = [row for row in selected if row["label"] == "benign"]
    tp = sum(1 for row in attacks if row["decision"] == "malicious")
    fn = len(attacks) - tp
    fp = sum(1 for row in benign if row["decision"] == "malicious")
    tn = len(benign) - fp
    return {
        "samples": len(selected),
        "attack_samples": len(attacks),
        "benign_samples": len(benign),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "attack_recall": tp / len(attacks) if attacks else 0.0,
        "benign_false_positive_rate": fp / len(benign) if benign else 1.0,
        "uncertain": sum(1 for row in selected if row["decision"] == "uncertain"),
    }


def evaluate(*, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _load(manifest_path)
    manifest_root = manifest_path.parent.resolve()
    result_rows: list[dict[str, Any]] = []
    blob_store = ImageBlobStore(max_total_bytes=64 * 1024 * 1024, max_records=256)
    crop_failures = 0
    quality_failures = 0
    low_confidence_drops = 0
    region_required = 0
    region_rescued = 0
    raw_media_boundary_findings = 0
    corpus_rows: list[dict[str, Any]] = []
    policy = OCRQualityPolicy(
        min_confidence=0.55,
        max_spans=256,
        max_span_chars=512,
        require_geometry=True,
        min_polygon_area_px=4.0,
    )
    region_policy = RegionPassPolicy(
        enabled=True,
        trigger_mode="uncertain",
        pressure_abs_max=0.12,
        confidence_max=0.80,
        max_tiles=5,
        overlap_ratio=0.08,
        include_center_crop=True,
    )
    for row in rows:
        image_path = (manifest_root / str(row["file"])).resolve()
        if manifest_root not in image_path.parents:
            raise ValueError("manifest path escape")
        if _sha(image_path) != str(row["sha256"]):
            raise ValueError(f"image sha mismatch: {row['id']}")
        width, height = int(row["image_width"]), int(row["image_height"])
        raw_spans = [
            OCRSpan(
                span_id=str(span.get("span_id", "")),
                text=str(span.get("text", "")),
                confidence=(float(span["confidence"]) if span.get("confidence") is not None else None),
                polygon_px=tuple((float(p[0]), float(p[1])) for p in span.get("polygon_px", [])),
                image_width=width,
                image_height=height,
                provider_order=int(span.get("provider_order", 0)),
            )
            for span in list(row.get("ocr_spans", []) or [])
        ]
        spans, quality = normalize_ocr_spans(
            raw_spans,
            max_chars=200_000,
            image_width=width,
            image_height=height,
            policy=policy,
        )
        low_confidence_drops += int(quality.dropped_low_confidence)
        if quality.status != "usable" or quality.kept_spans < 1:
            quality_failures += 1
        lookup = {span.span_id: span for span in spans}
        primary_span_ids = [span.span_id for span in spans if span.text.lower() != "noise"]
        source_wall = str(row.get("target_wall") or "")
        crop_items, crop_trace = build_ocr_adjudication_items(
            request_id=str(row["id"]),
            source_id=f"wave-b:{row['id']}",
            source_type="image",
            trust="untrusted",
            file_bytes=image_path.read_bytes(),
            triggered_span_ids=primary_span_ids,
            matched_span_ids=primary_span_ids if source_wall else [],
            supporting_span_ids=primary_span_ids,
            source_walls=[source_wall] if source_wall else [],
            span_lookup=lookup,
            source_image_meta={"width": width, "height": height},
            register_image_payload=lambda **payload: {
                "mime": str(payload["mime"]),
                "sha256": str(payload["sha256"]),
                "bytes_ref": blob_store.put(
                    scope_id=str(row["id"]),
                    data=bytes(payload["raw"]),
                    mime=str(payload["mime"]),
                    expected_sha256=str(payload["sha256"]),
                ),
                "size_bytes": len(bytes(payload["raw"])),
                "role": str(payload["role"]),
                "width": int(payload["width"]),
                "height": int(payload["height"]),
            },
            crop_strategy="contextual",
            max_tiles=2,
            max_crop_area_ratio=0.35,
            min_crop_width_px=160,
            min_crop_height_px=72,
        )
        if not crop_items or int(crop_trace.get("tile_count", 0)) < 1:
            crop_failures += 1
        for crop_item in crop_items:
            raw_media_boundary_findings += _count_raw_media_boundary_findings(crop_item.meta)
        full = dict(row["recorded_vision_full"])
        region = dict(row["recorded_vision_region"])
        ocr = dict(row["recorded_ocr_projection"])
        adjud = dict(row["recorded_adjudication"])
        full_positive = _positive(full)
        ocr_positive = _positive(ocr)
        decision = decide_region_pass(
            policy=region_policy,
            pressure_signed=dict(full.get("pressure_signed", {}) or {}),
            confidence=float(full.get("confidence", 0.0)),
            has_image=True,
            is_region_pass=False,
        )
        region_positive = _positive(region) if decision.run else False
        expected_region = bool(row.get("expected_region_required", False))
        if expected_region:
            region_required += 1
            if region_positive:
                region_rescued += 1
        adjudication = interpret_ocr_adjudication_projection(
            projection=_projection(adjud, doc_id=f"{row['id']}:adjudication"),
            source_walls=[source_wall] if source_wall else [],
            threshold=THRESHOLD,
        )
        decisions = {
            "vision_only": "malicious" if full_positive else "benign",
            "ocr_only": "malicious" if ocr_positive else "benign",
        }
        if full_positive:
            fused = "malicious"
        elif ocr_positive:
            fused = "malicious" if adjudication["result"] == "live_attack" else (
                "benign" if adjudication["result"] in {"quoted_or_defensive", "benign_ui"} else "uncertain"
            )
        else:
            fused = "benign"
        decisions["vision_plus_ocr"] = fused
        if full_positive or region_positive:
            regional = "malicious"
        else:
            regional = fused
        decisions["region_plus_ocr"] = regional
        for mode in MODES:
            result_rows.append(
                {
                    "id": str(row["id"]),
                    "label": str(row["label"]),
                    "mode": mode,
                    "decision": decisions[mode],
                    "vision_full_positive": full_positive,
                    "ocr_positive": ocr_positive,
                    "region_attempted": bool(decision.run),
                    "region_reason": str(decision.reason),
                    "region_positive": region_positive,
                    "adjudication_result": str(adjudication["result"]),
                    "ocr_quality": quality.to_dict(),
                    "crop_tile_count": int(crop_trace.get("tile_count", 0)),
                }
            )
        corpus_rows.append(
            {
                "id": row["id"],
                "sha256": row["sha256"],
                "label": row["label"],
                "target_wall": row.get("target_wall"),
                "recorded_signals_sha256": _canonical_sha(
                    {
                        "vision_full": full,
                        "vision_region": region,
                        "ocr": ocr,
                        "adjudication": adjud,
                        "spans": row.get("ocr_spans", []),
                    }
                ),
            }
        )
    metrics = {mode: _metrics(result_rows, mode=mode) for mode in MODES}
    summary = {
        "samples_total": len(rows),
        "attack_samples": sum(1 for row in rows if row["label"] == "malicious"),
        "benign_samples": sum(1 for row in rows if row["label"] == "benign"),
        "modes": metrics,
        "ocr_quality_failures": quality_failures,
        "spatial_crop_failures": crop_failures,
        "low_confidence_spans_dropped": low_confidence_drops,
        "region_required_samples": region_required,
        "region_rescue_rate": region_rescued / region_required if region_required else 0.0,
        "raw_media_boundary_findings": int(raw_media_boundary_findings),
    }
    gates = {
        "vision_plus_ocr_recall_ge_0_80": metrics["vision_plus_ocr"]["attack_recall"] >= 0.80,
        "region_plus_ocr_recall_eq_1": metrics["region_plus_ocr"]["attack_recall"] == 1.0,
        "region_plus_ocr_fp_eq_0": metrics["region_plus_ocr"]["benign_false_positive_rate"] == 0.0,
        "fused_not_worse_than_vision": metrics["region_plus_ocr"]["attack_recall"] >= metrics["vision_only"]["attack_recall"],
        "fused_fp_not_worse_than_ocr": metrics["region_plus_ocr"]["benign_false_positive_rate"] <= metrics["ocr_only"]["benign_false_positive_rate"],
        "region_rescue_rate_eq_1": summary["region_rescue_rate"] == 1.0,
        "ocr_quality_failures_eq_0": quality_failures == 0,
        "spatial_crop_failures_eq_0": crop_failures == 0,
        "low_confidence_filter_exercised": low_confidence_drops >= len(rows),
        "raw_media_boundary_findings_eq_0": raw_media_boundary_findings == 0,
    }
    report = {
        "schema_version": "1.0",
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "recorded_ocr_spatial_fusion_regression",
        "quality_claim": "none_live_ocr_or_vision_not_invoked",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha(manifest_path),
        "dataset_sha256": _canonical_sha(corpus_rows),
        "implementation_hashes": {path: _sha(ROOT / path) for path in KEY_IMPLEMENTATION_FILES},
        "summary": summary,
        "gates": gates,
        "samples": result_rows,
        "notes": [
            "Recorded outputs exercise OCR quality filtering, spatial crop construction, uncertainty routing and adjudication.",
            "OCR adjudication crops are registered directly in the request-scoped BlobStore; no raw-media fields may enter ContentItem metadata.",
            "A live representative benchmark remains required for provider-level OCR and cloud vision quality claims.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/data/vision_wave_b_frozen/manifest.jsonl")
    parser.add_argument("--output", default="artifacts/vision_wave_b/frozen/vision_wave_b_frozen_v1.json")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not output.is_absolute():
        output = ROOT / output
    report = evaluate(manifest_path=manifest, output_path=output)
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
