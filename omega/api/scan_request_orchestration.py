"""Scan-request orchestration extracted from api.server."""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from fastapi import HTTPException

from omega.api.chunk_pipeline import aggregate_semantic_execution_trace
from omega.api.incident_export import build_incident_record_from_scan
from omega.api.ocr_adjudication import (
    build_ocr_adjudication_items,
    interpret_ocr_adjudication_projection,
    polygon_to_rect_px,
)
from omega.api.ocr_span_attribution import matched_ocr_span_ids_for_item
from omega.api.runtime_factory import ScanRuntime
from omega.effects.runtime import evaluate_typed_effect_shadow
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, ProjectionResult
from omega.monitoring.enrichment import (
    build_downstream_summary,
    build_redacted_fragments,
)
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode
from omega.monitoring.models import MonitorEvent
from omega.notifications.models import utc_now_iso
from omega.rag.attachment_ingestion import AttachmentExtractResult
from omega.rag.source_policy import SourceTrustPolicy
from omega.runtime.integrity_policy import (
    assess_runtime_artifact,
    build_runtime_artifact,
    summarize_artifact_assessments,
)
from omega.runtime.skillbox import evaluate_skillbox_shadow
from omega.runtime.scan_pipeline import (
    apply_semantic_failure_policy_to_actions,
    compose_control_outcome_state,
    compose_effective_actions,
    dedupe_pressure_items_step_local,
    evaluate_projection_phase,
    normalize_action_types,
    run_core_step_phase,
)
from omega.telemetry.anonymous import build_telemetry_event
from omega.telemetry.ids import build_decision_id, build_trace_id_api
from omega.telemetry.incident_artifact import (
    build_incident_artifact,
    should_capture_incident_text,
    should_emit_incident_artifact,
)
from omega.telemetry.events import build_off_event


@dataclass(frozen=True)
class ScanRequestDeps:
    effective_runtime_mode: Callable[[ScanRuntime, Mapping[str, Any]], str]
    guard_mode: Callable[[ScanRuntime], GuardMode]
    infer_format: Callable[[Optional[str], Optional[str]], str]
    source_type_for_format: Callable[[str], str]
    resolve_control_outcome: Callable[[Sequence[str], str], str]
    source_risk_band: Callable[[Sequence[ContentItem]], str]
    normalize_trust_band: Callable[[str], str]
    monitor_attribution_rows: Callable[
        [Sequence[ContentItem], Sequence[Mapping[str, Any]]], List[Dict[str, Any]]
    ]
    build_api_risk_event: Callable[[Mapping[str, Any], Mapping[str, Any], bool], Any]
    build_document_scan_report: Callable[[Any, str, Sequence[str], int], Dict[str, Any]]
    notifications_cfg: Callable[[ScanRuntime], Dict[str, Any]]
    incident_export_enabled: Callable[[ScanRuntime], bool]
    incident_replay_enabled: Callable[[ScanRuntime], bool]
    sha256_hex: Callable[[str], str]
    clamp: Callable[[float, float, float], float]
    attestation_block: Callable[..., Any]
    score_chunks: Callable[..., Any]
    extract_text_payload: Callable[..., Any]
    extract_attachment: Callable[..., Any]


LOGGER = logging.getLogger(__name__)
_TRACE_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]+|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9\-]+|eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+)",
    flags=re.IGNORECASE,
)


def _profile_env(runtime: ScanRuntime) -> str:
    return str(((runtime.config.get("profiles", {}) or {}).get("env", ""))).strip().lower()


def _is_enterprise_production(runtime: ScanRuntime) -> bool:
    return _profile_env(runtime) in {"prod_enterprise", "prod_vision_enterprise"}


def _config_refs(runtime: ScanRuntime, sha256_hex: Callable[[str], str]) -> Dict[str, str]:
    return {
        "api_config_sha256": sha256_hex(
            json.dumps(runtime.config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        ),
        "policy_version": str((runtime.config.get("off_policy", {}) or {}).get("policy_version", "")),
    }


def _omega_thresholds(runtime: ScanRuntime) -> Dict[str, Any]:
    params = runtime.omega_core.params
    ocfg = runtime.config.get("omega", {}) or {}
    off = ocfg.get("off", {}) or {}
    return {
        "epsilon": float(getattr(params, "epsilon", ocfg.get("epsilon", 0.0))),
        "alpha": float(getattr(params, "alpha", ocfg.get("alpha", 1.0))),
        "beta": float(getattr(params, "beta", ocfg.get("beta", 0.0))),
        "lambda": float(getattr(params, "lam", ocfg.get("lambda", 0.0))),
        "off": {
            "tau": float(getattr(params, "off_tau", off.get("tau", 0.0))),
            "Theta": float(getattr(params, "off_Theta", off.get("Theta", 0.0))),
            "Sigma": float(getattr(params, "off_Sigma", off.get("Sigma", 0.0))),
            "theta": float(getattr(params, "off_theta", off.get("theta", 0.0))),
            "N": int(getattr(params, "off_N", off.get("N", 0))),
        },
        "attrib_gamma": float(getattr(params, "attrib_gamma", ((ocfg.get("attribution", {}) or {}).get("gamma", 0.0)))),
    }
_TRACE_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE
)


def _register_image_payload(
    *,
    projector: Any,
    scope_id: str,
    raw: bytes,
    mime: str,
    sha256: str,
    role: str = "untrusted_visual_content",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Dict[str, Any]:
    register = getattr(projector, "register_image_blob", None)
    if callable(register):
        bytes_ref = register(
            scope_id=str(scope_id),
            data=bytes(raw),
            mime=str(mime),
            expected_sha256=str(sha256),
        )
    else:
        # Safe opaque placeholder for non-semantic/fake projectors. Raw bytes are never
        # embedded in ContentItem.meta.
        bytes_ref = f"blob://{str(scope_id)[:160]}/{str(sha256)[:32]}"
    return {
        "mime": str(mime).strip().lower(),
        "sha256": str(sha256).strip().lower(),
        "bytes_ref": str(bytes_ref),
        "size_bytes": int(len(raw)),
        "role": str(role),
        "width": (int(width) if width is not None else None),
        "height": (int(height) if height is not None else None),
    }


def _materialize_item_image_refs(
    *, projector: Any, item: ContentItem, scope_id: str
) -> ContentItem:
    meta = dict(item.meta or {})
    image_meta = meta.get("semantic_image")
    if not isinstance(image_meta, Mapping):
        return item
    rows_in = image_meta.get("variants", [])
    rows = list(rows_in) if isinstance(rows_in, list) and rows_in else [image_meta]
    materialized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("bytes_ref", "")).strip():
            clean = {
                k: v
                for k, v in dict(row).items()
                if k not in {"bytes_b64", "raw_bytes", "file_bytes"}
            }
            materialized.append(clean)
            continue
        raw_b64 = str(row.get("bytes_b64", "")).strip()
        if not raw_b64:
            continue
        try:
            raw = base64.b64decode(raw_b64, validate=True)
        except Exception as exc:
            raise ValueError("invalid semantic image base64") from exc
        digest = str(row.get("sha256", "")).strip().lower() or _sha256_bytes(raw)
        materialized.append(
            _register_image_payload(
                projector=projector,
                scope_id=scope_id,
                raw=raw,
                mime=str(row.get("mime", "image/png")),
                sha256=digest,
                role=str(row.get("role", "ocr_target_crop")),
                width=(int(row["width"]) if row.get("width") is not None else None),
                height=(int(row["height"]) if row.get("height") is not None else None),
            )
        )
    if not materialized:
        meta.pop("semantic_image", None)
    elif len(materialized) == 1:
        meta["semantic_image"] = materialized[0]
    else:
        meta["semantic_image"] = {"variants": materialized}
    return replace(item, meta=meta)


def _attachment_chunk_kind_for(item: ContentItem) -> str:
    meta = item.meta if isinstance(item.meta, Mapping) else {}
    return str(meta.get("attachment_chunk_kind", "visible") or "visible")


def _build_api_artifact_integrity(
    *,
    config: Mapping[str, Any],
    items: Sequence[ContentItem],
    source_id: str,
    source_type: str,
    source_trust: str,
    request_boundary_step: int,
    request_id: str,
    effect_shadow: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = (config.get("runtime_integrity", {}) if isinstance(config, Mapping) else {}) or {}
    emit_artifact_trace = bool(cfg.get("emit_artifact_trace", True))
    artifacts = []
    assessments = []
    for item in list(items or []):
        artifact = build_runtime_artifact(item, trusted_control_excluded=False)
        assessment = assess_runtime_artifact(artifact, effect_shadow=effect_shadow)
        artifacts.append(artifact)
        assessments.append(assessment)
    request_item = ContentItem(
        doc_id=f"{request_id}:request",
        source_id=str(source_id),
        source_type=str(source_type),
        trust=str(source_trust),
        text="[api_request]",
        artifact_id=f"api-request-{request_id}",
        origin="api_request",
        content_hash=f"api-request:{request_id}",
        boundary_step=int(request_boundary_step),
        meta={"request_level_artifact": True, "request_id": str(request_id)},
    )
    request_artifact = build_runtime_artifact(
        request_item,
        trusted_control_excluded=False,
        operation_metadata={"request_level": True},
    )
    request_assessment = assess_runtime_artifact(
        request_artifact,
        effect_shadow=effect_shadow,
    )
    artifacts.append(request_artifact)
    assessments.append(request_assessment)
    summary = summarize_artifact_assessments(
        artifacts=artifacts,
        assessments=assessments,
    )
    summary["packet_effect_signal"] = {
        "effect_forecast_status": str(
            effect_shadow.get("effect_forecast_status", "disabled")
        ),
        "effect_policy_gate_status": str(
            effect_shadow.get("effect_policy_gate_status", "disabled")
        ),
        "has_effect_candidate": bool(
            isinstance(effect_shadow.get("effect_wall_candidate"), dict)
        ),
    }
    summary["named_skill_invocation"] = (
        dict(effect_shadow.get("named_skill_invocation", {}))
        if isinstance(effect_shadow.get("named_skill_invocation"), dict)
        else None
    )
    summary["skill_provenance_assessment"] = (
        dict(effect_shadow.get("skill_provenance_assessment", {}))
        if isinstance(effect_shadow.get("skill_provenance_assessment"), dict)
        else None
    )
    summary["skillbox_status"] = str(effect_shadow.get("skillbox_status", "disabled"))
    summary["skillbox_verification"] = (
        dict(effect_shadow.get("skillbox_verification", {}))
        if isinstance(effect_shadow.get("skillbox_verification"), dict)
        else None
    )
    summary["skillbox_ledger_hit"] = bool(effect_shadow.get("skillbox_ledger_hit", False))
    summary["skillbox_content_sha256"] = effect_shadow.get("skillbox_content_sha256")
    summary["skillbox_capabilities"] = [
        str(x) for x in list(effect_shadow.get("skillbox_capabilities", []) or [])
    ]
    summary["skillbox_gate_decision"] = str(
        effect_shadow.get("skillbox_gate_decision", "disabled")
    )
    return {
        "artifacts": [artifact.to_dict() for artifact in artifacts] if emit_artifact_trace else [],
        "artifact_assessments": [assessment.to_dict() for assessment in assessments] if emit_artifact_trace else [],
        "artifact_assessment_summary": summary,
    }


def _trace_sha256(text: str) -> str:
    return _sha256_bytes(str(text or "").encode("utf-8", errors="ignore"))


def _sha256_bytes(raw: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(bytes(raw))
    return h.hexdigest()


def _redacted_excerpt(text: str, *, limit: int = 80) -> str:
    cleaned = str(text or "").strip()
    if cleaned:
        return "[redacted_ocr_text]"
    return ""


def _ocr_layout_summary(
    spans: Sequence[Any], *, preview_limit: int = 5
) -> Dict[str, Any]:
    rows = list(spans or [])
    preview: List[Dict[str, Any]] = []
    spans_with_polygon_px = 0
    spans_with_confidence = 0
    provider_order_present = True if rows else False
    for idx, span in enumerate(rows):
        text = str(getattr(span, "text", "") or "").strip()
        polygon_px = getattr(span, "polygon_px", None)
        confidence = getattr(span, "confidence", None)
        provider_order = getattr(span, "provider_order", None)
        if polygon_px:
            spans_with_polygon_px += 1
        if confidence is not None:
            spans_with_confidence += 1
        if provider_order is None:
            provider_order_present = False
        if idx >= int(preview_limit):
            continue
        preview.append(
            {
                "span_id": str(getattr(span, "span_id", "") or ""),
                "text_sha256": _trace_sha256(text),
                "redacted_excerpt": _redacted_excerpt(text),
                "confidence": (float(confidence) if confidence is not None else None),
                "polygon_px": polygon_px,
                "rect_px": polygon_to_rect_px(polygon_px),
                "image_width": int(getattr(span, "image_width", 0) or 0),
                "image_height": int(getattr(span, "image_height", 0) or 0),
                "provider_order": (
                    int(provider_order) if provider_order is not None else None
                ),
                "char_start": (
                    int(getattr(span, "char_start", 0))
                    if getattr(span, "char_start", None) is not None
                    else None
                ),
                "char_end": (
                    int(getattr(span, "char_end", 0))
                    if getattr(span, "char_end", None) is not None
                    else None
                ),
            }
        )
    return {
        "span_count": int(len(rows)),
        "spans_with_polygon_px": int(spans_with_polygon_px),
        "spans_with_confidence": int(spans_with_confidence),
        "provider_order_present": bool(provider_order_present),
        "preview": preview,
    }


def _zero_projection_signal(proj: ProjectionResult) -> ProjectionResult:
    return ProjectionResult(
        doc_id=str(proj.doc_id),
        v=np.zeros_like(np.asarray(proj.v, dtype=float)),
        evidence=proj.evidence,
    )


def _max_chunk_score_band(
    *, chunk_scores: Sequence[Any], chunk_kind_by_doc_id: Mapping[str, str], kind: str
) -> str:
    row_scores = [
        float(row.score_max)
        for row in list(chunk_scores or [])
        if chunk_kind_by_doc_id.get(str(row.doc_id), "") == kind
    ]
    if not row_scores:
        return "allow"
    worst_local = max(row_scores)
    if worst_local >= 0.72:
        return "block"
    if worst_local >= 0.45:
        return "quarantine"
    return "allow"


def _matched_span_ids_for_projection(
    *,
    item: ContentItem | None,
    projection: ProjectionResult | None,
    ocr_span_lookup: Mapping[str, Any],
    active_walls: Sequence[str],
) -> List[str]:
    if item is None or projection is None:
        return []
    item_meta = item.meta if isinstance(item.meta, Mapping) else {}
    matches = (
        projection.evidence.matches
        if isinstance(getattr(projection.evidence, "matches", {}), Mapping)
        else {}
    )
    return matched_ocr_span_ids_for_item(
        item_text=str(item.text or ""),
        item_meta=item_meta,
        ocr_span_lookup=ocr_span_lookup,
        matches=matches,
        active_walls=[str(x) for x in list(active_walls or []) if str(x).strip()],
    )


def run_scan_request(
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    *,
    include_document_scan_report: bool = False,
    deps: ScanRequestDeps,
) -> Dict[str, Any]:
    _effective_runtime_mode = deps.effective_runtime_mode
    _guard_mode = deps.guard_mode
    _infer_format = deps.infer_format
    _source_type_for_format = deps.source_type_for_format
    _resolve_control_outcome = deps.resolve_control_outcome
    _source_risk_band = deps.source_risk_band
    _normalize_trust_band = deps.normalize_trust_band
    _monitor_attribution_rows = deps.monitor_attribution_rows
    _build_api_risk_event = deps.build_api_risk_event
    _build_document_scan_report = deps.build_document_scan_report
    _notifications_cfg = deps.notifications_cfg
    _incident_export_enabled = deps.incident_export_enabled
    _incident_replay_enabled = deps.incident_replay_enabled
    _sha256_hex = deps.sha256_hex
    _clamp = deps.clamp
    _attestation_block = deps.attestation_block
    _score_chunks = deps.score_chunks
    _extract_text_payload = deps.extract_text_payload
    _extract_attachment = deps.extract_attachment
    cfg = runtime.config
    api_cfg = cfg.get("api", {}) or {}
    attachment_cfg = ((cfg.get("retriever", {}) or {}).get("sqlite_fts", {}) or {}).get(
        "attachments", {}
    ) or {}

    filename = str(parsed.get("filename") or "").strip() or None
    mime = str(parsed.get("mime") or "").strip() or None
    tenant_id = str(parsed["tenant_id"])
    request_id = str(parsed["request_id"])
    runtime_mode = _effective_runtime_mode(runtime, parsed)
    guard_mode = _guard_mode(runtime)
    monitor_enabled = bool(guard_mode == GuardMode.MONITOR)
    session_id = (
        str(parsed.get("session_id") or "").strip()
        if runtime_mode == "stateful"
        else None
    )
    actor_id = (
        str(parsed.get("actor_id") or "").strip()
        if runtime_mode == "stateful"
        else None
    )
    state_step_prev = 0
    cross_carryover_applied = False
    cross_active_action_types: List[str] = []
    cross_actor_hash: Optional[str] = None
    cross_active_actions: List[Any] = []
    state_vec = np.zeros(4, dtype=float)
    session_store = runtime.session_store
    if runtime_mode == "stateful":
        if session_store is None or not session_id:
            raise HTTPException(
                status_code=503, detail="stateful_runtime_not_configured"
            )
        cached = session_store.get_cached_response(
            tenant_id=tenant_id, session_id=session_id, request_id=request_id
        )
        if cached is not None:
            return cached
        state_row = session_store.load_session_state(
            tenant_id=tenant_id, session_id=session_id
        )
        if state_row is not None:
            state_step_prev = int(state_row.step)
            state_vec = np.asarray(state_row.m, dtype=float)
            if not actor_id:
                actor_id = str(state_row.actor_id)
        if not actor_id:
            actor_id = str(session_id)

    source_id = f"api:{tenant_id}:{request_id}"
    if runtime_mode == "stateful" and session_id:
        source_id = f"api:{tenant_id}:{session_id}"

    ingestion_flags: List[str] = []

    if bool(parsed.get("use_extracted_text", False)):
        extracted = _extract_text_payload(
            text=str(parsed.get("extracted_text") or ""), cfg=attachment_cfg
        )
        reported_format = _infer_format(filename=filename, mime=mime)
        if reported_format == "zip":
            ingestion_flags.append("zip_deferred_runtime")
        fmt = "text"
    else:
        try:
            extracted = _extract_attachment(
                content_bytes=parsed.get("file_bytes"),
                filename=filename,
                mime=mime,
                cfg=attachment_cfg,
            )
        except Exception:
            extracted = AttachmentExtractResult(
                text="",
                chunks=[],
                format=_infer_format(filename=filename, mime=mime),
                text_empty=True,
                scan_like=False,
                hidden_text_chars=0,
                warnings=["ingestion_error", "text_empty"],
                recommended_verdict="quarantine",
            )
        fmt = extracted.format

    ingestion_flags.extend(list(extracted.warnings))
    if fmt == "zip" and "zip_deferred_runtime" not in ingestion_flags:
        ingestion_flags.append("zip_deferred_runtime")

    source_type = _source_type_for_format(fmt)
    source_trust_policy = SourceTrustPolicy.from_config(cfg)
    source_trust = source_trust_policy.trust_for(
        source_type=source_type, source_id=source_id
    )
    request_boundary_step = (
        int(state_step_prev) + 1 if runtime_mode == "stateful" else 1
    )
    items: List[ContentItem] = []
    image_meta_payload = None
    visual_manifest: List[Dict[str, Any]] = []
    visual_variants: List[Dict[str, Any]] = []
    for asset in list(getattr(extracted, "visual_assets", []) or []):
        try:
            raw_asset = asset.decode()
            row = _register_image_payload(
                projector=runtime.projector,
                scope_id=str(request_id),
                raw=raw_asset,
                mime=str(getattr(asset, "mime", "image/png")),
                sha256=str(getattr(asset, "sha256", "")),
                role=str(getattr(asset, "role", "untrusted_visual_content")),
                width=int(getattr(asset, "width", 0) or 0) or None,
                height=int(getattr(asset, "height", 0) or 0) or None,
            )
            row.update(
                {
                    "asset_id": str(getattr(asset, "asset_id", "")),
                    "source_kind": str(getattr(asset, "source_kind", "embedded")),
                    "page_number": getattr(asset, "page_number", None),
                    "embedded_index": getattr(asset, "embedded_index", None),
                }
            )
            visual_variants.append(row)
            visual_manifest.append({k: v for k, v in row.items() if k != "bytes_ref"})
        except Exception:
            ingestion_flags.append("visual_asset_registration_error")
    if not visual_variants and bool(getattr(extracted, "is_image", False)):
        raw_image = bytes(parsed.get("file_bytes") or b"")
        image_mime = (
            str(
                getattr(extracted, "image_mime", "")
                or str(mime or "application/octet-stream")
            )
            .strip()
            .lower()
        )
        image_sha256 = str(
            getattr(extracted, "image_sha256", "") or _sha256_bytes(raw_image)
        )
        visual_variants.append(
            _register_image_payload(
                projector=runtime.projector,
                scope_id=str(request_id),
                raw=raw_image,
                mime=image_mime,
                sha256=image_sha256,
                role="untrusted_visual_content",
            )
        )
    if len(visual_variants) == 1:
        image_meta_payload = dict(visual_variants[0])
    elif visual_variants:
        image_meta_payload = {"variants": [dict(x) for x in visual_variants]}
    text_chunks = [chunk for chunk in extracted.chunks if str(chunk.text).strip()]
    ocr_layout_summary = _ocr_layout_summary(getattr(extracted, "ocr_spans", []) or [])
    ocr_span_lookup = {
        str(getattr(span, "span_id", "") or ""): span
        for span in list(getattr(extracted, "ocr_spans", []) or [])
        if str(getattr(span, "span_id", "") or "").strip()
    }
    if not text_chunks and not bool(visual_variants):
        placeholder = (
            "[attachment_scan_like]"
            if extracted.scan_like
            else (
                "[attachment_text_empty]"
                if extracted.text_empty
                else "[attachment_ingestion_empty]"
            )
        )
        chunks_for_items = [
            {
                "text": placeholder,
                "kind": "visible",
                "is_hidden": False,
                "include_semantic_image": False,
            }
        ]
    else:
        chunks_for_items = [
            {
                "text": str(chunk.text),
                "kind": str(getattr(chunk, "kind", "visible") or "visible"),
                "is_hidden": bool(getattr(chunk, "is_hidden", False)),
                "include_semantic_image": False,
                "ocr_span_ids": [
                    str(x)
                    for x in list(getattr(chunk, "ocr_span_ids", []) or [])
                    if str(x).strip()
                ],
                "char_start": getattr(chunk, "char_start", None),
                "char_end": getattr(chunk, "char_end", None),
            }
            for chunk in text_chunks
        ]
    if bool(visual_variants):
        chunks_for_items.append(
            {
                "text": "[attachment_visual_only]"
                if not text_chunks
                else "[attachment_visual_semantic]",
                "kind": "image_semantic",
                "is_hidden": False,
                "include_semantic_image": True,
            }
        )

    for idx, chunk_row in enumerate(chunks_for_items):
        text_chunk = str(chunk_row["text"])
        content_hash = _sha256_hex(text_chunk)
        artifact_id = f"api-art-{_sha256_hex(f'{source_id}|{content_hash}|{int(request_boundary_step)}')[:16]}"
        item_meta = {
            "tenant_id": tenant_id,
            "request_id": request_id,
            "attachment_format": fmt,
            "attachment_chunk_kind": str(chunk_row.get("kind", "visible")),
            "attachment_modality": (
                "ocr"
                if str(chunk_row.get("kind", "visible")) == "ocr"
                else "native_text"
            ),
            "attachment_chunk_hidden": bool(chunk_row.get("is_hidden", False)),
            "ingestion_flags": sorted(set(ingestion_flags)),
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "boundary_step": int(request_boundary_step),
            "ocr_status": str(getattr(extracted, "ocr_status", "none") or "none"),
            "ocr_provider": getattr(extracted, "ocr_provider", None),
            "ocr_text_chars": int(getattr(extracted, "ocr_text_chars", 0) or 0),
            "ocr_quality": dict(
                getattr(
                    getattr(extracted, "ocr_quality", None), "to_dict", lambda: {}
                )()
            ),
            "visual_status": str(getattr(extracted, "visual_status", "none") or "none"),
            "visual_asset_count": int(len(visual_variants)),
            "visual_asset_manifest": list(visual_manifest),
            "data_region": str(parsed.get("data_region") or "unspecified"),
        }
        if str(chunk_row.get("kind", "visible")) == "ocr":
            item_meta["derived_from"] = "image"
            item_meta["ocr_derived"] = True
            item_meta["ocr_span_ids"] = [
                str(x)
                for x in list(chunk_row.get("ocr_span_ids", []) or [])
                if str(x).strip()
            ]
            item_meta["ocr_char_start"] = chunk_row.get("char_start")
            item_meta["ocr_char_end"] = chunk_row.get("char_end")
            item_meta["ocr_provenance"] = {
                "modality": "ocr",
                "derived_from": "image",
                "provider": getattr(extracted, "ocr_provider", None),
                "status": str(getattr(extracted, "ocr_status", "none") or "none"),
                "quality": dict(
                    getattr(
                        getattr(extracted, "ocr_quality", None), "to_dict", lambda: {}
                    )()
                ),
                "layout": dict(ocr_layout_summary),
            }
        if bool(chunk_row.get("include_semantic_image")) and image_meta_payload:
            item_meta["semantic_image"] = dict(image_meta_payload)
            item_meta["attachment_modality"] = "image_semantic"
        items.append(
            ContentItem(
                doc_id=f"{request_id}:c{idx:03d}",
                source_id=source_id,
                source_type=source_type,
                trust=str(source_trust),
                text=text_chunk,
                artifact_id=artifact_id,
                content_hash=content_hash,
                boundary_step=int(request_boundary_step),
                meta=item_meta,
            )
        )
    items, pressure_dedupe = dedupe_pressure_items_step_local(
        items=items,
        current_step=int(request_boundary_step),
    )

    chunk_agg = _score_chunks(
        projector=runtime.projector,
        items=items,
        walls=cfg["omega"]["walls"],
        cfg=api_cfg.get("chunk_pipeline", {})
        if isinstance(api_cfg.get("chunk_pipeline", {}), dict)
        else {},
    )
    chunk_kind_by_doc_id = {
        str(item.doc_id): _attachment_chunk_kind_for(item) for item in items
    }
    projection_by_doc_id = {
        str(proj.doc_id): proj for proj in list(chunk_agg.projections)
    }
    ocr_active_walls = sorted(
        {
            str(w)
            for row in list(chunk_agg.chunk_scores)
            if chunk_kind_by_doc_id.get(str(row.doc_id), "") == "ocr"
            for w in list(row.active_walls)
            if str(w).strip()
        }
    )
    image_active_walls = sorted(
        {
            str(w)
            for row in list(chunk_agg.chunk_scores)
            if chunk_kind_by_doc_id.get(str(row.doc_id), "") == "image_semantic"
            for w in list(row.active_walls)
            if str(w).strip()
        }
    )
    ocr_positive = bool(ocr_active_walls)
    vision_positive = bool(image_active_walls)
    ocr_vision_agreement = bool(set(ocr_active_walls) & set(image_active_walls))
    modality_positive_chunk_counts: Dict[str, int] = {}
    for row in list(chunk_agg.chunk_scores):
        kind = chunk_kind_by_doc_id.get(str(row.doc_id), "")
        if kind and list(row.active_walls):
            modality_positive_chunk_counts[kind] = (
                int(modality_positive_chunk_counts.get(kind, 0)) + 1
            )
    active_modalities = sorted(
        [k for k, count in modality_positive_chunk_counts.items() if int(count) > 0]
    )
    ocr_only_positive = bool(ocr_positive and not vision_positive)
    vision_only_positive = bool(vision_positive and not ocr_positive)
    modality_wall_max: Dict[str, Dict[str, float]] = {"ocr": {}, "image_semantic": {}}
    for kind in ("ocr", "image_semantic"):
        rows = [
            row
            for row in list(chunk_agg.chunk_scores)
            if chunk_kind_by_doc_id.get(str(row.doc_id), "") == kind
        ]
        wall_max_local: Dict[str, float] = {str(w): 0.0 for w in cfg["omega"]["walls"]}
        for row in rows:
            for wall, score in dict(row.wall_scores).items():
                if float(score) > float(wall_max_local.get(str(wall), 0.0)):
                    wall_max_local[str(wall)] = float(score)
        modality_wall_max[kind] = wall_max_local
    ocr_max_chunk_score_band = _max_chunk_score_band(
        chunk_scores=list(chunk_agg.chunk_scores),
        chunk_kind_by_doc_id=chunk_kind_by_doc_id,
        kind="ocr",
    )
    vision_max_chunk_score_band = _max_chunk_score_band(
        chunk_scores=list(chunk_agg.chunk_scores),
        chunk_kind_by_doc_id=chunk_kind_by_doc_id,
        kind="image_semantic",
    )
    ocr_only_pressure = bool(
        bool(getattr(extracted, "is_image", False))
        and str(getattr(extracted, "ocr_status", "none") or "none") == "success"
        and ocr_only_positive
    )

    projection_phase = evaluate_projection_phase(
        cfg=cfg, projections=list(chunk_agg.projections)
    )
    projections = list(projection_phase.projections)
    semantic_failure_policy = str(projection_phase.semantic_failure_policy)
    semantic_failure_detected = bool(projection_phase.semantic_failure_detected)
    if semantic_failure_detected and semantic_failure_policy == "fail_closed":
        raise HTTPException(status_code=503, detail="semantic_failure_fail_closed")
    core_projections = [
        (
            _zero_projection_signal(proj)
            if (
                ocr_only_pressure
                and chunk_kind_by_doc_id.get(str(item.doc_id), "") == "ocr"
            )
            else proj
        )
        for item, proj in zip(list(items), list(projections))
    ]
    if runtime_mode == "stateful":
        if session_store is None or not session_id:
            raise HTTPException(
                status_code=503, detail="stateful_runtime_not_configured"
            )
        state = OmegaState(
            session_id=f"api:{tenant_id}:{session_id}",
            m=np.asarray(state_vec, dtype=float),
            step=int(state_step_prev),
        )
        content_state = OmegaState(
            session_id=state.session_id,
            m=np.zeros_like(np.asarray(state_vec, dtype=float)),
            step=int(state_step_prev),
        )
        if runtime.cross_session is not None:
            hydrated = runtime.cross_session.hydrate_actor_state(
                actor_id=actor_id, session_id=state.session_id
            )
            state.m = np.maximum(
                state.m, np.asarray(hydrated.carried_scars_after_decay, dtype=float)
            )
            cross_carryover_applied = bool(hydrated.carryover_applied)
    else:
        state = OmegaState(
            session_id=f"api:{tenant_id}:{request_id}",
            m=np.zeros(4, dtype=float),
            step=0,
        )
        content_state = state
    state_base_m = np.asarray(state.m, dtype=float)
    state_base_step = int(state.step)
    content_state_base_m = np.asarray(content_state.m, dtype=float)
    content_state_base_step = int(content_state.step)

    def _run_core_bundle(
        projections_local: Sequence[ProjectionResult],
    ) -> tuple[Any, Any, bool]:
        state_local = OmegaState(
            session_id=str(state.session_id),
            m=np.asarray(state_base_m, dtype=float),
            step=int(state_base_step),
        )
        core_phase_local = run_core_step_phase(
            omega_core=runtime.omega_core,
            off_policy=runtime.off_policy,
            state=state_local,
            items=items,
            projections=projections_local,
        )
        content_core_phase_local = core_phase_local
        content_state_recomputed_local = False
        if runtime_mode == "stateful" and (
            int(state_step_prev) > 0 or bool(cross_carryover_applied)
        ):
            content_state_local = OmegaState(
                session_id=str(content_state.session_id),
                m=np.asarray(content_state_base_m, dtype=float),
                step=int(content_state_base_step),
            )
            content_core_phase_local = run_core_step_phase(
                omega_core=runtime.omega_core,
                off_policy=runtime.off_policy,
                state=content_state_local,
                items=items,
                projections=projections_local,
            )
            content_state_recomputed_local = True
        return (
            core_phase_local,
            content_core_phase_local,
            bool(content_state_recomputed_local),
        )

    def _compose_policy_state_bundle(
        *,
        step_result_local: Any,
        decision_local: Any,
        content_step_result_local: Any,
        content_decision_local: Any,
        cross_active_actions_local: Sequence[Any],
    ) -> Dict[str, Any]:
        try:
            content_actions_local = apply_semantic_failure_policy_to_actions(
                actions=list(content_decision_local.actions),
                semantic_phase=projection_phase,
                session_id=str(state.session_id),
                step=int(content_step_result_local.step),
            )
            content_action_types_local = normalize_action_types(content_actions_local)
            effective_actions_local = compose_effective_actions(
                policy_actions=list(
                    apply_semantic_failure_policy_to_actions(
                        actions=list(decision_local.actions),
                        semantic_phase=projection_phase,
                        session_id=str(state.session_id),
                        step=int(step_result_local.step),
                    )
                ),
                cross_active_actions=list(cross_active_actions_local),
            )
            effective_actions_local = apply_semantic_failure_policy_to_actions(
                actions=effective_actions_local,
                semantic_phase=projection_phase,
                session_id=str(state.session_id),
                step=int(step_result_local.step),
            )
        except RuntimeError as exc:
            if str(exc) == "semantic_failure_fail_closed":
                raise HTTPException(
                    status_code=503, detail="semantic_failure_fail_closed"
                ) from exc
            raise
        effective_action_types_local = normalize_action_types(effective_actions_local)
        phase_state_local = compose_control_outcome_state(
            walls=cfg["omega"]["walls"],
            step_result=step_result_local,
            policy_action_types=effective_action_types_local,
            cross_action_types=[],
            semantic_phase=projection_phase,
            extra_reason_flags=[str(x) for x in ingestion_flags if str(x).strip()]
            + list(chunk_agg.reasons),
        )
        content_phase_state_local = compose_control_outcome_state(
            walls=cfg["omega"]["walls"],
            step_result=content_step_result_local,
            policy_action_types=content_action_types_local,
            cross_action_types=[],
            semantic_phase=projection_phase,
            extra_reason_flags=[str(x) for x in ingestion_flags if str(x).strip()]
            + list(chunk_agg.reasons),
        )
        walls_triggered_local = list(phase_state_local.walls_triggered)
        content_walls_triggered_local = list(content_phase_state_local.walls_triggered)
        return {
            "content_action_types": list(content_action_types_local),
            "effective_action_types": list(effective_action_types_local),
            "phase_state": phase_state_local,
            "walls_triggered": walls_triggered_local,
            "action_types": list(phase_state_local.action_types),
            "intended_action_types": list(phase_state_local.intended_action_types),
            "reason_flags": list(phase_state_local.reason_flags),
            "content_walls_triggered": content_walls_triggered_local,
            "off": bool(step_result_local.off),
            "severity": str(decision_local.severity),
            "content_off": bool(content_step_result_local.off),
            "content_severity": str(content_decision_local.severity),
            "has_exfil": "secret_exfiltration" in walls_triggered_local,
            "content_has_exfil": "secret_exfiltration" in content_walls_triggered_local,
        }

    core_phase, content_core_phase, content_state_recomputed = _run_core_bundle(
        core_projections
    )
    step_result = core_phase.step_result
    decision = core_phase.policy_decision
    content_step_result = content_core_phase.step_result
    content_decision = content_core_phase.policy_decision
    cross_active_actions = []
    cross_active_action_types: List[str] = []
    cross_snapshot: Dict[str, Any] = {}
    if (
        runtime_mode == "stateful"
        and session_id
        and actor_id
        and runtime.cross_session is not None
    ):
        cross_active_actions = runtime.cross_session.active_actions(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_snapshot = runtime.cross_session.snapshot(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_active_action_types = sorted({str(a.type) for a in cross_active_actions})
        cross_part = (
            cross_snapshot.get("cross_session", {})
            if isinstance(cross_snapshot, dict)
            else {}
        ) or {}
        cross_carryover_applied = bool(
            cross_part.get("carryover_applied", cross_carryover_applied)
        )
        actor_hash_val = str(cross_part.get("actor_hash", "")).strip()
        cross_actor_hash = actor_hash_val if actor_hash_val else None
    policy_state = _compose_policy_state_bundle(
        step_result_local=step_result,
        decision_local=decision,
        content_step_result_local=content_step_result,
        content_decision_local=content_decision,
        cross_active_actions_local=cross_active_actions,
    )
    content_action_types = list(policy_state["content_action_types"])
    phase_state = policy_state["phase_state"]
    walls_triggered = list(policy_state["walls_triggered"])
    action_types = list(policy_state["action_types"])
    intended_action_types = list(policy_state["intended_action_types"])
    reasons = list(policy_state["reason_flags"])
    content_walls_triggered = list(policy_state["content_walls_triggered"])
    off = bool(policy_state["off"])
    severity = str(policy_state["severity"])
    content_off = bool(policy_state["content_off"])
    content_severity = str(policy_state["content_severity"])
    has_exfil = bool(policy_state["has_exfil"])
    content_has_exfil = bool(policy_state["content_has_exfil"])
    api_status = aggregate_semantic_execution_trace(list(chunk_agg.projections))
    if not api_status:
        api_status_fn = getattr(runtime.projector, "api_perception_status", None)
        api_status = api_status_fn() if callable(api_status_fn) else {}
    skillbox_shadow = evaluate_skillbox_shadow(
        config=runtime.config,
        items=items,
        user_query=str(getattr(extracted, "text", "") or ""),
        source_meta={
            "tenant_id": str(tenant_id),
            "request_id": str(request_id),
            "session_id": str(session_id or ""),
            "actor_id": str(actor_id or ""),
            "surface": "api",
        },
        skillbox=getattr(runtime, "skillbox", None),
    )
    effect_shadow = evaluate_typed_effect_shadow(
        config=runtime.config,
        projector=runtime.projector,
        items=items,
        user_query=str(getattr(extracted, "text", "") or ""),
        source_meta={
            "tenant_id": str(tenant_id),
            "request_id": str(request_id),
            "session_id": str(session_id or ""),
            "actor_id": str(actor_id or ""),
            "surface": "api",
        },
        skillbox=getattr(runtime, "skillbox", None),
    )
    effect_shadow = {
        **dict(effect_shadow),
        **{
            key: value
            for key, value in dict(skillbox_shadow).items()
            if key
            in {
                "named_skill_invocation",
                "skill_provenance_assessment",
                "skillbox_status",
                "skillbox_verification",
                "skillbox_ledger_hit",
                "skillbox_content_sha256",
                "skillbox_capabilities",
                "skillbox_gate_decision",
            }
        },
    }
    artifact_integrity = _build_api_artifact_integrity(
        config=runtime.config,
        items=items,
        source_id=str(source_id),
        source_type=str(source_type),
        source_trust=str(source_trust),
        request_boundary_step=int(request_boundary_step),
        request_id=str(request_id),
        effect_shadow=effect_shadow,
    )
    operation_gate_events = []
    if isinstance(effect_shadow.get("skillbox_verification"), dict):
        skillbox_verification = dict(effect_shadow["skillbox_verification"])
        operation_gate_events.append(
            {
                "status": str(effect_shadow.get("skillbox_gate_decision", "shadow_only")),
                "reason_code": str(skillbox_verification.get("reason_code", "")),
                "shadow_only": True,
                "would_enforce": False,
                "requires_approval": bool(skillbox_verification.get("requires_approval", False)),
                "hard_invariant_hits": [],
                "details": {
                    "operation_type": "skill_run",
                    "target": str(skillbox_verification.get("skill_name", "") or "unknown"),
                    "provenance_status": str(skillbox_verification.get("verification_status", "")),
                    "simulated_block": bool(skillbox_verification.get("simulated_block", False)),
                    "ledger_hit": bool(effect_shadow.get("skillbox_ledger_hit", False)),
                },
            }
        )
    elif isinstance(effect_shadow.get("skill_provenance_assessment"), dict):
        skill_provenance = dict(effect_shadow["skill_provenance_assessment"])
        operation_gate_events.append(
            {
                "status": "shadow_only",
                "reason_code": str(skill_provenance.get("reason_code", "")),
                "shadow_only": True,
                "would_enforce": False,
                "requires_approval": bool(skill_provenance.get("requires_approval", False)),
                "hard_invariant_hits": [],
                "details": {
                    "operation_type": "skill_run",
                    "target": str(skill_provenance.get("skill_name", "") or "unknown"),
                    "provenance_status": str(skill_provenance.get("status", "")),
                    "simulated_block": bool(skill_provenance.get("simulated_block", False)),
                },
            }
        )
    image_semantic_only = "image_semantic_only" in set(ingestion_flags)
    image_semantic_active = bool(
        image_semantic_only
        and str((api_status or {}).get("vision_semantic_status", "")).strip().lower()
        == "vision_semantic_active"
        and str((api_status or {}).get("semantic_status", "")).strip().lower()
        != "semantic_failed"
    )
    ingestion_override = any(
        x in reasons for x in ("scan_like", "zip_deferred_runtime", "ingestion_error")
    ) or (("text_empty" in set(reasons)) and not bool(image_semantic_active))

    policy_mapper_cfg = (
        api_cfg.get("policy_mapper", {})
        if isinstance(api_cfg.get("policy_mapper", {}), dict)
        else {}
    )
    block_score_threshold = float(policy_mapper_cfg.get("block_score_threshold", 0.72))
    quarantine_score_threshold = float(
        policy_mapper_cfg.get("quarantine_score_threshold", 0.45)
    )
    quarantine_worst_threshold = float(
        policy_mapper_cfg.get("quarantine_worst_threshold", 0.38)
    )
    quarantine_synergy_threshold = float(
        policy_mapper_cfg.get("quarantine_synergy_threshold", 0.20)
    )
    exfil_block_wall_threshold = float(
        policy_mapper_cfg.get("exfil_block_wall_threshold", 0.60)
    )
    confidence_block_threshold = float(
        policy_mapper_cfg.get("confidence_block_threshold", 0.55)
    )
    hgl_cfg = (
        policy_mapper_cfg.get("hallucination_guard_lite", {})
        if isinstance(policy_mapper_cfg.get("hallucination_guard_lite", {}), dict)
        else {}
    )

    chunk_block = float(chunk_agg.doc_score) >= block_score_threshold or (
        float(chunk_agg.wall_max.get("secret_exfiltration", 0.0))
        >= exfil_block_wall_threshold
        and float(chunk_agg.confidence) >= confidence_block_threshold
    )
    chunk_quarantine = (
        float(chunk_agg.doc_score) >= quarantine_score_threshold
        or float(chunk_agg.worst_chunk_score) >= quarantine_worst_threshold
        or float(chunk_agg.pattern_synergy) >= quarantine_synergy_threshold
    )
    top_chunk_lookup = {str(item.doc_id): item for item in list(items)}

    if ingestion_override:
        content_verdict = "quarantine"
    elif chunk_block or (
        content_off and (content_severity == "L3" or content_has_exfil)
    ):
        content_verdict = "block"
    elif chunk_quarantine or content_off:
        content_verdict = "quarantine"
    else:
        content_verdict = "allow"
    content_verdict_before_ocr_gate = str(content_verdict)
    content_control_outcome = _resolve_control_outcome(
        action_types=content_action_types, verdict=content_verdict
    )
    if ingestion_override:
        verdict = "quarantine"
    elif chunk_block or (off and (severity == "L3" or has_exfil)):
        verdict = "block"
    elif chunk_quarantine or off:
        verdict = "quarantine"
    else:
        verdict = "allow"
    source_quarantine_active = any(
        str(a.type) == "SOURCE_QUARANTINE" and source_id in set(a.source_ids or [])
        for a in cross_active_actions
    )
    tool_freeze_active = any(str(a.type) == "TOOL_FREEZE" for a in cross_active_actions)
    if source_quarantine_active and verdict == "allow":
        verdict = "quarantine"
    if source_quarantine_active:
        reasons.append("source_quarantine_active")
    if tool_freeze_active:
        reasons.append("tool_freeze_active")
    ocr_gate_applied = False
    ocr_gate_reason = "none"
    ocr_adjudication_status = "not_needed"
    ocr_adjudication_result = "none"
    ocr_adjudication_trace: Dict[str, Any] = {
        "attempted": False,
        "triggered_span_ids": [],
        "matched_span_ids": [],
        "context_span_ids": [],
        "supporting_span_ids": [],
        "crop_rect_px": None,
        "crop_sha256": None,
        "candidate_text_sha256": None,
        "confirmed_walls": [],
        "negative_walls": [],
        "confidence": None,
        "semantic_status": "none",
    }
    if ocr_only_pressure:
        adjudication_items: List[ContentItem] = []
        top_ocr_row = next(
            (
                row
                for row in list(chunk_agg.top_chunks)
                if chunk_kind_by_doc_id.get(str(row.get("doc_id", "")), "") == "ocr"
            ),
            None,
        )
        if isinstance(top_ocr_row, Mapping):
            top_item = top_chunk_lookup.get(str(top_ocr_row.get("doc_id", "")).strip())
            top_meta = (
                top_item.meta
                if top_item is not None and isinstance(top_item.meta, Mapping)
                else {}
            )
            chunk_span_ids = [
                str(x)
                for x in list(top_meta.get("ocr_span_ids", []) or [])
                if str(x).strip()
            ]
            matched_span_ids = _matched_span_ids_for_projection(
                item=top_item,
                projection=projection_by_doc_id.get(
                    str(top_ocr_row.get("doc_id", "")).strip()
                ),
                ocr_span_lookup=ocr_span_lookup,
                active_walls=[
                    str(x)
                    for x in list(top_ocr_row.get("active_walls", []) or [])
                    if str(x).strip()
                ],
            )
            triggered_span_ids = list(matched_span_ids or chunk_span_ids)
            if triggered_span_ids:
                source_walls = [
                    str(x)
                    for x in list(top_ocr_row.get("active_walls", []) or [])
                    if str(x).strip()
                ]
                adjudication_items, adjudication_build_trace = (
                    build_ocr_adjudication_items(
                        request_id=str(request_id),
                        source_id=str(source_id),
                        source_type=str(source_type),
                        trust=str(source_trust),
                        file_bytes=bytes(parsed.get("file_bytes") or b""),
                        triggered_span_ids=triggered_span_ids,
                        matched_span_ids=matched_span_ids,
                        supporting_span_ids=chunk_span_ids,
                        source_walls=source_walls,
                        span_lookup=ocr_span_lookup,
                        source_image_meta=(image_meta_payload or {}),
                        crop_strategy="contextual",
                        context_span_radius=2,
                        max_context_spans=5,
                        min_crop_width_px=160.0,
                        min_crop_height_px=72.0,
                        include_candidate_text=True,
                        variant_id="contextual_image_text",
                        register_image_payload=lambda **payload: (
                            _register_image_payload(
                                projector=runtime.projector,
                                scope_id=str(request_id),
                                raw=bytes(payload["raw"]),
                                mime=str(payload["mime"]),
                                sha256=str(payload["sha256"]),
                                role=str(payload["role"]),
                                width=int(payload["width"]),
                                height=int(payload["height"]),
                            )
                        ),
                    )
                )
                ocr_adjudication_trace["source_walls"] = list(source_walls)
                ocr_adjudication_trace["build"] = dict(adjudication_build_trace)
                ocr_adjudication_trace["triggered_span_ids"] = list(triggered_span_ids)
                ocr_adjudication_trace["supporting_span_ids"] = list(chunk_span_ids)
                ocr_adjudication_trace["matched_span_ids"] = list(matched_span_ids)
                ocr_adjudication_trace["exact_attribution"] = bool(matched_span_ids)
            else:
                adjudication_items = []
        if not adjudication_items:
            ocr_adjudication_status = "required_unavailable_no_layout"
            if content_verdict_before_ocr_gate == "block":
                ocr_adjudication_result = "degraded_block_to_quarantine"
            elif content_verdict_before_ocr_gate == "quarantine":
                ocr_adjudication_result = "preserved_quarantine"
            else:
                ocr_adjudication_result = "preserved_allow_no_layout"
            if content_verdict_before_ocr_gate != "allow" and content_verdict != "quarantine":
                content_verdict = "quarantine"
                verdict = "quarantine"
                content_control_outcome = _resolve_control_outcome(
                    action_types=content_action_types, verdict=content_verdict
                )
                ocr_gate_applied = True
            if not vision_positive:
                ocr_gate_reason = "ocr_without_vision_confirmation"
            else:
                ocr_gate_reason = "ocr_without_wall_agreement"
            reasons.extend(
                [
                    "ocr_provenance_present",
                    "ocr_unconfirmed_capped",
                    str(ocr_gate_reason),
                ]
            )
        else:
            ocr_adjudication_trace["attempted"] = True
            try:
                adjudication_threshold = (
                    api_cfg.get("chunk_pipeline", {}).get(
                        "wall_trigger_threshold", 0.12
                    )
                    if isinstance(api_cfg.get("chunk_pipeline", {}), dict)
                    else 0.12
                )
                tile_traces: List[Dict[str, Any]] = []
                live_attack_hits = 0
                benign_ui_hits = 0
                quoted_or_defensive_hits = 0
                insufficient_context_hits = 0
                for adjudication_item in list(adjudication_items):
                    adjudication_meta = (
                        adjudication_item.meta
                        if isinstance(adjudication_item.meta, Mapping)
                        else {}
                    )
                    adjudication_target = (
                        adjudication_meta.get("ocr_adjudication_target", {})
                        if isinstance(
                            adjudication_meta.get("ocr_adjudication_target", {}),
                            Mapping,
                        )
                        else {}
                    )
                    adjudication_proj = runtime.projector.project(adjudication_item)
                    adjudication_eval = interpret_ocr_adjudication_projection(
                        projection=adjudication_proj,
                        source_walls=list(
                            ocr_adjudication_trace.get("source_walls", []) or []
                        ),
                        threshold=float(adjudication_threshold),
                    )
                    tile_traces.append(
                        {
                            "variant_id": str(
                                adjudication_target.get("variant_id", "")
                            ),
                            "crop_strategy": str(
                                adjudication_target.get("crop_strategy", "")
                            ),
                            "tile_index": int(
                                adjudication_target.get("tile_index", len(tile_traces))
                            ),
                            "context_span_ids": [
                                str(x)
                                for x in list(
                                    adjudication_target.get("context_span_ids", [])
                                    or []
                                )
                                if str(x).strip()
                            ],
                            "candidate_span_ids": [
                                str(x)
                                for x in list(
                                    adjudication_target.get("candidate_span_ids", [])
                                    or []
                                )
                                if str(x).strip()
                            ],
                            "crop_rect_px": dict(
                                adjudication_target.get("crop_rect_px", {}) or {}
                            )
                            or None,
                            "crop_area_ratio": adjudication_target.get(
                                "crop_area_ratio"
                            ),
                            "crop_sha256": (
                                str(adjudication_target.get("crop_sha256", "")) or None
                            ),
                            "candidate_text_sha256": (
                                str(
                                    adjudication_target.get("candidate_text_sha256", "")
                                )
                                or None
                            ),
                            "exact_attribution": bool(
                                adjudication_target.get("exact_attribution", False)
                            ),
                            "result": str(adjudication_eval["result"]),
                            "confirmed_walls": list(
                                adjudication_eval["confirmed_walls"]
                            ),
                            "negative_walls": list(adjudication_eval["negative_walls"]),
                            "confidence": float(adjudication_eval["confidence"]),
                            "directive_intent": dict(
                                adjudication_eval.get("directive_intent", {}) or {}
                            ),
                            "semantic_status": str(
                                adjudication_eval["semantic_status"]
                            ),
                        }
                    )
                    if str(adjudication_eval["result"]) == "live_attack":
                        live_attack_hits += 1
                    elif str(adjudication_eval["result"]) == "benign_ui":
                        benign_ui_hits += 1
                    elif str(adjudication_eval["result"]) == "quoted_or_defensive":
                        quoted_or_defensive_hits += 1
                    else:
                        insufficient_context_hits += 1
                ocr_adjudication_trace["tiles"] = list(tile_traces)
                ocr_adjudication_trace["tile_count"] = int(len(tile_traces))
                ocr_adjudication_trace["context_span_ids"] = [
                    str(x)
                    for row in tile_traces
                    for x in list(row.get("context_span_ids", []) or [])
                    if str(x).strip()
                ]
                ocr_adjudication_trace["supporting_span_ids"] = sorted(
                    {
                        str(x)
                        for x in (
                            list(
                                ocr_adjudication_trace.get("supporting_span_ids", [])
                                or []
                            )
                            + [
                                str(y)
                                for row in tile_traces
                                for y in list(row.get("context_span_ids", []) or [])
                                if str(y).strip()
                            ]
                        )
                        if str(x).strip()
                    }
                )
                ocr_adjudication_trace["crop_rect_px"] = (
                    tile_traces[0].get("crop_rect_px") if tile_traces else None
                )
                ocr_adjudication_trace["crop_sha256"] = (
                    tile_traces[0].get("crop_sha256") if tile_traces else None
                )
                ocr_adjudication_trace["candidate_text_sha256"] = (
                    tile_traces[0].get("candidate_text_sha256") if tile_traces else None
                )
                ocr_adjudication_trace["candidate_span_ids"] = sorted(
                    {
                        str(x)
                        for row in tile_traces
                        for x in list(row.get("candidate_span_ids", []) or [])
                        if str(x).strip()
                    }
                )
                ocr_adjudication_trace["confirmed_walls"] = sorted(
                    {
                        str(w)
                        for row in tile_traces
                        for w in list(row.get("confirmed_walls", []) or [])
                        if str(w).strip()
                    }
                )
                ocr_adjudication_trace["negative_walls"] = sorted(
                    {
                        str(w)
                        for row in tile_traces
                        for w in list(row.get("negative_walls", []) or [])
                        if str(w).strip()
                    }
                )
                ocr_adjudication_trace["directive_intent"] = {
                    str(k): bool(v)
                    for row in tile_traces
                    for k, v in dict(row.get("directive_intent", {}) or {}).items()
                    if str(k).strip()
                }
                ocr_adjudication_trace["confidence"] = (
                    max(float(row.get("confidence", 0.0) or 0.0) for row in tile_traces)
                    if tile_traces
                    else None
                )
                ocr_adjudication_trace["semantic_status"] = (
                    tile_traces[0].get("semantic_status", "unknown")
                    if tile_traces
                    else "unknown"
                )
                if live_attack_hits > 0:
                    ocr_adjudication_trace["outcome"] = "live_attack"
                elif quoted_or_defensive_hits > 0:
                    ocr_adjudication_trace["outcome"] = "quoted_or_defensive"
                elif benign_ui_hits > 0:
                    ocr_adjudication_trace["outcome"] = "benign_ui"
                else:
                    ocr_adjudication_trace["outcome"] = "insufficient_context"
                exact_attribution = bool(
                    ocr_adjudication_trace.get("exact_attribution", False)
                )
                weak_unattributed_ocr_only_uncertain = bool(
                    not exact_attribution
                    and not vision_positive
                    and str(ocr_max_chunk_score_band) == "allow"
                    and str(vision_max_chunk_score_band) == "allow"
                    and live_attack_hits == 0
                    and insufficient_context_hits > 0
                )
                if live_attack_hits > 0 and exact_attribution:
                    ocr_adjudication_status = "completed"
                    ocr_adjudication_result = "live_attack"
                    (
                        confirmed_core_phase,
                        confirmed_content_core_phase,
                        confirmed_content_state_recomputed,
                    ) = _run_core_bundle(projections)
                    core_phase = confirmed_core_phase
                    content_core_phase = confirmed_content_core_phase
                    content_state_recomputed = bool(confirmed_content_state_recomputed)
                    step_result = core_phase.step_result
                    decision = core_phase.policy_decision
                    content_step_result = content_core_phase.step_result
                    content_decision = content_core_phase.policy_decision
                    policy_state = _compose_policy_state_bundle(
                        step_result_local=step_result,
                        decision_local=decision,
                        content_step_result_local=content_step_result,
                        content_decision_local=content_decision,
                        cross_active_actions_local=cross_active_actions,
                    )
                    content_action_types = list(policy_state["content_action_types"])
                    phase_state = policy_state["phase_state"]
                    walls_triggered = list(policy_state["walls_triggered"])
                    action_types = list(policy_state["action_types"])
                    intended_action_types = list(policy_state["intended_action_types"])
                    reasons = list(policy_state["reason_flags"])
                    content_walls_triggered = list(
                        policy_state["content_walls_triggered"]
                    )
                    off = bool(policy_state["off"])
                    severity = str(policy_state["severity"])
                    content_off = bool(policy_state["content_off"])
                    content_severity = str(policy_state["content_severity"])
                    has_exfil = bool(policy_state["has_exfil"])
                    content_has_exfil = bool(policy_state["content_has_exfil"])
                    if source_quarantine_active:
                        reasons.append("source_quarantine_active")
                    if tool_freeze_active:
                        reasons.append("tool_freeze_active")
                    content_verdict = "block"
                    verdict = "block"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types, verdict=content_verdict
                    )
                    ocr_gate_reason = "ocr_confirmed_by_targeted_vision"
                    reasons.extend(
                        ["ocr_provenance_present", "ocr_targeted_vision_confirmed"]
                    )
                elif live_attack_hits > 0:
                    ocr_adjudication_status = "completed"
                    if content_verdict_before_ocr_gate == "block":
                        ocr_adjudication_result = (
                            "insufficient_context_degraded_block_to_quarantine"
                        )
                    elif content_verdict_before_ocr_gate == "quarantine":
                        ocr_adjudication_result = (
                            "insufficient_context_preserved_quarantine"
                        )
                    else:
                        ocr_adjudication_result = (
                            "insufficient_context_promoted_allow_to_quarantine"
                        )
                    if content_verdict != "quarantine":
                        content_verdict = "quarantine"
                        verdict = "quarantine"
                        content_control_outcome = _resolve_control_outcome(
                            action_types=content_action_types,
                            verdict=content_verdict,
                        )
                        ocr_gate_applied = True
                    ocr_gate_reason = "ocr_missing_exact_attribution"
                    reasons.extend(
                        [
                            "ocr_provenance_present",
                            "ocr_unconfirmed_capped",
                            str(ocr_gate_reason),
                        ]
                    )
                elif (
                    quoted_or_defensive_hits > 0
                    and insufficient_context_hits == 0
                    and live_attack_hits == 0
                ):
                    ocr_adjudication_status = "completed"
                    ocr_adjudication_result = "quoted_or_defensive"
                    content_verdict = "allow"
                    verdict = "allow"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types, verdict=content_verdict
                    )
                    ocr_gate_applied = content_verdict_before_ocr_gate != "allow"
                    ocr_gate_reason = "ocr_benign_context_confirmed"
                    reasons.extend(
                        ["ocr_provenance_present", "ocr_targeted_vision_benign"]
                    )
                elif (
                    benign_ui_hits > 0
                    and insufficient_context_hits == 0
                    and live_attack_hits == 0
                ):
                    ocr_adjudication_status = "completed"
                    ocr_adjudication_result = "benign_ui"
                    content_verdict = "allow"
                    verdict = "allow"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types, verdict=content_verdict
                    )
                    ocr_gate_applied = content_verdict_before_ocr_gate != "allow"
                    ocr_gate_reason = "ocr_benign_ui_confirmed"
                    reasons.extend(
                        ["ocr_provenance_present", "ocr_targeted_vision_benign_ui"]
                    )
                elif weak_unattributed_ocr_only_uncertain:
                    ocr_adjudication_status = "completed"
                    ocr_adjudication_result = (
                        "insufficient_context_reverted_allow_weak_ocr_only"
                    )
                    content_verdict = "allow"
                    verdict = "allow"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types,
                        verdict=content_verdict,
                    )
                    ocr_gate_applied = content_verdict_before_ocr_gate != "allow"
                    ocr_gate_reason = "ocr_weak_unattributed_uncertain_allow"
                    reasons.extend(
                        [
                            "ocr_provenance_present",
                            "ocr_weak_signal_reverted_to_allow",
                            str(ocr_gate_reason),
                        ]
                    )
                else:
                    ocr_adjudication_status = "completed"
                    if content_verdict_before_ocr_gate == "block":
                        ocr_adjudication_result = (
                            "insufficient_context_degraded_block_to_quarantine"
                        )
                    elif content_verdict_before_ocr_gate == "quarantine":
                        ocr_adjudication_result = (
                            "insufficient_context_preserved_quarantine"
                        )
                    else:
                        ocr_adjudication_result = (
                            "insufficient_context_preserved_allow_uncertain"
                        )
                    if content_verdict_before_ocr_gate != "allow" and content_verdict != "quarantine":
                        content_verdict = "quarantine"
                        verdict = "quarantine"
                        content_control_outcome = _resolve_control_outcome(
                            action_types=content_action_types,
                            verdict=content_verdict,
                        )
                        ocr_gate_applied = True
                    ocr_gate_reason = "ocr_targeted_vision_uncertain"
                    reasons.extend(
                        [
                            "ocr_provenance_present",
                            "ocr_unconfirmed_capped",
                            str(ocr_gate_reason),
                        ]
                    )
            except Exception:
                ocr_adjudication_status = "failed_provider"
                if content_verdict_before_ocr_gate == "block":
                    ocr_adjudication_result = (
                        "provider_failed_degraded_block_to_quarantine"
                    )
                elif content_verdict_before_ocr_gate == "quarantine":
                    ocr_adjudication_result = "provider_failed_preserved_quarantine"
                else:
                    ocr_adjudication_result = "provider_failed_preserved_allow"
                if content_verdict_before_ocr_gate != "allow" and content_verdict != "quarantine":
                    content_verdict = "quarantine"
                    verdict = "quarantine"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types, verdict=content_verdict
                    )
                    ocr_gate_applied = True
                ocr_gate_reason = "ocr_targeted_vision_failed"
                reasons.extend(
                    [
                        "ocr_provenance_present",
                        "ocr_unconfirmed_capped",
                        str(ocr_gate_reason),
                    ]
                )
    elif ocr_vision_agreement:
        ocr_adjudication_status = "not_needed_agreement"
        ocr_adjudication_result = "agreement"
    elif vision_only_positive:
        ocr_adjudication_status = "not_needed_vision_only_positive"
        ocr_adjudication_result = "vision_kept"
    intended_verdict = str(verdict)
    intended_control_outcome = _resolve_control_outcome(
        action_types=action_types, verdict=verdict
    )
    if (
        runtime_mode == "stateful"
        and session_id
        and actor_id
        and runtime.cross_session is not None
    ):
        runtime.cross_session.record_step(
            actor_id=actor_id,
            session_id=state.session_id,
            step_result=step_result,
            policy_actions=decision.actions,
            packet_items=items,
        )
        cross_active_actions = runtime.cross_session.active_actions(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_snapshot = runtime.cross_session.snapshot(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_active_action_types = sorted({str(a.type) for a in cross_active_actions})
        cross_part = (
            cross_snapshot.get("cross_session", {})
            if isinstance(cross_snapshot, dict)
            else {}
        ) or {}
        cross_carryover_applied = bool(
            cross_part.get("carryover_applied", cross_carryover_applied)
        )
        actor_hash_val = str(cross_part.get("actor_hash", "")).strip()
        cross_actor_hash = actor_hash_val if actor_hash_val else None

    trace_top_chunks: List[Dict[str, Any]] = []
    for row in list(chunk_agg.top_chunks):
        doc_id = str(row.get("doc_id", "")).strip()
        item = top_chunk_lookup.get(doc_id)
        if item is None:
            trace_top_chunks.append(dict(row))
            continue
        item_meta = item.meta if isinstance(item.meta, Mapping) else {}
        chunk_span_ids = [
            str(x)
            for x in list(item_meta.get("ocr_span_ids", []) or [])
            if str(x).strip()
        ]
        matched_span_ids = _matched_span_ids_for_projection(
            item=item,
            projection=projection_by_doc_id.get(doc_id),
            ocr_span_lookup=ocr_span_lookup,
            active_walls=[
                str(x)
                for x in list(row.get("active_walls", []) or [])
                if str(x).strip()
            ],
        )
        triggered_span_ids = list(matched_span_ids or chunk_span_ids)
        triggered_spans: List[Dict[str, Any]] = []
        for span_id in triggered_span_ids:
            span = ocr_span_lookup.get(span_id)
            if span is None:
                continue
            span_text = str(getattr(span, "text", "") or "")
            polygon_px = getattr(span, "polygon_px", None)
            triggered_spans.append(
                {
                    "span_id": str(span_id),
                    "text_sha256": _trace_sha256(span_text),
                    "redacted_excerpt": _redacted_excerpt(span_text),
                    "confidence": (
                        float(getattr(span, "confidence", 0.0))
                        if getattr(span, "confidence", None) is not None
                        else None
                    ),
                    "polygon_px": polygon_px,
                    "rect_px": polygon_to_rect_px(polygon_px),
                    "image_width": int(getattr(span, "image_width", 0) or 0),
                    "image_height": int(getattr(span, "image_height", 0) or 0),
                    "provider_order": (
                        int(getattr(span, "provider_order", 0))
                        if getattr(span, "provider_order", None) is not None
                        else None
                    ),
                    "char_start": (
                        int(getattr(span, "char_start", 0))
                        if getattr(span, "char_start", None) is not None
                        else None
                    ),
                    "char_end": (
                        int(getattr(span, "char_end", 0))
                        if getattr(span, "char_end", None) is not None
                        else None
                    ),
                }
            )
        trace_row = dict(row)
        if triggered_span_ids:
            trace_row["chunk_span_ids"] = chunk_span_ids
            trace_row["matched_span_ids"] = matched_span_ids
            trace_row["triggered_span_ids"] = triggered_span_ids
            trace_row["triggered_spans"] = triggered_spans
        trace_top_chunks.append(trace_row)
    source_risk_band = _source_risk_band(items)
    allowed_trust_bands = {
        _normalize_trust_band(x)
        for x in list(hgl_cfg.get("apply_when_source_trust", ["untrusted", "mixed"]))
        if str(x).strip()
    }
    if not allowed_trust_bands:
        allowed_trust_bands = {"untrusted", "mixed"}
    low_confidence_lte = float(hgl_cfg.get("low_confidence_lte", 0.35))
    only_if_intended_allow = bool(hgl_cfg.get("only_if_intended_allow", True))
    hallucination_reason_code = "hallucination_guard_lite_low_confidence_untrusted"
    hallucination_guard_triggered = False
    response_constraints: Dict[str, Any] = {
        "enabled": False,
        "disclaimer_required": False,
        "citation_required": False,
        "reason_code": None,
        "citation_candidates": [],
        "suggested_mode": None,
    }
    hallucination_guard_summary: Dict[str, Any] = {
        "triggered": False,
        "source_risk_band": str(source_risk_band),
        "confidence": float(chunk_agg.confidence),
        "reason_code": None,
    }
    hallucination_guard_enabled = bool(hgl_cfg.get("enabled", False))
    if hallucination_guard_enabled:
        low_confidence_hit = float(chunk_agg.confidence) <= low_confidence_lte
        trust_band_hit = str(source_risk_band) in allowed_trust_bands
        intended_allow_hit = (
            (str(intended_control_outcome) == "ALLOW")
            if only_if_intended_allow
            else True
        )
        if low_confidence_hit and trust_band_hit and intended_allow_hit:
            hallucination_guard_triggered = True
            reasons.append(hallucination_reason_code)
            if "WARN" not in set(intended_action_types):
                intended_action_types = sorted(
                    set(list(intended_action_types) + ["WARN"])
                )
            if "WARN" not in set(content_action_types):
                content_action_types = sorted(
                    set(list(content_action_types) + ["WARN"])
                )
            intended_control_outcome = "WARN"
            content_control_outcome = "WARN"
            response_constraints = {
                "enabled": True,
                "disclaimer_required": True,
                "citation_required": True,
                "reason_code": hallucination_reason_code,
                "citation_candidates": [
                    {
                        "doc_id": str(doc_id),
                        "source_id": str(top_chunk_lookup[doc_id].source_id),
                        "trust": str(top_chunk_lookup[doc_id].trust),
                    }
                    for row in list(chunk_agg.top_chunks)
                    for doc_id in [str(row.get("doc_id", "")).strip()]
                    if doc_id and doc_id in top_chunk_lookup
                ],
                "suggested_mode": "answer_with_uncertainty_and_citations",
            }
            soft_q_cfg = (
                hgl_cfg.get("soft_quarantine", {})
                if isinstance(hgl_cfg.get("soft_quarantine", {}), dict)
                else {}
            )
            if bool(soft_q_cfg.get("enabled", False)):
                mixed_ok = (not bool(soft_q_cfg.get("mixed_only", True))) or (
                    str(source_risk_band) == "mixed"
                )
                very_low_hit = float(chunk_agg.confidence) <= float(
                    soft_q_cfg.get("very_low_confidence_lte", 0.20)
                )
                synergy_hit = float(chunk_agg.pattern_synergy) >= float(
                    soft_q_cfg.get("pattern_synergy_gte", 0.30)
                )
                if (
                    mixed_ok
                    and very_low_hit
                    and synergy_hit
                    and str(intended_verdict).lower() == "allow"
                ):
                    intended_verdict = "quarantine"
                    verdict = "quarantine"
                    content_verdict = "quarantine"
                    content_control_outcome = _resolve_control_outcome(
                        action_types=content_action_types,
                        verdict=content_verdict,
                    )
        hallucination_guard_summary = {
            "triggered": bool(hallucination_guard_triggered),
            "source_risk_band": str(source_risk_band),
            "confidence": float(chunk_agg.confidence),
            "reason_code": (
                hallucination_reason_code if hallucination_guard_triggered else None
            ),
        }

    effective_verdict = str(intended_verdict)
    effective_control_outcome = str(intended_control_outcome)
    carryover_action_types = list(cross_active_action_types)
    if (
        ocr_only_pressure
        and str(ocr_adjudication_result)
        not in {
            "live_attack",
            "quoted_or_defensive",
            "benign_ui",
            "insufficient_context_reverted_allow_weak_ocr_only",
            "preserved_allow_no_layout",
            "insufficient_context_preserved_allow_uncertain",
            "provider_failed_preserved_allow",
        }
        and not carryover_action_types
        and str(effective_control_outcome) in {"ALLOW", "SOFT_BLOCK"}
    ):
        intended_verdict = "quarantine"
        intended_control_outcome = _resolve_control_outcome(
            action_types=intended_action_types,
            verdict=intended_verdict,
        )
        effective_verdict = str(intended_verdict)
        effective_control_outcome = str(intended_control_outcome)
        ocr_gate_applied = True
    stateful_carryover_present = bool(
        runtime_mode == "stateful"
        and (
            int(state_step_prev) > 0
            or bool(cross_carryover_applied)
            or bool(carryover_action_types)
        )
    )
    blocked_by_carryover = bool(
        stateful_carryover_present
        and str(content_control_outcome) == "ALLOW"
        and str(effective_control_outcome) != "ALLOW"
    )
    if "SOURCE_QUARANTINE" in set(carryover_action_types):
        effective_verdict = "quarantine"
    elif "TOOL_FREEZE" in set(carryover_action_types):
        effective_verdict = "block"

    action_types = list(intended_action_types)
    if monitor_enabled:
        verdict = "allow"
        control_outcome = "ALLOW"
        action_types = []
    else:
        verdict = str(intended_verdict)
        control_outcome = str(intended_control_outcome)
        if "SOURCE_QUARANTINE" in set(carryover_action_types):
            verdict = "quarantine"
        elif "TOOL_FREEZE" in set(carryover_action_types):
            verdict = "block"

    max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
    sum_m_next = float(np.sum(step_result.m_next))
    off_sigma = (
        float(runtime.omega_core.params.off_Sigma)
        if float(runtime.omega_core.params.off_Sigma) > 0
        else 1.0
    )
    m_ratio = min(1.0, sum_m_next / off_sigma)
    sev = {"L1": 0.0, "L2": 0.5, "L3": 1.0}.get(severity, 0.0)
    risk_legacy = int(
        round(_clamp(100.0 * (0.60 * max_p + 0.30 * m_ratio + 0.10 * sev), 0.0, 100.0))
    )
    risk_chunk = int(round(_clamp(100.0 * float(chunk_agg.doc_score), 0.0, 100.0)))
    risk_score = max(risk_legacy, risk_chunk)
    if (
        str(ocr_adjudication_result)
        in {
            "quoted_or_defensive",
            "benign_ui",
            "insufficient_context_reverted_allow_weak_ocr_only",
        }
        and str(intended_control_outcome) == "ALLOW"
    ):
        risk_score = min(int(risk_score), 24)
    if ocr_only_pressure and str(intended_control_outcome) == "WARN":
        risk_score = min(int(risk_score), 69)
    if "zip_deferred_runtime" in reasons:
        risk_score = max(risk_score, 60)
    if ("scan_like" in reasons) or (
        ("text_empty" in reasons) and not bool(image_semantic_active)
    ):
        risk_score = max(risk_score, 55)

    reasons_sorted = sorted(reasons)
    evidence_id = str(uuid.uuid4())
    trace_id = build_trace_id_api(tenant_id=str(tenant_id), request_id=str(request_id))
    decision_id = build_decision_id(
        trace_id=trace_id,
        control_outcome=(
            str(intended_control_outcome) if monitor_enabled else control_outcome
        ),
        action_types=(intended_action_types if monitor_enabled else action_types),
        severity=severity,
        off=off,
    )
    evidence_summary = {
        "walls_triggered": list(walls_triggered),
        "rule_ids": list(getattr(chunk_agg, "rule_ids", []) or []),
        "chunk_ids": list(getattr(chunk_agg, "triggered_chunk_ids", []) or []),
        "top_chunk_ids": [str(x.get("doc_id", "")) for x in list(chunk_agg.top_chunks)],
        "text_included": False,
        "control_outcome": control_outcome,
        "trace_id": trace_id,
        "decision_id": decision_id,
    }

    payload: Dict[str, Any] = {
        "request_id": request_id,
        "trace_id": trace_id,
        "decision_id": decision_id,
        "tenant_id": tenant_id,
        "risk_score": int(risk_score),
        "verdict": verdict,
        "content_verdict": str(content_verdict),
        "content_control_outcome": str(content_control_outcome),
        "effective_verdict": str(effective_verdict),
        "effective_control_outcome": str(effective_control_outcome),
        "blocked_by_carryover": bool(blocked_by_carryover),
        "carryover_action_types": list(carryover_action_types),
        "control_outcome": control_outcome,
        "reasons": reasons_sorted,
        "evidence_id": evidence_id,
        "effect_wall_candidate": effect_shadow.get("effect_wall_candidate"),
        "effect_policy_gate": effect_shadow.get("effect_policy_gate"),
        "effect_policy_gate_status": str(
            effect_shadow.get("effect_policy_gate_status", "disabled")
        ),
        "effect_forecast_status": str(effect_shadow.get("effect_forecast_status", "disabled")),
        "named_skill_invocation": effect_shadow.get("named_skill_invocation"),
        "skill_provenance_assessment": effect_shadow.get("skill_provenance_assessment"),
        "skillbox_status": str(effect_shadow.get("skillbox_status", "disabled")),
        "skillbox_verification": effect_shadow.get("skillbox_verification"),
        "skillbox_ledger_hit": bool(effect_shadow.get("skillbox_ledger_hit", False)),
        "skillbox_content_sha256": effect_shadow.get("skillbox_content_sha256"),
        "skillbox_capabilities": list(effect_shadow.get("skillbox_capabilities", []) or []),
        "skillbox_gate_decision": str(effect_shadow.get("skillbox_gate_decision", "disabled")),
        "artifact_assessment_summary": dict(
            artifact_integrity.get("artifact_assessment_summary", {})
        ),
        "operation_gate": {
            "events": list(operation_gate_events),
            "summary": {"event_count": int(len(operation_gate_events))},
        },
        "evidence": evidence_summary,
        "policy_trace": {
            "trace_id": trace_id,
            "decision_id": decision_id,
            "control_outcome": control_outcome,
            "intended_control_outcome": str(intended_control_outcome),
            "actual_control_outcome": str(control_outcome),
            "content_verdict": str(content_verdict),
            "content_control_outcome": str(content_control_outcome),
            "content_action_types": list(content_action_types),
            "content_walls_triggered": list(content_walls_triggered),
            "content_off": bool(content_off),
            "content_severity": str(content_severity),
            "content_state_scope": "current_request_only",
            "content_state_recomputed": bool(content_state_recomputed),
            "effective_verdict": str(effective_verdict),
            "effective_control_outcome": str(effective_control_outcome),
            "blocked_by_carryover": bool(blocked_by_carryover),
            "carryover_action_types": list(carryover_action_types),
            "off": off,
            "severity": severity,
            "walls_triggered": walls_triggered,
            "action_types": action_types,
            "intended_action_types": list(intended_action_types),
            "max_p": max_p,
            "sum_m_next": sum_m_next,
            "top_docs_count": int(len(step_result.top_docs)),
            "runtime_mode": runtime_mode,
            "state_step_prev": int(state_step_prev),
            "state_step_next": int(step_result.step),
            "ingestion_flags": sorted(set(ingestion_flags)),
            "ocr_status": str(getattr(extracted, "ocr_status", "none") or "none"),
            "ocr_provider": getattr(extracted, "ocr_provider", None),
            "ocr_text_chars": int(getattr(extracted, "ocr_text_chars", 0) or 0),
            "ocr_quality": dict(
                getattr(
                    getattr(extracted, "ocr_quality", None), "to_dict", lambda: {}
                )()
            ),
            "ocr_layout": dict(ocr_layout_summary),
            "visual_status": str(getattr(extracted, "visual_status", "none") or "none"),
            "visual_asset_count": int(len(visual_variants)),
            "visual_asset_manifest": list(visual_manifest),
            "data_region": str(parsed.get("data_region") or "unspecified"),
            "semantic_failure_status": str(phase_state.semantic_failure_status),
            "semantic_failure_policy": str(phase_state.semantic_failure_policy),
            "semantic_failure_policy_branch": str(phase_state.semantic_policy_branch),
            "effect_wall_candidate": effect_shadow.get("effect_wall_candidate"),
            "effect_policy_gate": effect_shadow.get("effect_policy_gate"),
            "effect_policy_gate_status": str(
                effect_shadow.get("effect_policy_gate_status", "disabled")
            ),
            "effect_forecast": effect_shadow.get("effect_forecast"),
            "effect_forecast_status": str(
                effect_shadow.get("effect_forecast_status", "disabled")
            ),
            "named_skill_invocation": effect_shadow.get("named_skill_invocation"),
            "skill_provenance_assessment": effect_shadow.get("skill_provenance_assessment"),
            "skillbox_status": str(effect_shadow.get("skillbox_status", "disabled")),
            "skillbox_verification": effect_shadow.get("skillbox_verification"),
            "skillbox_ledger_hit": bool(effect_shadow.get("skillbox_ledger_hit", False)),
            "skillbox_content_sha256": effect_shadow.get("skillbox_content_sha256"),
            "skillbox_capabilities": list(effect_shadow.get("skillbox_capabilities", []) or []),
            "skillbox_gate_decision": str(effect_shadow.get("skillbox_gate_decision", "disabled")),
            "artifact_assessment_summary": dict(
                artifact_integrity.get("artifact_assessment_summary", {})
            ),
            "operation_gate": {
                "events": list(operation_gate_events),
                "summary": {"event_count": int(len(operation_gate_events))},
            },
            "vision_attempted": bool((api_status or {}).get("vision_attempted", False)),
            "vision_provider_supported": bool(
                (api_status or {}).get("vision_provider_supported", False)
            ),
            "vision_failure_policy": str(
                (api_status or {}).get("vision_failure_policy", "none") or "none"
            ),
            "vision_fallback_used": bool(
                (api_status or {}).get("vision_fallback_used", False)
            ),
            "vision_semantic_status": str(
                (api_status or {}).get("vision_semantic_status", "none") or "none"
            ),
            "semantic_input_kind": str(
                (api_status or {}).get("semantic_input_kind", "text_only")
                or "text_only"
            ),
            "provider": str((api_status or {}).get("provider", "") or ""),
            "provider_id": str((api_status or {}).get("provider_id", "") or ""),
            "provider_capabilities": dict(
                (api_status or {}).get("provider_capabilities", {}) or {}
            ),
            "provider_route": list((api_status or {}).get("provider_route", []) or []),
            "visual_egress_decision": str(
                (api_status or {}).get("visual_egress_decision", "not_evaluated")
            ),
            "visual_egress_reason": str(
                (api_status or {}).get("visual_egress_reason", "none")
            ),
            "provider_processing_region": str(
                (api_status or {}).get("provider_processing_region", "")
            ),
            "trace_source": str(
                (api_status or {}).get("trace_source", "projector_status")
                or "projector_status"
            ),
            "projection_trace_count": int(
                (api_status or {}).get("projection_trace_count", 0) or 0
            ),
            "image_region_pass_enabled": bool(
                (api_status or {}).get("image_region_pass_enabled", False)
            ),
            "provider_call_count": int(
                (api_status or {}).get("provider_call_count", 0) or 0
            ),
            "retry_count": int((api_status or {}).get("retry_count", 0) or 0),
            "cache_hit_last_request": bool(
                (api_status or {}).get("cache_hit_last_request", False)
            ),
            "semantic_latency_ms": (
                (api_status or {}).get("semantic_latency_ms", None)
            ),
            "first_pass_latency_ms": (
                (api_status or {}).get("first_pass_latency_ms", None)
            ),
            "second_pass_latency_ms": (
                (api_status or {}).get("second_pass_latency_ms", None)
            ),
            "second_pass_attempted": bool(
                (api_status or {}).get("second_pass_attempted", False)
            ),
            "second_pass_result": str(
                (api_status or {}).get("second_pass_result", "not_attempted")
                or "not_attempted"
            ),
            "region_trigger_reason": str(
                (api_status or {}).get("region_trigger_reason", "none") or "none"
            ),
            "region_variant_count": int(
                (api_status or {}).get("region_variant_count", 0) or 0
            ),
            "token_usage": dict((api_status or {}).get("token_usage", {}) or {}),
            "ocr_modality_present": bool(
                str(getattr(extracted, "ocr_status", "none") or "none") == "success"
            ),
            "ocr_active_walls": list(ocr_active_walls),
            "vision_active_walls": list(image_active_walls),
            "active_modalities": list(active_modalities),
            "modality_positive_chunk_counts": dict(modality_positive_chunk_counts),
            "ocr_max_chunk_score_band": str(ocr_max_chunk_score_band),
            "vision_max_chunk_score_band": str(vision_max_chunk_score_band),
            "modality_wall_max": dict(modality_wall_max),
            "ocr_vision_agreement": bool(ocr_vision_agreement),
            "ocr_gate_applied": bool(ocr_gate_applied),
            "ocr_gate_reason": str(ocr_gate_reason),
            "ocr_adjudication_status": str(ocr_adjudication_status),
            "ocr_adjudication_result": str(ocr_adjudication_result),
            "ocr_adjudication": dict(ocr_adjudication_trace),
            "ocr_only_provisional_evidence": bool(ocr_only_pressure),
            "hallucination_guard_lite": dict(hallucination_guard_summary),
            "response_constraints": dict(response_constraints),
            "chunk_pipeline": {
                "chunks_total": int(len(items)),
                "pressure_dedupe": dict(pressure_dedupe),
                "worst_chunk_score": float(chunk_agg.worst_chunk_score),
                "pattern_synergy": float(chunk_agg.pattern_synergy),
                "confidence": float(chunk_agg.confidence),
                "doc_score": float(chunk_agg.doc_score),
                "pair_hits": list(chunk_agg.pair_hits),
                "wall_max": dict(chunk_agg.wall_max),
                "top_chunks": list(trace_top_chunks),
            },
            "evidence": evidence_summary,
        },
        "response_constraints": dict(response_constraints),
    }
    monitor_attribution = _monitor_attribution_rows(
        items=items, top_chunks=list(chunk_agg.top_chunks)
    )
    monitor_fragments = build_redacted_fragments(
        attribution_rows=monitor_attribution,
        item_text_by_doc={str(item.doc_id): str(item.text) for item in items},
        item_meta_by_doc={
            str(item.doc_id): (
                dict(item.meta) if isinstance(item.meta, Mapping) else {}
            )
            for item in items
        },
        max_fragments=4,
        max_chars=240,
    )
    intended_blocked_doc_ids: List[str] = []
    if str(intended_verdict).lower() in {"block", "quarantine"}:
        intended_blocked_doc_ids = [
            str(chunk.get("doc_id", "")).strip()
            for chunk in list(chunk_agg.top_chunks)
            if str(chunk.get("doc_id", "")).strip()
        ]
    intended_quarantined_sources: List[str] = []
    if "SOURCE_QUARANTINE" in set(intended_action_types) or bool(
        source_quarantine_active
    ):
        intended_quarantined_sources = [str(source_id)]
    intended_prevented_tools: List[str] = (
        ["*"] if "TOOL_FREEZE" in set(intended_action_types) else []
    )
    monitor_downstream = build_downstream_summary(
        intended_action=str(intended_control_outcome),
        action_types=list(intended_action_types),
        blocked_doc_ids=intended_blocked_doc_ids,
        quarantined_source_ids=intended_quarantined_sources,
        prevented_tools=intended_prevented_tools,
    )
    monitor_rules = {
        "triggered_rules": list(walls_triggered),
        "reason_codes": list(reasons_sorted),
    }
    fp_hint = infer_false_positive_hint(
        risk_score=float(risk_score) / 100.0,
        intended_action=str(intended_control_outcome),
        reason_codes=list(reasons_sorted),
        triggered_rules=list(walls_triggered),
        attribution=monitor_attribution,
        config=runtime.config,
    )
    monitor_payload = {
        "enabled": bool(monitor_enabled),
        "guard_mode": str(guard_mode.value).lower(),
        "intended_action": str(intended_control_outcome),
        "actual_action": str(control_outcome),
        "triggered_rules": list(walls_triggered),
        "rules": monitor_rules,
        "fragments": monitor_fragments,
        "downstream": monitor_downstream,
        "false_positive_hint": fp_hint,
        "hallucination_guard_lite": dict(hallucination_guard_summary),
        "response_constraints": dict(response_constraints),
    }
    payload["monitor"] = monitor_payload
    collector = runtime.monitor_collector
    if collector is not None and bool(collector.enabled):
        collector.emit(
            MonitorEvent(
                ts=utc_now_iso(),
                surface="api",
                session_id=str(session_id or request_id),
                actor_id=str(actor_id or session_id or request_id),
                mode=str(guard_mode.value).lower(),
                risk_score=float(risk_score) / 100.0,
                intended_action=str(intended_control_outcome),
                actual_action=str(control_outcome),
                triggered_rules=list(walls_triggered),
                attribution=list(monitor_attribution),
                reason_codes=list(reasons_sorted),
                rules=monitor_rules,
                fragments=monitor_fragments,
                downstream=monitor_downstream,
                trace_id=str(trace_id),
                decision_id=str(decision_id),
                false_positive_hint=(str(fp_hint) if fp_hint else None),
                metadata={
                    "tenant_id": str(tenant_id),
                    "request_id": str(request_id),
                    "runtime_mode": str(runtime_mode),
                    "verdict": str(verdict),
                    "hallucination_guard_lite": dict(hallucination_guard_summary),
                    "response_constraints": dict(response_constraints),
                },
            )
        )
    payload["monitoring_metrics"] = (
        collector.health_snapshot() if collector is not None else {"enabled": False}
    )
    if runtime_mode == "stateful" and session_id:
        payload["session_id"] = str(session_id)
        payload["policy_trace"]["session_id"] = str(session_id)
        payload["policy_trace"]["cross_session"] = {
            "carryover_applied": bool(cross_carryover_applied),
            "active_action_types": list(cross_active_action_types),
            "actor_hash": cross_actor_hash,
        }
    api_config_refs = _config_refs(runtime, _sha256_hex)
    if bool(step_result.off):
        payload["omega_off_event"] = build_off_event(
            step_result=step_result,
            decision=decision,
            items=items,
            config_refs=api_config_refs,
            thresholds=_omega_thresholds(runtime),
            capture_text="NEVER",
            trace_id=trace_id,
            decision_id=decision_id,
        )
    if should_emit_incident_artifact(
        config=runtime.config, control_outcome=control_outcome
    ):
        capture_incident_text = should_capture_incident_text(config=runtime.config)
        item_by_id = {item.doc_id: item for item in items}
        top_chunk_ids = [
            str(x)
            for x in list(evidence_summary.get("top_chunk_ids", []) or [])
            if str(x).strip()
        ]
        if not top_chunk_ids:
            top_chunk_ids = [
                str(x)
                for x in list(evidence_summary.get("chunk_ids", []) or [])
                if str(x).strip()
            ]
        top_docs = []
        for doc_id in top_chunk_ids:
            item = item_by_id.get(doc_id)
            if item is None:
                continue
            top_docs.append(
                {
                    "doc_id": item.doc_id,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "trust": item.trust,
                    "text_sha256": _sha256_hex(str(item.text)),
                    **({"text": str(item.text)} if capture_incident_text else {}),
                }
            )
        blocked_doc_ids: List[str] = []
        if verdict in {"block", "quarantine"}:
            blocked_doc_ids = [
                doc_id for doc_id in top_chunk_ids if doc_id in item_by_id
            ]
        quarantined_source_ids = sorted(
            {
                str(source_item)
                for action in cross_active_actions
                if str(getattr(action, "type", "")) == "SOURCE_QUARANTINE"
                for source_item in list(getattr(action, "source_ids", []) or [])
                if str(source_item).strip()
            }
        )
        if verdict == "quarantine":
            quarantined_source_ids = sorted(
                set(quarantined_source_ids) | {str(source_id)}
            )
        incident_artifact = build_incident_artifact(
            config=runtime.config,
            surface="api",
            session_id=str(session_id) if session_id else f"req:{request_id}",
            step=int(step_result.step),
            request_id=request_id,
            control_outcome=control_outcome,
            off=off,
            severity=severity,
            verdict=verdict,
            actions=list(decision.actions) + list(cross_active_actions),
            reason_flags=reasons_sorted,
            contributing_signals={
                "max_p": max_p,
                "sum_m_next": sum_m_next,
                "walls_triggered": list(walls_triggered),
                "chunk_pipeline": {
                    "doc_score": float(chunk_agg.doc_score),
                    "worst_chunk_score": float(chunk_agg.worst_chunk_score),
                    "pattern_synergy": float(chunk_agg.pattern_synergy),
                    "confidence": float(chunk_agg.confidence),
                    "wall_max": dict(chunk_agg.wall_max),
                },
            },
            top_docs=top_docs,
            blocked_doc_ids=blocked_doc_ids,
            quarantined_source_ids=quarantined_source_ids,
            context_total_docs=len(items),
            context_allowed_docs=max(0, len(items) - len(set(blocked_doc_ids))),
            evidence_id=evidence_id,
            config_refs=api_config_refs,
            refs={
                "source_id": str(source_id),
                "cross_active_action_types": list(cross_active_action_types),
                "cross_actor_hash": cross_actor_hash,
            },
            trace_id=trace_id,
            decision_id=decision_id,
        )
        payload["incident_artifact_id"] = str(
            incident_artifact.get("incident_artifact_id", "")
        )
        payload["incident_artifact"] = incident_artifact

    if _incident_export_enabled(runtime):
        store = runtime.incident_export_store
        if store is None:
            if _is_enterprise_production(runtime):
                raise HTTPException(status_code=503, detail="incident_export_unavailable")
        else:
            try:
                incident_record = build_incident_record_from_scan(
                    payload=payload,
                    parsed=parsed,
                    environment=(
                        str(
                            (
                                runtime.incident_export_cfg.default_env
                                if runtime.incident_export_cfg is not None
                                else "staging"
                            )
                        )
                    ),
                    runtime_mode=str(runtime_mode),
                    capture_incident_text=should_capture_incident_text(
                        config=runtime.config
                    ),
                )
                payload["incident_export_id"] = str(
                    incident_record.get("incident_id", "")
                )
                store.insert_record(incident_record)
            except Exception as exc:  # noqa: BLE001
                if _is_enterprise_production(runtime):
                    raise HTTPException(status_code=503, detail="incident_export_write_failed") from exc
                LOGGER.warning("incident_export_store_write_failed: %s", exc)

    dispatcher = runtime.notification_dispatcher
    notifications_enabled = bool(_notifications_cfg(runtime).get("enabled", False))
    if notifications_enabled and dispatcher is not None:
        semantic_fallback = not bool(
            getattr(runtime.projector, "semantic_active", True)
        )
        api_fallback = bool((api_status or {}).get("llm_fallback_active", False))
        fallback_active = bool(semantic_fallback or api_fallback)
        risk_event = _build_api_risk_event(
            payload=payload, parsed=parsed, fallback_active=fallback_active
        )
        dispatcher.emit_risk_event(risk_event)
        action_types = [
            str(x)
            for x in list(
                (
                    ((payload.get("policy_trace", {}) or {}).get("action_types", []))
                    or []
                )
            )
        ]
        approval_required = ("HUMAN_ESCALATE" in action_types) or (
            "REQUIRE_APPROVAL" in action_types
        )
        approval_id: Optional[str] = None
        approval_status = "none"
        session_ref = str(parsed.get("session_id") or payload.get("request_id", ""))
        existing = dispatcher.latest_approval_for_session(
            tenant_id=str(parsed.get("tenant_id", "")),
            session_id=session_ref,
        )
        if existing is not None:
            approval_id = str(existing.approval_id)
            approval_status = str(existing.status)
        if approval_required:
            timeout_sec = int(
                (
                    (_notifications_cfg(runtime).get("approvals", {}) or {}).get(
                        "timeout_sec", 900
                    )
                )
            )
            approval = dispatcher.create_action_request(
                risk_event=risk_event,
                required_action="HUMAN_ESCALATE"
                if "HUMAN_ESCALATE" in action_types
                else "REQUIRE_APPROVAL",
                timeout_sec=max(10, timeout_sec),
            )
            approval_id = str(approval.approval_id)
            approval_status = str(approval.status)
        payload["approval_required"] = bool(approval_required)
        if approval_id:
            payload["approval_id"] = str(approval_id)
            payload["approval_status"] = str(approval_status)
        payload["notification_metrics"] = dispatcher.metrics_snapshot()

    telemetry = getattr(runtime, "telemetry_service", None)
    if telemetry is not None:
        fallback_active = (
            not bool(getattr(runtime.projector, "semantic_active", True))
        ) or bool((api_status or {}).get("llm_fallback_active", False))
        fallback_level = str((api_status or {}).get("fallback_level", "none") or "none")
        orchestrator_cfg = (
            (runtime.config.get("projector", {}) or {}).get("api_perception", {}) or {}
        ).get("orchestrator", {}) or {}
        module_flags = {
            "orchestrator": bool(orchestrator_cfg.get("enabled", False)),
            "incident_export": bool(_incident_export_enabled(runtime)),
            "incident_replay": bool(_incident_replay_enabled(runtime)),
            "monitoring": bool(
                (runtime.config.get("monitoring", {}) or {}).get("enabled", False)
            ),
            "notifications": bool(notifications_enabled),
        }
        telemetry.emit_event(
            build_telemetry_event(
                surface="api",
                control_outcome=str(control_outcome),
                severity=str(severity),
                walls_triggered=list(walls_triggered),
                reason_codes=list(reasons_sorted),
                action_types=list(action_types),
                risk_score=float(risk_score) / 100.0,
                fallback_active=bool(fallback_active),
                fallback_level=str(fallback_level),
                accumulation_steps=int(step_result.step),
                provenance_type=str(source_type),
                module_flags=module_flags,
                fp_reported=False,
            )
        )

    att, att_reason = _attestation_block(
        response_wo_attestation=payload, runtime=runtime
    )
    if att is not None:
        payload["attestation"] = att
    elif att_reason:
        reasons_sorted = sorted(set(reasons_sorted + [att_reason]))
        payload["reasons"] = reasons_sorted
        payload["policy_trace"]["attestation_status"] = att_reason
    if include_document_scan_report:
        payload["document_scan_report"] = _build_document_scan_report(
            chunk_agg=chunk_agg,
            fmt=fmt,
            ingestion_flags=ingestion_flags,
            max_chunks=runtime.debug.max_report_chunks,
        )
    if (
        runtime_mode == "stateful"
        and session_id
        and actor_id
        and session_store is not None
    ):
        session_store.save_state_and_cached_response(
            tenant_id=tenant_id,
            session_id=session_id,
            actor_id=actor_id,
            m=np.asarray(step_result.m_next, dtype=float),
            step=int(step_result.step),
            request_id=request_id,
            response_payload=payload,
        )
    release_scope = getattr(runtime.projector, "release_image_scope", None)
    if callable(release_scope):
        release_scope(str(request_id))
    return payload
