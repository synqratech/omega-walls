"""Request payload parsers for API routes."""

from __future__ import annotations

import base64
import re
import uuid
from typing import Any, Dict

from fastapi import HTTPException, Request

from omega.api.runtime_factory import ApiLimits

_DATA_REGION_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")


async def _parse_request_payload(request: Request, limits: ApiLimits) -> Dict[str, Any]:
    ctype = str(request.headers.get("content-type", "")).lower()
    payload: Dict[str, Any] = {
        "tenant_id": None,
        "request_id": None,
        "session_id": None,
        "actor_id": None,
        "runtime_mode": None,
        "data_region": None,
        "filename": None,
        "mime": None,
        "file_bytes": None,
        "extracted_text": None,
        "input_mode": None,
        "request_id_provided": False,
    }

    if "application/json" in ctype:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid_json_body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid_json_body")
        payload["tenant_id"] = body.get("tenant_id")
        payload["request_id"] = body.get("request_id")
        payload["session_id"] = body.get("session_id")
        payload["actor_id"] = body.get("actor_id")
        payload["runtime_mode"] = body.get("runtime_mode")
        payload["data_region"] = body.get("data_region")
        payload["filename"] = body.get("filename")
        payload["mime"] = body.get("mime")
        if body.get("extracted_text") is not None:
            payload["extracted_text"] = str(body.get("extracted_text"))
            payload["input_mode"] = "extracted_text"
        if body.get("file_base64") is not None:
            try:
                file_bytes = base64.b64decode(
                    str(body.get("file_base64")), validate=True
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail="invalid_file_base64"
                ) from exc
            if len(file_bytes) > limits.max_file_bytes:
                raise HTTPException(status_code=413, detail="file_too_large")
            payload["file_bytes"] = file_bytes
            if payload["input_mode"] is None:
                payload["input_mode"] = "file_base64"
    elif "multipart/form-data" in ctype:
        try:
            form = await request.form(
                max_files=limits.max_multipart_files,
                max_fields=limits.max_multipart_fields,
                max_part_size=limits.max_multipart_part_bytes,
            )
        except TypeError:  # compatibility with older Starlette
            form = await request.form(
                max_files=limits.max_multipart_files,
                max_fields=limits.max_multipart_fields,
            )
        payload["tenant_id"] = form.get("tenant_id")
        payload["request_id"] = form.get("request_id")
        payload["session_id"] = form.get("session_id")
        payload["actor_id"] = form.get("actor_id")
        payload["runtime_mode"] = form.get("runtime_mode")
        payload["data_region"] = form.get("data_region")
        payload["filename"] = form.get("filename")
        payload["mime"] = form.get("mime")
        extracted_text = form.get("extracted_text")
        if extracted_text is not None:
            payload["extracted_text"] = str(extracted_text)
            payload["input_mode"] = "extracted_text"
        upload = form.get("file")
        if upload is not None:
            try:
                file_bytes = await upload.read()
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail="invalid_multipart_file"
                ) from exc
            if len(file_bytes) > limits.max_file_bytes:
                raise HTTPException(status_code=413, detail="file_too_large")
            payload["file_bytes"] = file_bytes
            if payload["filename"] in (None, ""):
                payload["filename"] = getattr(upload, "filename", None)
            if payload["mime"] in (None, ""):
                payload["mime"] = getattr(upload, "content_type", None)
            if payload["input_mode"] is None:
                payload["input_mode"] = "file_multipart"
    else:
        raise HTTPException(status_code=415, detail="unsupported_content_type")

    tenant_id = str(payload.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id_required")
    payload["tenant_id"] = tenant_id

    request_id = str(payload.get("request_id") or "").strip() or str(uuid.uuid4())
    payload["request_id_provided"] = bool(str(payload.get("request_id") or "").strip())
    payload["request_id"] = request_id

    session_id = str(payload.get("session_id") or "").strip()
    payload["session_id"] = session_id or None
    actor_id = str(payload.get("actor_id") or "").strip()
    payload["actor_id"] = actor_id or None
    runtime_mode = str(payload.get("runtime_mode") or "").strip().lower()
    if runtime_mode:
        if runtime_mode not in {"stateless", "stateful"}:
            raise HTTPException(status_code=400, detail="invalid_runtime_mode")
        payload["runtime_mode"] = runtime_mode
    else:
        payload["runtime_mode"] = None

    data_region = (
        str(payload.get("data_region") or "unspecified").strip().lower()
        or "unspecified"
    )
    if len(data_region) > 64 or not _DATA_REGION_RE.fullmatch(data_region):
        raise HTTPException(status_code=400, detail="invalid_data_region")
    payload["data_region"] = data_region

    extracted_text = payload.get("extracted_text")
    if extracted_text is not None:
        extracted_text = str(extracted_text)
        if len(extracted_text) > limits.max_extracted_text_chars:
            raise HTTPException(status_code=413, detail="extracted_text_too_large")
        payload["extracted_text"] = extracted_text

    has_extracted = bool(str(payload.get("extracted_text") or "").strip())
    has_file = payload.get("file_bytes") is not None
    if not has_extracted and not has_file:
        raise HTTPException(status_code=400, detail="missing_payload")

    payload["use_extracted_text"] = has_extracted
    return payload


async def _parse_session_reset_payload(request: Request) -> Dict[str, Any]:
    ctype = str(request.headers.get("content-type", "")).lower()
    if "application/json" not in ctype:
        raise HTTPException(status_code=415, detail="unsupported_content_type")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_json_body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid_json_body")

    tenant_id = str(body.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id_required")
    request_id = str(body.get("request_id") or "").strip() or str(uuid.uuid4())
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id_required")
    actor_id = str(body.get("actor_id") or "").strip() or None
    return {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "session_id": session_id,
        "actor_id": actor_id,
    }
