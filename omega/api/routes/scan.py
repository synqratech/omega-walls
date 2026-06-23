"""Scan routes extracted from api.server with callback-based wiring."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from omega.api.openapi_models import (
    ATTACHMENT_SCAN_REQUEST_BODY,
    AttachmentScanResponse,
    ScanErrorResponse,
)
from omega.api.runtime_factory import ScanRuntime


ATTACHMENT_SCAN_RESPONSES = {
    400: {"model": ScanErrorResponse, "description": "Malformed body, invalid base64, missing payload, or invalid runtime mode."},
    401: {"model": ScanErrorResponse, "description": "Unauthorized, invalid HMAC signature, or replay-protection failure."},
    403: {"model": ScanErrorResponse, "description": "Debug scan report is disabled."},
    409: {"model": ScanErrorResponse, "description": "Replay nonce or request id conflict."},
    413: {"model": ScanErrorResponse, "description": "Request body, attachment, multipart part, or extracted text exceeds configured limits."},
    415: {"model": ScanErrorResponse, "description": "Unsupported content type, attachment type, MIME/magic mismatch, or unsupported image format."},
    422: {"model": ScanErrorResponse, "description": "Request validation failure."},
    429: {"model": ScanErrorResponse, "description": "Rate limit exceeded."},
    503: {"model": ScanErrorResponse, "description": "Semantic vision/provider unavailable and configured failure policy requires escalation or fail-closed handling."},
}


def build_scan_router(
    *,
    enforce_transport_security: Callable[..., None],
    valid_api_key: Callable[[str, Any], bool],
    parse_request_payload: Callable[..., Awaitable[Dict[str, Any]]],
    verify_hmac_request: Callable[..., None],
    effective_runtime_mode: Callable[..., str],
    scan_request: Callable[..., Dict[str, Any]],
    audit_log_api_response: Callable[..., None],
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/scan/attachment",
        response_model=AttachmentScanResponse,
        responses=ATTACHMENT_SCAN_RESPONSES,
        openapi_extra={"requestBody": ATTACHMENT_SCAN_REQUEST_BODY},
        summary="Scan a text, document, or image attachment",
        description=(
            "Scans attachments through Omega rule, semantic, and multimodal controls. "
            "Images use the same endpoint; provider-dependent vision unavailability is explicit "
            "in policy_trace and follows the configured semantic failure policy rather than silently allowing content."
        ),
    )
    async def scan_attachment(
        request: Request,
        debug: bool = False,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key", description="Omega API key."),
        x_signature: Optional[str] = Header(default=None, alias="X-Signature", description="HMAC signature when enabled."),
        x_timestamp: Optional[str] = Header(default=None, alias="X-Timestamp", description="Unix timestamp for HMAC authentication."),
        x_nonce: Optional[str] = Header(default=None, alias="X-Nonce", description="Single-use nonce for replay protection."),
    ) -> Dict[str, Any]:
        _ = (x_signature, x_timestamp, x_nonce)
        runtime: ScanRuntime = request.app.state.scan_runtime
        enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await parse_request_payload(request=request, limits=runtime.limits)
        verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        if debug and not runtime.debug.enable_document_scan_report:
            raise HTTPException(status_code=403, detail="debug_mode_disabled")
        mode = effective_runtime_mode(runtime, parsed)
        if mode == "stateful":
            lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
            async with lock:
                payload = scan_request(
                    runtime=runtime,
                    parsed=parsed,
                    include_document_scan_report=bool(debug),
                )
        else:
            payload = scan_request(
                runtime=runtime,
                parsed=parsed,
                include_document_scan_report=bool(debug),
            )
        audit_log_api_response(
            runtime=runtime,
            request=request,
            parsed=parsed,
            body_bytes=body_bytes,
            response_payload=payload,
        )
        return payload

    @router.post(
        "/v1/scan/attachment/document_scan_report",
        response_model=AttachmentScanResponse,
        responses=ATTACHMENT_SCAN_RESPONSES,
        openapi_extra={"requestBody": ATTACHMENT_SCAN_REQUEST_BODY},
        include_in_schema=False,
    )
    async def scan_attachment_document_report(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        x_signature: Optional[str] = Header(default=None, alias="X-Signature"),
        x_timestamp: Optional[str] = Header(default=None, alias="X-Timestamp"),
        x_nonce: Optional[str] = Header(default=None, alias="X-Nonce"),
    ) -> Dict[str, Any]:
        _ = (x_signature, x_timestamp, x_nonce)
        runtime: ScanRuntime = request.app.state.scan_runtime
        if not runtime.debug.enable_document_scan_report:
            raise HTTPException(status_code=403, detail="debug_mode_disabled")
        enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await parse_request_payload(request=request, limits=runtime.limits)
        verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        mode = effective_runtime_mode(runtime, parsed)
        if mode == "stateful":
            lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
            async with lock:
                payload = scan_request(runtime=runtime, parsed=parsed, include_document_scan_report=True)
        else:
            payload = scan_request(runtime=runtime, parsed=parsed, include_document_scan_report=True)
        audit_log_api_response(
            runtime=runtime,
            request=request,
            parsed=parsed,
            body_bytes=body_bytes,
            response_payload=payload,
        )
        return payload

    return router

