"""Session management routes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from omega.api.auth import _enforce_transport_security, _valid_api_key, _verify_hmac_request
from omega.api.request_parsing import _parse_session_reset_payload
from omega.api.response_builder import ProblemDetailResponse
from omega.api.runtime_factory import ScanRuntime


class SessionResetRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    request_id: str = Field(
        min_length=1,
        description="Required for HMAC-authenticated requests; echoed back in the reset response.",
    )
    session_id: str = Field(min_length=1)
    actor_id: Optional[str] = None


class SessionResetResponse(BaseModel):
    request_id: str
    tenant_id: str
    session_id: str
    reset: bool
    existed: bool
    approvals_cleared: bool


SESSION_RESET_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": SessionResetRequest.model_json_schema(),
            }
        },
    }
}


SESSION_RESET_RESPONSES = {
    400: {"model": ProblemDetailResponse, "description": "Invalid JSON body or required session fields missing."},
    401: {"model": ProblemDetailResponse, "description": "Unauthorized or invalid HMAC signature."},
    409: {"model": ProblemDetailResponse, "description": "Replay nonce detected for an HMAC-authenticated reset request."},
    415: {"model": ProblemDetailResponse, "description": "Unsupported content type; only application/json is accepted."},
    503: {"model": ProblemDetailResponse, "description": "Stateful session runtime is not configured."},
}


def build_session_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/session/reset",
        response_model=SessionResetResponse,
        responses=SESSION_RESET_RESPONSES,
        openapi_extra=SESSION_RESET_OPENAPI,
    )
    async def reset_session(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        x_signature: Optional[str] = Header(
            default=None,
            alias="X-Signature",
            description="Required when API HMAC authentication is enabled.",
        ),
        x_timestamp: Optional[str] = Header(
            default=None,
            alias="X-Timestamp",
            description="Unix timestamp header required when API HMAC authentication is enabled.",
        ),
        x_nonce: Optional[str] = Header(
            default=None,
            alias="X-Nonce",
            description="Replay-protection nonce required when API HMAC authentication is enabled.",
        ),
    ) -> Dict[str, Any]:
        _ = (x_signature, x_timestamp, x_nonce)
        runtime: ScanRuntime = request.app.state.scan_runtime
        _enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not _valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await _parse_session_reset_payload(request=request)
        _verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        if runtime.session_store is None:
            raise HTTPException(status_code=503, detail="stateful_runtime_not_configured")
        lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
        async with lock:
            existed = runtime.session_store.clear_session(
                tenant_id=str(parsed["tenant_id"]),
                session_id=str(parsed["session_id"]),
            )
            approvals_cleared = False
            if runtime.notification_dispatcher is not None:
                approvals_cleared = runtime.notification_dispatcher.store.clear_session(
                    tenant_id=str(parsed["tenant_id"]),
                    session_id=str(parsed["session_id"]),
                )
        return {
            "request_id": str(parsed["request_id"]),
            "tenant_id": str(parsed["tenant_id"]),
            "session_id": str(parsed["session_id"]),
            "reset": True,
            "existed": bool(existed),
            "approvals_cleared": bool(approvals_cleared),
        }

    return router
