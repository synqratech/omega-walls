"""Notification callback and approval routes."""

from __future__ import annotations

import json
import os
from omega.runtime.environment import read_secret_value
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, Header, HTTPException, Request

from omega.api.runtime_factory import ScanRuntime
from omega.notifications.models import ApprovalDecision


def build_notifications_router(
    *,
    notifications_cfg: Callable[[ScanRuntime], Dict[str, Any]],
    approval_internal_auth_cfg: Callable[[ScanRuntime], Dict[str, Any]],
    valid_api_key: Callable[[str, Any], bool],
    verify_slack_signature: Callable[..., bool],
    verify_telegram_secret_token: Callable[..., bool],
    verify_internal_hmac: Callable[..., bool],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/notifications/callback/slack")
    async def slack_callback(request: Request) -> Dict[str, Any]:
        runtime: ScanRuntime = request.app.state.scan_runtime
        cfg = notifications_cfg(runtime)
        dispatcher = runtime.notification_dispatcher
        if not bool(cfg.get("enabled", False)) or dispatcher is None:
            raise HTTPException(status_code=503, detail="notifications_not_enabled")
        slack_cfg = cfg.get("slack", {}) if isinstance(cfg.get("slack", {}), dict) else {}
        if not bool(slack_cfg.get("enabled", False)):
            raise HTTPException(status_code=503, detail="slack_not_enabled")

        signing_secret_env = str(slack_cfg.get("signing_secret_env", "SLACK_SIGNING_SECRET")).strip()
        signing_secret = str(os.environ.get(signing_secret_env, "")).strip()
        if not signing_secret:
            raise HTTPException(status_code=503, detail="slack_signing_secret_missing")
        body_bytes = await request.body()
        sig = str(request.headers.get("X-Slack-Signature", "")).strip()
        ts = str(request.headers.get("X-Slack-Request-Timestamp", "")).strip()
        if not verify_slack_signature(
            body_bytes=body_bytes,
            signature=sig,
            timestamp=ts,
            signing_secret=signing_secret,
        ):
            raise HTTPException(status_code=401, detail="invalid_slack_signature")

        ctype = str(request.headers.get("content-type", "")).lower()
        payload_obj: Dict[str, Any] = {}
        if "application/x-www-form-urlencoded" in ctype:
            form = parse_qs(body_bytes.decode("utf-8", errors="replace"), keep_blank_values=True)
            payload_raw = str((form.get("payload") or [""])[0])
            if payload_raw.strip():
                try:
                    parsed = json.loads(payload_raw)
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=400, detail="invalid_callback_payload") from exc
                if isinstance(parsed, dict):
                    payload_obj = parsed
        else:
            try:
                parsed = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="invalid_callback_payload") from exc
            if isinstance(parsed, dict):
                payload_obj = parsed

        if "challenge" in payload_obj:
            return {"challenge": payload_obj.get("challenge")}

        actions = payload_obj.get("actions", []) if isinstance(payload_obj.get("actions", []), list) else []
        if not actions:
            return {"ok": True, "ignored": True}
        action = actions[0] if isinstance(actions[0], dict) else {}
        value_raw = str(action.get("value", "")).strip()
        if not value_raw:
            raise HTTPException(status_code=400, detail="missing_action_value")
        try:
            value_obj = json.loads(value_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid_action_value") from exc
        if not isinstance(value_obj, dict):
            raise HTTPException(status_code=400, detail="invalid_action_value")
        approval_id = str(value_obj.get("approval_id", "")).strip()
        decision = str(value_obj.get("decision", "")).strip().lower()
        if decision not in {"approved", "denied"}:
            raise HTTPException(status_code=400, detail="invalid_action_decision")
        if not approval_id:
            raise HTTPException(status_code=400, detail="missing_approval_id")
        actor_id = str(((payload_obj.get("user", {}) or {}).get("id", ""))).strip()
        record = dispatcher.resolve_approval(
            approval_id=approval_id,
            decision=ApprovalDecision(
                decision=decision,
                actor_id=actor_id,
                source="slack_callback",
            ).normalized(),
        )
        if record is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        return {"ok": True, "approval_id": approval_id, "status": str(record.status)}

    @router.post("/v1/notifications/callback/telegram")
    async def telegram_callback(request: Request) -> Dict[str, Any]:
        runtime: ScanRuntime = request.app.state.scan_runtime
        cfg = notifications_cfg(runtime)
        dispatcher = runtime.notification_dispatcher
        if not bool(cfg.get("enabled", False)) or dispatcher is None:
            raise HTTPException(status_code=503, detail="notifications_not_enabled")
        tg_cfg = cfg.get("telegram", {}) if isinstance(cfg.get("telegram", {}), dict) else {}
        if not bool(tg_cfg.get("enabled", False)):
            raise HTTPException(status_code=503, detail="telegram_not_enabled")
        secret_env = str(tg_cfg.get("secret_token_env", "TG_BOT_SECRET_TOKEN")).strip()
        expected_secret = str(os.environ.get(secret_env, "")).strip()
        if not expected_secret:
            raise HTTPException(status_code=503, detail="telegram_secret_missing")
        provided_secret = str(request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")).strip()
        if not verify_telegram_secret_token(provided=provided_secret, expected=expected_secret):
            raise HTTPException(status_code=401, detail="invalid_telegram_secret")
        try:
            payload_obj = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid_callback_payload") from exc
        if not isinstance(payload_obj, dict):
            raise HTTPException(status_code=400, detail="invalid_callback_payload")
        callback_query = payload_obj.get("callback_query", {}) if isinstance(payload_obj.get("callback_query", {}), dict) else {}
        data = str(callback_query.get("data", "")).strip()
        if not data:
            return {"ok": True, "ignored": True}
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != "omega" or parts[1] not in {"approved", "denied"}:
            raise HTTPException(status_code=400, detail="invalid_callback_data")
        approval_id = str(parts[2]).strip()
        decision = str(parts[1]).strip()
        actor_id = str((((callback_query.get("from", {}) or {}).get("id", "")))).strip()
        record = dispatcher.resolve_approval(
            approval_id=approval_id,
            decision=ApprovalDecision(
                decision=decision,
                actor_id=actor_id,
                source="telegram_callback",
            ).normalized(),
        )
        if record is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        return {"ok": True, "approval_id": approval_id, "status": str(record.status)}

    @router.get("/v1/approvals/{approval_id}")
    async def get_approval(
        request: Request,
        approval_id: str,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    ) -> Dict[str, Any]:
        runtime: ScanRuntime = request.app.state.scan_runtime
        if not x_api_key or not valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        dispatcher = runtime.notification_dispatcher
        if dispatcher is None:
            raise HTTPException(status_code=503, detail="notifications_not_enabled")
        record = dispatcher.get_approval(str(approval_id))
        if record is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        tenant_id = str(x_tenant_id or "").strip()
        profile_env = str(((runtime.config.get("profiles", {}) or {}).get("env", ""))).strip().lower()
        if not tenant_id and profile_env not in {"prod", "production"}:
            tenant_id = str(record.tenant_id)
        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id_required")
        if tenant_id != str(record.tenant_id):
            raise HTTPException(status_code=404, detail="approval_not_found")
        return {"approval": record.to_dict()}

    @router.post("/v1/approvals/{approval_id}/resolve")
    async def resolve_approval(
        approval_id: str,
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        runtime: ScanRuntime = request.app.state.scan_runtime
        if not x_api_key or not valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        dispatcher = runtime.notification_dispatcher
        if dispatcher is None:
            raise HTTPException(status_code=503, detail="notifications_not_enabled")
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid_json_body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid_json_body")
        tenant_id = str(body.get("tenant_id", "")).strip()
        request_id = str(body.get("request_id", "")).strip() or str(uuid.uuid4())
        decision_raw = str(body.get("decision", "")).strip().lower()
        actor_id = str(body.get("actor_id", "")).strip()
        reason = str(body.get("reason", "")).strip()
        source = str(body.get("source", "internal_manual")).strip()
        existing = dispatcher.get_approval(str(approval_id))
        if existing is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id_required")
        if tenant_id != str(existing.tenant_id):
            # Avoid disclosing cross-tenant approval identifiers.
            raise HTTPException(status_code=404, detail="approval_not_found")

        try:
            decision = ApprovalDecision(
                decision=decision_raw,
                actor_id=actor_id,
                source=source,
                reason=reason,
            ).normalized()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid_approval_decision") from exc

        internal_auth = approval_internal_auth_cfg(runtime)
        if bool(internal_auth.get("require_hmac", True)):
            headers_cfg = internal_auth.get("headers", {}) if isinstance(internal_auth.get("headers", {}), dict) else {}
            sig_header = str(headers_cfg.get("signature", "X-Internal-Signature")).strip()
            ts_header = str(headers_cfg.get("timestamp", "X-Internal-Timestamp")).strip()
            nonce_header = str(headers_cfg.get("nonce", "X-Internal-Nonce")).strip()
            signature = str(request.headers.get(sig_header, "")).strip()
            ts = str(request.headers.get(ts_header, "")).strip()
            nonce = str(request.headers.get(nonce_header, "")).strip()
            secret_env = str(internal_auth.get("hmac_secret_env", "OMEGA_NOTIFICATION_HMAC_SECRET")).strip()
            try:
                secret = read_secret_value(secret_env, required=True)
            except ValueError:
                raise HTTPException(status_code=503, detail="notification_hmac_secret_missing")
            max_skew = int(internal_auth.get("max_clock_skew_sec", 300))
            valid = verify_internal_hmac(
                method=request.method,
                path=request.url.path,
                body_bytes=body_bytes,
                tenant_id=tenant_id,
                request_id=request_id,
                signature=signature,
                timestamp=ts,
                nonce=nonce,
                secret=secret,
                seen_nonces=dispatcher.nonce_cache,
                max_skew_sec=max_skew,
            )
            if not valid:
                raise HTTPException(status_code=401, detail="invalid_internal_signature")

        record = dispatcher.resolve_approval(approval_id=str(approval_id), decision=decision)
        if record is None:
            raise HTTPException(status_code=404, detail="approval_not_found")
        return {"approval": record.to_dict()}

    return router
