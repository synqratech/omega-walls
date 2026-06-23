"""Authentication and transport security helpers for API server."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import ipaddress
from typing import Any, Dict, Sequence

from fastapi import HTTPException, Request

from omega.api.runtime_factory import ApiSecurity, ScanRuntime
from omega.runtime.environment import read_secret_value


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _b64url_encode(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _valid_api_key(provided: str, configured_keys: Sequence[str]) -> bool:
    for raw in configured_keys:
        item = str(raw).strip()
        if not item:
            continue
        if item.startswith("sha256:"):
            digest = item.split(":", 1)[1].strip().lower()
            if hashlib.sha256(provided.encode("utf-8")).hexdigest() == digest:
                return True
            continue
        if hmac.compare_digest(provided, item):
            return True
    return False


def _request_is_https_proxy_mode(request: Request, security: ApiSecurity) -> bool:
    peer = str(getattr(getattr(request, "client", None), "host", "") or "").strip()
    try:
        peer_ip = ipaddress.ip_address(peer)
        trusted = any(peer_ip in ipaddress.ip_network(cidr, strict=False) for cidr in security.trusted_proxy_cidrs)
    except ValueError:
        trusted = False
    if not trusted:
        return False
    xfp = str(request.headers.get("x-forwarded-proto", "")).strip().lower()
    if xfp:
        first = xfp.split(",")[0].strip()
        if first == "https":
            return True
    forwarded = str(request.headers.get("forwarded", "")).strip()
    if forwarded:
        parts = [p.strip() for p in forwarded.split(";")]
        for part in parts:
            if part.lower().startswith("proto="):
                proto = part.split("=", 1)[1].strip().strip('"').lower()
                if proto == "https":
                    return True
    return False


def _enforce_transport_security(request: Request, security: ApiSecurity) -> None:
    if not security.require_https:
        return

    mode = str(security.transport_mode).strip().lower()
    if mode == "proxy_tls":
        # Do not trust the ASGI scheme here: proxy middleware can rewrite it before
        # the application sees the request.  Proxy TLS is accepted only when the
        # actual ASGI peer is explicitly allowlisted and supplied the HTTPS marker.
        if not _request_is_https_proxy_mode(request, security):
            raise HTTPException(status_code=400, detail="insecure_transport")
        return

    if mode == "direct_tls":
        if str(request.url.scheme).lower() != "https":
            raise HTTPException(status_code=400, detail="insecure_transport")
        return

    raise HTTPException(status_code=400, detail="insecure_transport")


def _canonical_request_string(
    *,
    method: str,
    path: str,
    body_sha256_hex: str,
    tenant_id: str,
    request_id: str,
    timestamp: str,
    nonce: str,
) -> str:
    return "\n".join(
        [
            str(method).upper().strip(),
            str(path).strip(),
            str(body_sha256_hex).strip(),
            str(tenant_id).strip(),
            str(request_id).strip(),
            str(timestamp).strip(),
            str(nonce).strip(),
        ]
    )


def _verify_hmac_request(
    *,
    request: Request,
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    body_bytes: bytes,
    provided_api_key: str,
) -> None:
    auth = runtime.auth
    if not auth.require_hmac:
        return

    signature = str(request.headers.get(auth.header_signature, "")).strip()
    ts_raw = str(request.headers.get(auth.header_timestamp, "")).strip()
    nonce = str(request.headers.get(auth.header_nonce, "")).strip()
    if not signature or not ts_raw or not nonce:
        raise HTTPException(status_code=401, detail="invalid_signature")

    try:
        ts_i = int(ts_raw)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid_signature") from exc
    now_i = int(time.time())
    if abs(now_i - ts_i) > int(auth.max_clock_skew_sec):
        raise HTTPException(status_code=401, detail="stale_timestamp")

    secret_env = auth.hmac_secret_env
    try:
        secret = read_secret_value(secret_env, required=True)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid_signature")

    tenant_id = str(parsed.get("tenant_id") or "").strip()
    request_id = str(parsed.get("request_id") or "").strip()
    body_hash = _sha256_bytes_hex(body_bytes)
    canonical = _canonical_request_string(
        method=request.method,
        path=request.url.path,
        body_sha256_hex=body_hash,
        tenant_id=tenant_id,
        request_id=request_id,
        timestamp=ts_raw,
        nonce=nonce,
    )
    expected_sig = _b64url_encode(hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(status_code=401, detail="invalid_signature")

    replay_key = _sha256_hex(f"{tenant_id}|{hashlib.sha256(provided_api_key.encode('utf-8')).hexdigest()}|{nonce}")
    if not runtime.replay_cache.check_and_mark(key=replay_key, now_ts=float(now_i)):
        raise HTTPException(status_code=409, detail="replay_detected")


def _notifications_cfg(runtime: ScanRuntime) -> Dict[str, Any]:
    raw = runtime.config.get("notifications", {}) if isinstance(runtime.config, dict) else {}
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _approval_internal_auth_cfg(runtime: ScanRuntime) -> Dict[str, Any]:
    cfg = _notifications_cfg(runtime)
    approvals = cfg.get("approvals", {}) if isinstance(cfg.get("approvals", {}), dict) else {}
    internal_auth = approvals.get("internal_auth", {}) if isinstance(approvals.get("internal_auth", {}), dict) else {}
    return dict(internal_auth or {})

