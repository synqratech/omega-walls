"""Security helpers for notification callback verification."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
from typing import Dict, Optional


def _utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode_b64url(sig: str) -> bytes:
    payload = str(sig or "").strip()
    if not payload:
        return b""
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    return base64.urlsafe_b64decode((payload + padding).encode("ascii"))


def verify_slack_signature(
    *,
    body_bytes: bytes,
    signature: str,
    timestamp: str,
    signing_secret: str,
    max_skew_sec: int = 300,
) -> bool:
    sig = str(signature or "").strip()
    ts = str(timestamp or "").strip()
    secret = str(signing_secret or "").strip()
    if not sig or not ts or not secret:
        return False
    try:
        ts_i = int(ts)
    except ValueError:
        return False
    if abs(_utc_now_ts() - ts_i) > int(max_skew_sec):
        return False
    base = f"v0:{ts}:{body_bytes.decode('utf-8', errors='replace')}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(sig, expected)


def verify_telegram_secret_token(*, provided: str, expected: str) -> bool:
    left = str(provided or "").strip()
    right = str(expected or "").strip()
    if not left or not right:
        return False
    return hmac.compare_digest(left, right)


def verify_internal_hmac(
    *,
    method: str,
    path: str,
    body_bytes: bytes,
    tenant_id: str,
    request_id: str,
    signature: str,
    timestamp: str,
    nonce: str,
    secret: str,
    seen_nonces: Optional[Dict[str, int]] = None,
    max_skew_sec: int = 300,
) -> bool:
    sig_raw = str(signature or "").strip()
    ts_raw = str(timestamp or "").strip()
    nonce_raw = str(nonce or "").strip()
    secret_raw = str(secret or "").strip()
    if not sig_raw or not ts_raw or not nonce_raw or not secret_raw:
        return False
    try:
        ts_i = int(ts_raw)
    except ValueError:
        return False
    now_ts = _utc_now_ts()
    if abs(now_ts - ts_i) > int(max_skew_sec):
        return False

    if seen_nonces is not None:
        key = f"{tenant_id}:{request_id}:{nonce_raw}"
        expires = int(seen_nonces.get(key, 0))
        if expires > now_ts:
            return False
        seen_nonces[key] = now_ts + int(max_skew_sec) * 2

    body_sha = hashlib.sha256(body_bytes).hexdigest()
    canonical = "\n".join(
        [
            str(method or "").upper(),
            str(path or ""),
            body_sha,
            str(tenant_id or ""),
            str(request_id or ""),
            ts_raw,
            nonce_raw,
        ]
    )
    expected = hmac.new(secret_raw.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).digest()
    try:
        provided = _decode_b64url(sig_raw)
    except Exception:
        return False
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def sign_internal_hmac(
    *,
    method: str,
    path: str,
    body_bytes: bytes,
    tenant_id: str,
    request_id: str,
    timestamp: str,
    nonce: str,
    secret: str,
) -> str:
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    canonical = "\n".join(
        [
            str(method or "").upper(),
            str(path or ""),
            body_sha,
            str(tenant_id or ""),
            str(request_id or ""),
            str(timestamp or ""),
            str(nonce or ""),
        ]
    )
    digest = hmac.new(str(secret).encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(digest)

