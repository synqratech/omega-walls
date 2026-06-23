"""Canonical tool-intent binding for server-side human approvals."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from omega.interfaces.contracts_v1 import ToolRequest


_UNTRUSTED_META_ARG_KEYS = {
    "human_approved",
    "approval_id",
    "intent_id",
    "request_origin",
}


def _canonical_json_value(value: Any, *, path: str = "args") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda x: str(x)):
            key_s = str(key)
            if path == "args" and key_s in _UNTRUSTED_META_ARG_KEYS:
                continue
            out[key_s] = _canonical_json_value(value[key], path=f"{path}.{key_s}")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item, path=f"{path}[]") for item in value]
    raise ValueError(f"{path} contains unsupported value type: {type(value).__name__}")


def canonical_tool_args(args: Mapping[str, Any]) -> bytes:
    canonical = _canonical_json_value(dict(args or {}), path="args")
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def tool_args_sha256(args: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_tool_args(args)).hexdigest()


def tool_intent_id(request: ToolRequest) -> str:
    tenant_id = str(request.tenant_id or "").strip()
    session_id = str(request.session_id or "").strip()
    actor_id = str(request.actor_id or "").strip()
    tool_name = str(request.tool_name or "").strip().lower()
    if not tenant_id or not session_id or not tool_name:
        raise ValueError("tool intent requires tenant_id, session_id, and tool_name")
    args_hash = tool_args_sha256(request.args)
    payload = "\n".join([tenant_id, session_id, actor_id, tool_name, args_hash])
    return f"tint_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"
