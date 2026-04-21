"""Per-tool argument validation for ToolGateway (fail-closed)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse


_META_KEYS = {"human_approved", "intent_id", "request_origin"}
_DEFAULT_SHELL_NAME_PATTERNS = ["shell", "bash", "cmd", "exec", "execute", "powershell", "terminal", "sh"]
_DEFAULT_SHELL_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/f\s+/q\b",
    r"\bformat\s+[a-z]:",
    r"\bmkfs(\.| )",
    r"\bdd\s+if=",
    r"\bshutdown\s+(-s|/s)\b",
]


@dataclass(frozen=True)
class ToolArgValidationVerdict:
    checked: bool
    allowed: bool
    reason_code: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class ToolArgValidationConfig:
    enabled: bool
    fail_mode: str
    max_payload_bytes: int
    max_header_count: int
    max_header_key_chars: int
    max_header_value_chars: int
    max_filename_chars: int
    max_content_bytes: int
    shell_like_name_patterns: List[str]
    shell_like_destructive_patterns: List[str]
    max_shell_command_chars: int

    @classmethod
    def from_tools_config(cls, raw_cfg: Mapping[str, Any] | None) -> "ToolArgValidationConfig":
        cfg = dict(raw_cfg or {})
        fail_mode = str(cfg.get("fail_mode", "deny")).strip().lower()
        if fail_mode != "deny":
            raise ValueError("tools.arg_validation.fail_mode must be deny in v1")

        net = dict(cfg.get("network_post", {}) or {})
        wr = dict(cfg.get("write_file", {}) or {})
        sh = dict(cfg.get("shell_like", {}) or {})

        shell_name_patterns = cfg.get("shell_like_name_patterns", _DEFAULT_SHELL_NAME_PATTERNS)
        if not isinstance(shell_name_patterns, list):
            raise ValueError("tools.arg_validation.shell_like_name_patterns must be a list")
        shell_name_patterns = [str(x).strip() for x in shell_name_patterns if str(x).strip()]
        if not shell_name_patterns:
            shell_name_patterns = list(_DEFAULT_SHELL_NAME_PATTERNS)

        destructive_patterns = sh.get("destructive_patterns", _DEFAULT_SHELL_DESTRUCTIVE_PATTERNS)
        if not isinstance(destructive_patterns, list):
            raise ValueError("tools.arg_validation.shell_like.destructive_patterns must be a list")
        destructive_patterns = [str(x).strip() for x in destructive_patterns if str(x).strip()]
        if not destructive_patterns:
            destructive_patterns = list(_DEFAULT_SHELL_DESTRUCTIVE_PATTERNS)

        instance = cls(
            enabled=bool(cfg.get("enabled", True)),
            fail_mode=fail_mode,
            max_payload_bytes=int(net.get("max_payload_bytes", 8192)),
            max_header_count=int(net.get("max_headers", 16)),
            max_header_key_chars=int(net.get("max_header_key_chars", 64)),
            max_header_value_chars=int(net.get("max_header_value_chars", 256)),
            max_filename_chars=int(wr.get("max_filename_chars", 120)),
            max_content_bytes=int(wr.get("max_content_bytes", 8192)),
            shell_like_name_patterns=shell_name_patterns,
            shell_like_destructive_patterns=destructive_patterns,
            max_shell_command_chars=int(sh.get("max_command_chars", 2048)),
        )
        if (
            instance.max_payload_bytes <= 0
            or instance.max_header_count <= 0
            or instance.max_header_key_chars <= 0
            or instance.max_header_value_chars <= 0
            or instance.max_filename_chars <= 0
            or instance.max_content_bytes <= 0
            or instance.max_shell_command_chars <= 0
        ):
            raise ValueError("tools.arg_validation limits must be > 0")
        return instance


def validate_tool_args(tool_name: str, args: Mapping[str, Any], cfg: ToolArgValidationConfig) -> ToolArgValidationVerdict:
    if not cfg.enabled:
        return ToolArgValidationVerdict(checked=False, allowed=True)

    tool_norm = str(tool_name or "").strip().lower()
    payload = dict(args or {})

    if tool_norm == "network_post":
        return _validate_network_post(payload, cfg)
    if tool_norm == "write_file":
        return _validate_write_file(payload, cfg)
    if _is_shell_like(tool_norm, cfg.shell_like_name_patterns):
        return _validate_shell_like(payload, cfg)
    return ToolArgValidationVerdict(checked=False, allowed=True)


def _is_shell_like(tool_name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        p = str(pattern).strip()
        if not p:
            continue
        try:
            if re.search(r"[.^$*+?{}\[\]|()\\]", p):
                if re.search(p, tool_name, flags=re.IGNORECASE):
                    return True
                continue
            token = re.escape(p.lower())
            if re.search(rf"(^|[_:\-]){token}($|[_:\-])", tool_name.lower()):
                return True
        except re.error:
            continue
    return False


def _deny_schema(reason: str) -> ToolArgValidationVerdict:
    return ToolArgValidationVerdict(
        checked=True,
        allowed=False,
        reason_code="INVALID_TOOL_ARGS_SCHEMA",
        reason=reason,
    )


def _deny_security(reason: str) -> ToolArgValidationVerdict:
    return ToolArgValidationVerdict(
        checked=True,
        allowed=False,
        reason_code="INVALID_TOOL_ARGS_SECURITY",
        reason=reason,
    )


def _deny_shell(reason: str) -> ToolArgValidationVerdict:
    return ToolArgValidationVerdict(
        checked=True,
        allowed=False,
        reason_code="INVALID_TOOL_ARGS_SHELLLIKE",
        reason=reason,
    )


def _validate_network_post(args: Dict[str, Any], cfg: ToolArgValidationConfig) -> ToolArgValidationVerdict:
    allowed_keys = {"url", "payload", "body", "raw_args", "headers"} | _META_KEYS
    extra = sorted(set(args.keys()) - allowed_keys)
    if extra:
        return _deny_schema(f"network_post unsupported keys: {', '.join(extra)}")

    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        maybe = _extract_url_from_text(str(args.get("raw_args", "")))
        if not maybe:
            return _deny_schema("network_post requires non-empty url")
        url = maybe

    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _deny_security("network_post url must be absolute http/https")

    payload = args.get("payload")
    if payload is None:
        payload = args.get("body")
    if payload is None:
        payload = args.get("raw_args")
    if payload is None:
        return _deny_schema("network_post requires payload|body|raw_args")
    payload_bytes = len(str(payload).encode("utf-8"))
    if payload_bytes > cfg.max_payload_bytes:
        return _deny_security("network_post payload exceeds max_payload_bytes")

    headers = args.get("headers")
    if headers is not None:
        if not isinstance(headers, dict):
            return _deny_schema("network_post headers must be object")
        if len(headers) > cfg.max_header_count:
            return _deny_security("network_post headers exceed max_headers")
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return _deny_schema("network_post headers must be string:string")
            if len(key) > cfg.max_header_key_chars or len(value) > cfg.max_header_value_chars:
                return _deny_security("network_post header key/value exceeds limits")

    return ToolArgValidationVerdict(checked=True, allowed=True)


def _validate_write_file(args: Dict[str, Any], cfg: ToolArgValidationConfig) -> ToolArgValidationVerdict:
    allowed_keys = {"filename", "content", "raw_args"} | _META_KEYS
    extra = sorted(set(args.keys()) - allowed_keys)
    if extra:
        return _deny_schema(f"write_file unsupported keys: {', '.join(extra)}")

    filename = args.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        return _deny_schema("write_file requires non-empty filename")
    filename = filename.strip()
    if len(filename) > cfg.max_filename_chars:
        return _deny_security("write_file filename exceeds max_filename_chars")
    if ".." in filename:
        return _deny_security("write_file filename cannot contain traversal '..'")
    if filename.startswith(("/", "\\")) or re.match(r"^[a-zA-Z]:", filename):
        return _deny_security("write_file filename must be relative basename only")
    if "/" in filename or "\\" in filename:
        return _deny_security("write_file filename cannot include path separators")

    content = args.get("content")
    if content is None:
        content = args.get("raw_args")
    if content is None:
        return _deny_schema("write_file requires content|raw_args")
    content_bytes = len(str(content).encode("utf-8"))
    if content_bytes > cfg.max_content_bytes:
        return _deny_security("write_file content exceeds max_content_bytes")
    return ToolArgValidationVerdict(checked=True, allowed=True)


def _validate_shell_like(args: Dict[str, Any], cfg: ToolArgValidationConfig) -> ToolArgValidationVerdict:
    allowed_keys = {"command", "raw_args"} | _META_KEYS
    extra = sorted(set(args.keys()) - allowed_keys)
    if extra:
        return _deny_shell(f"shell-like tool unsupported keys: {', '.join(extra)}")

    command = args.get("command")
    if command is None:
        command = args.get("raw_args")
    if not isinstance(command, str) or not command.strip():
        return _deny_shell("shell-like tools require non-empty command|raw_args")
    command = command.strip()
    if len(command) > cfg.max_shell_command_chars:
        return _deny_shell("shell-like command exceeds max_command_chars")

    low = command.lower()
    for pattern in cfg.shell_like_destructive_patterns:
        try:
            if re.search(pattern, low, flags=re.IGNORECASE):
                return _deny_shell("shell-like command matches destructive pattern")
        except re.error:
            continue
    return ToolArgValidationVerdict(checked=True, allowed=True)


def _extract_url_from_text(text: str) -> Optional[str]:
    m = re.search(r"https?://[^\s,\"')]+", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(0)
