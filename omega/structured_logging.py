"""FW-005 structured logging via structlog."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as pkg_version
import sys
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import structlog

from omega.log_contract import OmegaLogEvent


_BASE_LOGGER = logging.getLogger(__name__)

_SENSITIVE_KEY_TOKENS = (
    "raw_prompt",
    "full_context",
    "tool_args",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "ssn",
    "email",
    "phone",
    "pii",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s()\-]{7,}")
_PII_TEXT_KEYS = (
    "message",
    "detail",
    "description",
    "fp_hint",
    "text",
    "excerpt",
    "summary",
)
_SAFE_EXACT_KEYS = {
    "ts",
    "event",
    "session_id",
    "mode",
    "level",
    "engine_version",
    "trace_id",
    "decision_id",
    "chunk_hash",
}


def engine_version() -> str:
    try:
        return str(pkg_version("omega-walls"))
    except PackageNotFoundError:
        return "0.1.2"
    except Exception:  # noqa: BLE001
        return "0.1.2"


def _sanitize_value(value: Any, *, key_context: str = "") -> Any:
    if isinstance(value, str):
        key_norm = str(key_context).strip().lower()
        if key_norm in _SAFE_EXACT_KEYS:
            return value
        if any(tok in key_norm for tok in _PII_TEXT_KEYS):
            out = _EMAIL_RE.sub("[REDACTED]", value)
            out = _PHONE_RE.sub("[REDACTED]", out)
            return out
        return value
    if isinstance(value, list):
        return [_sanitize_value(v, key_context=key_context) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_kv(k, v) for k, v in value.items()}
    return value


def _sanitize_kv(key: Any, value: Any) -> Any:
    key_norm = str(key).strip().lower()
    if any(tok in key_norm for tok in _SENSITIVE_KEY_TOKENS):
        return "[REDACTED]"
    return _sanitize_value(value, key_context=key_norm)


def _sanitize_processor(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in dict(event_dict).items():
        sanitized[str(key)] = _sanitize_kv(key, value)
    return sanitized


def _bind_engine_version(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if "engine_version" not in event_dict:
        event_dict["engine_version"] = engine_version()
    return event_dict


def configure_omega_logging(*, log_level: str = "INFO", json_output: bool = True) -> None:
    level_name = str(log_level or "INFO").strip().upper()
    level_value = int(getattr(logging, level_name, logging.INFO))
    renderer = (
        structlog.processors.JSONRenderer(serializer=json.dumps, sort_keys=False)
        if bool(json_output)
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            _bind_engine_version,
            _sanitize_processor,
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "omega.structured") -> Any:
    return structlog.get_logger(name)


@dataclass
class StructuredLogEmitter:
    enabled: bool
    validate: bool
    logger: Any
    events_total: int = 0
    emit_failures: int = 0
    last_error: Optional[str] = None

    def emit(self, payload: OmegaLogEvent | Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            if isinstance(payload, OmegaLogEvent):
                event = payload if not self.validate else OmegaLogEvent.model_validate(payload.model_dump())
            else:
                event = OmegaLogEvent.model_validate(dict(payload)) if self.validate else OmegaLogEvent(**dict(payload))
            body = event.model_dump(exclude_none=True)
            method = str(body.get("level", "INFO")).lower()
            fn = getattr(self.logger, method, self.logger.info)
            fn(**body)
            self.events_total += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.emit_failures += 1
            self.last_error = str(exc)
            _BASE_LOGGER.warning("structured_log_emit_failed: %s", exc)

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "events_total": int(self.events_total),
            "emit_failures": int(self.emit_failures),
            "last_error": self.last_error,
        }


def build_structured_emitter_from_config(
    *,
    config: Mapping[str, Any],
    logger_name: str,
) -> StructuredLogEmitter:
    logging_cfg = (config.get("logging", {}) or {}) if isinstance(config.get("logging", {}), Mapping) else {}
    structured = (
        (logging_cfg.get("structured", {}) or {})
        if isinstance(logging_cfg.get("structured", {}), Mapping)
        else {}
    )
    enabled = bool(structured.get("enabled", False))
    level = str(structured.get("level", "INFO")).strip().upper() or "INFO"
    json_output = bool(structured.get("json_output", True))
    validate = bool(structured.get("validate", True))
    if enabled:
        configure_omega_logging(log_level=level, json_output=json_output)
    return StructuredLogEmitter(
        enabled=enabled,
        validate=validate,
        logger=get_logger(logger_name),
    )
