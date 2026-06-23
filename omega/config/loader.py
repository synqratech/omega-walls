"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib.resources as importlib_resources
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import yaml
from omega.runtime.environment import expand_omega_environment, parse_env_override
from omega.config.validators import (
    validate_api_config,
    validate_benchmark_configs,
    validate_effects_config,
    validate_off_policy_config,
    validate_projector_config,
    validate_production_profile_contract,
    validate_release_gate_config,
    validate_licensing_config,
    validate_runtime_integrity_config,
    validate_skillbox_config,
    validate_telemetry_config,
    validate_tools_config,
)

LOGGER = logging.getLogger(__name__)

_CONFIG_LAYER_FILES: Dict[str, str] = {
    "pi0": "pi0_defaults.yml",
    "pi0_semantic": "pi0_semantic.yml",
    "projector": "projector.yml",
    "effects": "effects.yml",
    "runtime_integrity": "runtime_integrity.yml",
    "skillbox": "skillbox.yml",
    "omega": "omega_defaults.yml",
    "off_policy": "off_policy.yml",
    "source_policy": "source_policy.yml",
    "tools": "tools.yml",
    "retriever": "retriever.yml",
    "api": "api.yml",
    "monitoring": "monitoring.yml",
    "notifications": "notifications.yml",
    "telemetry": "telemetry.yml",
    "bipia": "bipia.yml",
    "deepset": "deepset.yml",
    "pitheta_dataset_registry": "pitheta_dataset_registry.yml",
    "pitheta_train": "pitheta_train.yml",
    "release_gate": "release_gate.yml",
    "licensing": "licensing.yml",
}
_CONFIG_LAYER_ORDER: Tuple[str, ...] = (
    "pi0",
    "pi0_semantic",
    "projector",
    "effects",
    "runtime_integrity",
    "skillbox",
    "omega",
    "off_policy",
    "source_policy",
    "tools",
    "retriever",
    "api",
    "monitoring",
    "notifications",
    "telemetry",
    "bipia",
    "deepset",
    "pitheta_dataset_registry",
    "pitheta_train",
    "release_gate",
    "licensing",
)
_BUNDLED_CONFIG_PACKAGE = "omega.config"
_BUNDLED_CONFIG_ROOT = "resources"


@dataclass
class ConfigSnapshot:
    resolved: Dict[str, Any]
    file_hashes: Dict[str, str]
    resolved_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_yaml_bytes(content: bytes, source: str) -> Dict[str, Any]:
    parsed = yaml.safe_load(content) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML root must be mapping: {source}")
    return parsed


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_bytes()
    return _parse_yaml_bytes(content, source=str(path))


def _load_bundled_yaml(*parts: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    traversable = importlib_resources.files(_BUNDLED_CONFIG_PACKAGE).joinpath(_BUNDLED_CONFIG_ROOT, *parts)
    if not traversable.exists():
        return {}, None, None
    content = traversable.read_bytes()
    source = f"pkg://{_BUNDLED_CONFIG_PACKAGE}/{_BUNDLED_CONFIG_ROOT}/{'/'.join(parts)}"
    digest = _sha256_bytes(content)
    return _parse_yaml_bytes(content, source=source), source, digest




def _load_enterprise_bundled_yaml(*parts: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    try:
        traversable = importlib_resources.files("omega_walls_enterprise.config").joinpath(_BUNDLED_CONFIG_ROOT, *parts)
    except ModuleNotFoundError:
        return {}, None, None
    if not traversable.exists():
        return {}, None, None
    content = traversable.read_bytes()
    source = f"pkg://omega_walls_enterprise.config/{_BUNDLED_CONFIG_ROOT}/{'/'.join(parts)}"
    digest = _sha256_bytes(content)
    return _parse_yaml_bytes(content, source=source), source, digest

def _raise_profile_not_found(*, profile: str) -> None:
    raise ValueError(f"profile not found: {profile}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any], env: Dict[str, str], prefix: str = "OMEGA__") -> Dict[str, Any]:
    """Apply env vars like ``OMEGA__API__PORT=8080`` to nested keys.

    Structured YAML/JSON values are accepted so list-valued deployment fields
    such as trusted proxy CIDRs can be configured without custom files.
    """
    updated = dict(config)
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        if any(not part for part in path):
            raise ValueError(f"invalid nested environment override: {key}")
        cur: Dict[str, Any] = updated
        for part in path[:-1]:
            next_val = cur.get(part)
            if not isinstance(next_val, dict):
                next_val = {}
                cur[part] = next_val
            cur = next_val
        cur[path[-1]] = parse_env_override(value)
    return updated


def validate_resolved_config(config: Dict[str, Any]) -> None:
    # Wave A/B/C split: keep facade entrypoint stable while delegating by domain.
    validate_off_policy_config(config)

    source_policy = config.get("source_policy", {})
    default_trust = source_policy.get("default_trust", "untrusted")
    valid_trust = {"trusted", "semi", "untrusted", "semi_trusted"}
    if default_trust not in valid_trust:
        raise ValueError("source_policy.default_trust must be trusted|semi|semi_trusted|untrusted")

    validate_tools_config(config)
    validate_effects_config(config)
    validate_runtime_integrity_config(config)
    validate_skillbox_config(config)

    runtime_cfg = config.get("runtime", {}) or {}
    if runtime_cfg and not isinstance(runtime_cfg, dict):
        raise ValueError("runtime must be a mapping")
    guard_mode = str(runtime_cfg.get("guard_mode", "enforce")).strip().lower()
    if guard_mode not in {"enforce", "monitor"}:
        raise ValueError("runtime.guard_mode must be enforce|monitor")
    required_components = runtime_cfg.get("required_components", [])
    if not isinstance(required_components, list):
        raise ValueError("runtime.required_components must be a list")
    allowed_required_components = {"attachments", "vision"}
    for idx, component in enumerate(required_components):
        if str(component) not in allowed_required_components:
            raise ValueError(
                f"runtime.required_components[{idx}] must be attachments|vision"
            )

    monitoring_cfg = config.get("monitoring", {}) or {}
    if monitoring_cfg and not isinstance(monitoring_cfg, dict):
        raise ValueError("monitoring must be a mapping")
    if isinstance(monitoring_cfg, dict) and monitoring_cfg:
        _ = bool(monitoring_cfg.get("enabled", False))
        agg_window = str(monitoring_cfg.get("aggregation_window", "1h")).strip().lower()
        if not agg_window:
            raise ValueError("monitoring.aggregation_window must be non-empty")
        export_cfg = monitoring_cfg.get("export", {}) or {}
        if export_cfg and not isinstance(export_cfg, dict):
            raise ValueError("monitoring.export must be a mapping")
        if isinstance(export_cfg, dict) and export_cfg:
            if not str(export_cfg.get("path", "artifacts/monitor/monitor_events.jsonl")).strip():
                raise ValueError("monitoring.export.path must be non-empty")
            rotation = str(export_cfg.get("rotation", "none")).strip().lower()
            if rotation not in {"none", "daily", "size"}:
                raise ValueError("monitoring.export.rotation must be none|daily|size")
            if int(export_cfg.get("rotation_size_mb", 100)) <= 0:
                raise ValueError("monitoring.export.rotation_size_mb must be > 0")
            out_format = str(export_cfg.get("format", "jsonl")).strip().lower()
            if out_format not in {"jsonl", "csv"}:
                raise ValueError("monitoring.export.format must be jsonl|csv")
        hints_cfg = monitoring_cfg.get("false_positive_hints", {}) or {}
        if hints_cfg and not isinstance(hints_cfg, dict):
            raise ValueError("monitoring.false_positive_hints must be a mapping")
        if isinstance(hints_cfg, dict) and hints_cfg:
            for hint_name in ("low_confidence_near_threshold", "trusted_source_mismatch", "transient_spike"):
                sub_cfg = hints_cfg.get(hint_name, {}) or {}
                if sub_cfg and not isinstance(sub_cfg, dict):
                    raise ValueError(f"monitoring.false_positive_hints.{hint_name} must be a mapping")

    logging_cfg = config.get("logging", {}) or {}
    if logging_cfg and not isinstance(logging_cfg, dict):
        raise ValueError("logging must be a mapping")
    if isinstance(logging_cfg, dict) and logging_cfg:
        log_mode = str(logging_cfg.get("mode", "OFF_ONLY")).strip().upper()
        if log_mode not in {"OFF_ONLY", "PER_STEP"}:
            raise ValueError("logging.mode must be OFF_ONLY|PER_STEP")
        capture_text = str(logging_cfg.get("capture_text", "NEVER")).strip().upper()
        if capture_text not in {"NEVER", "REDACTED", "ALLOWLISTED"}:
            raise ValueError("logging.capture_text must be NEVER|REDACTED|ALLOWLISTED")
        if int(logging_cfg.get("max_text_chars", 800)) <= 0:
            raise ValueError("logging.max_text_chars must be > 0")
        allowlisted = logging_cfg.get("allowlisted_sources", [])
        if allowlisted is not None and not isinstance(allowlisted, list):
            raise ValueError("logging.allowlisted_sources must be a list")
        structured_cfg = logging_cfg.get("structured", {}) or {}
        if structured_cfg and not isinstance(structured_cfg, dict):
            raise ValueError("logging.structured must be a mapping")
        if isinstance(structured_cfg, dict) and structured_cfg:
            _ = bool(structured_cfg.get("enabled", False))
            level = str(structured_cfg.get("level", "INFO")).strip().upper()
            if level not in {"DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"}:
                raise ValueError("logging.structured.level must be DEBUG|INFO|WARN|ERROR|CRITICAL")
            _ = bool(structured_cfg.get("json_output", True))
            _ = bool(structured_cfg.get("validate", True))

    deployment_cfg = config.get("deployment", {}) or {}
    if deployment_cfg:
        if not isinstance(deployment_cfg, dict):
            raise ValueError("deployment must be a mapping")
        mode = str(deployment_cfg.get("mode", "library")).strip().lower()
        if mode not in {"library", "customer_self_hosted", "container", "sidecar"}:
            raise ValueError("deployment.mode must be library|customer_self_hosted|container|sidecar")
        data_dir = str(deployment_cfg.get("data_dir", "")).strip()
        if mode == "customer_self_hosted" and not data_dir:
            raise ValueError("deployment.data_dir is required for customer_self_hosted mode")
        if bool(deployment_cfg.get("require_absolute_data_dir", False)) and data_dir:
            data_path = Path(data_dir).expanduser()
            posix_data_path = PurePosixPath(data_dir)
            if not data_path.is_absolute() and not posix_data_path.is_absolute():
                raise ValueError("deployment.data_dir must be absolute")
            if data_path in {Path('/'), Path('/proc'), Path('/sys'), Path('/dev')} or posix_data_path in {
                PurePosixPath("/"),
                PurePosixPath("/proc"),
                PurePosixPath("/sys"),
                PurePosixPath("/dev"),
            }:
                raise ValueError("deployment.data_dir points to an unsafe filesystem root")
        required_subdirs = deployment_cfg.get("required_subdirs", [])
        if not isinstance(required_subdirs, list):
            raise ValueError("deployment.required_subdirs must be a list")
        allowed_subdirs = {"state", "logs", "audit", "replay", "backups", "tmp", "control-plane"}
        unknown_subdirs = [str(x) for x in required_subdirs if str(x) not in allowed_subdirs]
        if unknown_subdirs:
            raise ValueError("deployment.required_subdirs contains unsupported values")

    retriever_cfg = config.get("retriever", {})
    if retriever_cfg:
        backend = str(retriever_cfg.get("backend", "sqlite_fts")).strip().lower()
        if backend not in {"sqlite_fts", "external"}:
            raise ValueError("retriever.backend must be sqlite_fts|external")
        sqlite_cfg = retriever_cfg.get("sqlite_fts", {}) or {}
        top_k = int(sqlite_cfg.get("default_top_k", 4))
        if top_k <= 0:
            raise ValueError("retriever.sqlite_fts.default_top_k must be > 0")
        include_ext = sqlite_cfg.get("include_extensions", [".txt", ".md"])
        if not isinstance(include_ext, list):
            raise ValueError("retriever.sqlite_fts.include_extensions must be a list")
        attachments_cfg = sqlite_cfg.get("attachments", {}) or {}
        if attachments_cfg and not isinstance(attachments_cfg, dict):
            raise ValueError("retriever.sqlite_fts.attachments must be a mapping")
        if isinstance(attachments_cfg, dict) and attachments_cfg:
            for key in ("max_file_bytes", "max_extracted_chars", "max_chunk_chars"):
                if int(attachments_cfg.get(key, 1)) <= 0:
                    raise ValueError(f"retriever.sqlite_fts.attachments.{key} must be > 0")
            overlap = int(attachments_cfg.get("chunk_overlap", 0))
            max_chunk = int(attachments_cfg.get("max_chunk_chars", 2000))
            if overlap < 0:
                raise ValueError("retriever.sqlite_fts.attachments.chunk_overlap must be >= 0")
            if overlap >= max_chunk:
                raise ValueError("retriever.sqlite_fts.attachments.chunk_overlap must be < max_chunk_chars")
            scan_alpha = float(attachments_cfg.get("scan_like_min_alpha_ratio", 0.3))
            if scan_alpha < 0.0 or scan_alpha > 1.0:
                raise ValueError("retriever.sqlite_fts.attachments.scan_like_min_alpha_ratio must be in [0,1]")
            if int(attachments_cfg.get("scan_like_min_chars_per_page", 1)) <= 0:
                raise ValueError("retriever.sqlite_fts.attachments.scan_like_min_chars_per_page must be > 0")
            zip_cfg = attachments_cfg.get("zip", {}) or {}
            if zip_cfg and not isinstance(zip_cfg, dict):
                raise ValueError("retriever.sqlite_fts.attachments.zip must be a mapping")
            if isinstance(zip_cfg, dict) and zip_cfg:
                for key in ("max_files", "max_depth", "max_total_bytes"):
                    if int(zip_cfg.get(key, 1)) <= 0:
                        raise ValueError(f"retriever.sqlite_fts.attachments.zip.{key} must be > 0")
                _ = bool(zip_cfg.get("enabled", False))
                _ = bool(zip_cfg.get("allow_encrypted", False))
            ocr_cfg = attachments_cfg.get("ocr", {}) or {}
            if ocr_cfg and not isinstance(ocr_cfg, dict):
                raise ValueError("retriever.sqlite_fts.attachments.ocr must be a mapping")
            if isinstance(ocr_cfg, dict) and ocr_cfg:
                enabled = str(ocr_cfg.get("enabled", "auto")).strip().lower()
                if enabled not in {"auto", "true", "false"}:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.enabled must be auto|true|false")
                provider = str(ocr_cfg.get("provider", "rapidocr")).strip().lower()
                if provider not in {"paddleocr", "rapidocr"}:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.provider must be rapidocr|paddleocr")
                execution_mode = str(ocr_cfg.get("execution_mode", "inline")).strip().lower()
                if execution_mode not in {"inline", "persistent_worker"}:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.execution_mode must be inline|persistent_worker")
                if execution_mode == "persistent_worker" and provider != "rapidocr":
                    raise ValueError("persistent OCR worker currently supports rapidocr only")
                if "prewarm" in ocr_cfg and type(ocr_cfg.get("prewarm")) is not bool:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.prewarm must be boolean")
                if float(ocr_cfg.get("worker_startup_timeout_sec", 25.0)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_startup_timeout_sec must be > 0")
                if float(ocr_cfg.get("worker_request_timeout_sec", 15.0)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_request_timeout_sec must be > 0")
                if int(ocr_cfg.get("worker_max_memory_mb", 2048)) < 256:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_max_memory_mb must be >= 256")
                if int(ocr_cfg.get("worker_max_requests", 500)) < 1:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_max_requests must be >= 1")
                pool_size = int(ocr_cfg.get("worker_pool_size", 1))
                if pool_size < 1 or pool_size > 8:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_pool_size must be in [1,8]")
                max_pending = int(ocr_cfg.get("worker_max_pending_requests", 2))
                if max_pending < 0 or max_pending > 128:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_max_pending_requests must be in [0,128]")
                if float(ocr_cfg.get("worker_queue_timeout_sec", 1.0)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.worker_queue_timeout_sec must be > 0")
                if not str(ocr_cfg.get("lang", "en")).strip():
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.lang must be non-empty")
                _ = bool(ocr_cfg.get("use_angle_cls", True))
                if int(ocr_cfg.get("max_text_chars", 1)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.max_text_chars must be > 0")
                if int(ocr_cfg.get("max_spans_per_chunk", 1)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.max_spans_per_chunk must be > 0")
                min_confidence = float(ocr_cfg.get("min_confidence", 0.50))
                if min_confidence < 0.0 or min_confidence > 1.0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.min_confidence must be in [0,1]")
                if int(ocr_cfg.get("max_spans", 1)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.max_spans must be > 0")
                if int(ocr_cfg.get("max_span_chars", 1)) <= 0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.max_span_chars must be > 0")
                if float(ocr_cfg.get("min_polygon_area_px", 0.0)) < 0.0:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.min_polygon_area_px must be >= 0")
                _ = bool(ocr_cfg.get("require_geometry", True))
                failure_policy = str(ocr_cfg.get("failure_policy", "degrade")).strip().lower()
                if failure_policy not in {"degrade", "quarantine", "fail_closed"}:
                    raise ValueError("retriever.sqlite_fts.attachments.ocr.failure_policy must be degrade|quarantine|fail_closed")

    validate_production_profile_contract(config)
    validate_api_config(config)

    notifications_cfg = config.get("notifications", {}) or {}
    if notifications_cfg:
        if not isinstance(notifications_cfg, dict):
            raise ValueError("notifications must be a mapping")
        _ = bool(notifications_cfg.get("enabled", False))
        startup_cfg = notifications_cfg.get("startup", {}) or {}
        if startup_cfg and not isinstance(startup_cfg, dict):
            raise ValueError("notifications.startup must be a mapping")
        if isinstance(startup_cfg, dict) and startup_cfg:
            for startup_name in ("preflight", "outreach"):
                section = startup_cfg.get(startup_name, {}) or {}
                if section and not isinstance(section, dict):
                    raise ValueError(f"notifications.startup.{startup_name} must be a mapping")
                if not isinstance(section, dict):
                    continue
                _ = bool(section.get("enabled", True))
                _ = bool(section.get("terminal", True))
                _ = bool(section.get("channels", True))
                _ = bool(section.get("once_per_process", True))
            outreach_cfg = startup_cfg.get("outreach", {}) or {}
            if isinstance(outreach_cfg, dict):
                if bool(outreach_cfg.get("enabled", True)):
                    for key, default in (
                        ("github_url", "https://github.com/synqratech/omega-walls"),
                        ("docs_url", "https://github.com/synqratech/omega-walls/tree/main/docs"),
                        ("linkedin_url", "https://www.linkedin.com/in/anvifedotov/"),
                        ("contact_email", "anton.f@synqra.tech"),
                    ):
                        if not str(outreach_cfg.get(key, default)).strip():
                            raise ValueError(f"notifications.startup.outreach.{key} must be non-empty")
                _ = bool(outreach_cfg.get("commercial_cta_enabled", True))
        approvals_cfg = notifications_cfg.get("approvals", {}) or {}
        if approvals_cfg and not isinstance(approvals_cfg, dict):
            raise ValueError("notifications.approvals must be a mapping")
        if isinstance(approvals_cfg, dict) and approvals_cfg:
            backend = str(approvals_cfg.get("backend", "memory")).strip().lower()
            if backend not in {"memory", "sqlite"}:
                raise ValueError("notifications.approvals.backend must be memory|sqlite")
            if backend == "sqlite" and not str(
                approvals_cfg.get("sqlite_path", "artifacts/state/notification_approvals.db")
            ).strip():
                raise ValueError("notifications.approvals.sqlite_path must be non-empty for sqlite backend")
            if int(approvals_cfg.get("timeout_sec", 900)) <= 0:
                raise ValueError("notifications.approvals.timeout_sec must be > 0")
            internal_auth = approvals_cfg.get("internal_auth", {}) or {}
            if internal_auth and not isinstance(internal_auth, dict):
                raise ValueError("notifications.approvals.internal_auth must be a mapping")
            if isinstance(internal_auth, dict) and internal_auth:
                _ = bool(internal_auth.get("require_hmac", True))
                if not str(internal_auth.get("hmac_secret_env", "OMEGA_NOTIFICATION_HMAC_SECRET")).strip():
                    raise ValueError("notifications.approvals.internal_auth.hmac_secret_env must be non-empty")
                headers = internal_auth.get("headers", {}) or {}
                if headers and not isinstance(headers, dict):
                    raise ValueError("notifications.approvals.internal_auth.headers must be a mapping")
                for key, default_header in (
                    ("signature", "X-Internal-Signature"),
                    ("timestamp", "X-Internal-Timestamp"),
                    ("nonce", "X-Internal-Nonce"),
                ):
                    if not str(headers.get(key, default_header)).strip():
                        raise ValueError(
                            f"notifications.approvals.internal_auth.headers.{key} must be non-empty"
                        )
                if int(internal_auth.get("max_clock_skew_sec", 300)) <= 0:
                    raise ValueError(
                        "notifications.approvals.internal_auth.max_clock_skew_sec must be > 0"
                    )

        for provider_name in ("slack", "telegram"):
            provider_cfg = notifications_cfg.get(provider_name, {}) or {}
            if provider_cfg and not isinstance(provider_cfg, dict):
                raise ValueError(f"notifications.{provider_name} must be a mapping")
            if not isinstance(provider_cfg, dict):
                continue
            _ = bool(provider_cfg.get("enabled", False))
            triggers = provider_cfg.get("triggers", [])
            if triggers is not None and not isinstance(triggers, list):
                raise ValueError(f"notifications.{provider_name}.triggers must be a list")
            min_risk = provider_cfg.get("min_risk_score", None)
            if min_risk is not None:
                mr = float(min_risk)
                if mr < 0.0 or mr > 1.0:
                    raise ValueError(f"notifications.{provider_name}.min_risk_score must be in [0,1]")
            throttle_cfg = provider_cfg.get("throttle_windows_sec", {}) or {}
            if throttle_cfg and not isinstance(throttle_cfg, dict):
                raise ValueError(f"notifications.{provider_name}.throttle_windows_sec must be a mapping")
            if isinstance(throttle_cfg, dict) and throttle_cfg:
                for key in ("WARN", "BLOCK"):
                    if int(throttle_cfg.get(key, 0)) < 0:
                        raise ValueError(f"notifications.{provider_name}.throttle_windows_sec.{key} must be >= 0")
            if provider_name == "slack":
                if not str(provider_cfg.get("bot_token_env", "SLACK_BOT_TOKEN")).strip():
                    raise ValueError("notifications.slack.bot_token_env must be non-empty")
                if not str(provider_cfg.get("channel_env", "SLACK_ALERT_CHANNEL")).strip():
                    raise ValueError("notifications.slack.channel_env must be non-empty")
                if not str(provider_cfg.get("signing_secret_env", "SLACK_SIGNING_SECRET")).strip():
                    raise ValueError("notifications.slack.signing_secret_env must be non-empty")
            if provider_name == "telegram":
                if not str(provider_cfg.get("bot_token_env", "TG_BOT_TOKEN")).strip():
                    raise ValueError("notifications.telegram.bot_token_env must be non-empty")
                if not str(provider_cfg.get("chat_id_env", "TG_ADMIN_CHAT_ID")).strip():
                    raise ValueError("notifications.telegram.chat_id_env must be non-empty")
                if not str(provider_cfg.get("secret_token_env", "TG_BOT_SECRET_TOKEN")).strip():
                    raise ValueError("notifications.telegram.secret_token_env must be non-empty")
        webhook_cfg = notifications_cfg.get("webhook", {}) or {}
        if webhook_cfg and not isinstance(webhook_cfg, dict):
            raise ValueError("notifications.webhook must be a mapping")
        if isinstance(webhook_cfg, dict):
            _ = bool(webhook_cfg.get("enabled", False))
            if bool(webhook_cfg.get("enabled", False)):
                if not str(webhook_cfg.get("url", "")).strip():
                    raise ValueError("notifications.webhook.url must be non-empty when enabled")
            types = webhook_cfg.get("types", [])
            if types and not isinstance(types, list):
                raise ValueError("notifications.webhook.types must be a list")

    validate_telemetry_config(config)
    validate_benchmark_configs(config)
    validate_release_gate_config(config)
    validate_licensing_config(config)

    pi0_cfg = config.get("pi0", {})
    fuzzy_runtime_cfg = (pi0_cfg.get("fuzzy_runtime", {}) or {})
    if fuzzy_runtime_cfg:
        if not isinstance(fuzzy_runtime_cfg, dict):
            raise ValueError("pi0.fuzzy_runtime must be a mapping")
        _ = bool(fuzzy_runtime_cfg.get("enabled", True))
        long_thr = int(fuzzy_runtime_cfg.get("long_text_threshold_chars", 1800))
        if long_thr <= 0:
            raise ValueError("pi0.fuzzy_runtime.long_text_threshold_chars must be > 0")
        _ = bool(fuzzy_runtime_cfg.get("require_pre_hit_for_long_text", True))
        window_chars = int(fuzzy_runtime_cfg.get("window_chars", 220))
        if window_chars <= 0:
            raise ValueError("pi0.fuzzy_runtime.window_chars must be > 0")
        max_windows = int(fuzzy_runtime_cfg.get("max_windows", 12))
        if max_windows <= 0:
            raise ValueError("pi0.fuzzy_runtime.max_windows must be > 0")
        max_total = int(fuzzy_runtime_cfg.get("max_total_scan_chars", 2200))
        if max_total <= 0:
            raise ValueError("pi0.fuzzy_runtime.max_total_scan_chars must be > 0")
        prefix_cap = int(fuzzy_runtime_cfg.get("prefix_fallback_chars", 1200))
        if prefix_cap <= 0:
            raise ValueError("pi0.fuzzy_runtime.prefix_fallback_chars must be > 0")
        if prefix_cap > max_total:
            raise ValueError("pi0.fuzzy_runtime.prefix_fallback_chars must be <= max_total_scan_chars")

    semantic_cfg = pi0_cfg.get("semantic", {})
    if semantic_cfg:
        enabled = str(semantic_cfg.get("enabled", "auto")).lower()
        if enabled not in {"auto", "true", "false"}:
            raise ValueError("pi0.semantic.enabled must be auto|true|false")
        fusion_mode = str(semantic_cfg.get("fusion_mode", "additive_cap")).lower()
        if fusion_mode != "additive_cap":
            raise ValueError("pi0.semantic.fusion_mode must be additive_cap")
        _ = bool(semantic_cfg.get("promotion_requires_rule_signal", True))

        for key in ("sim_thresholds", "polarity_semantic_threshold"):
            table = semantic_cfg.get(key, {}) or {}
            if not isinstance(table, dict):
                raise ValueError(f"pi0.semantic.{key} must be a mapping")
            for wall in (
                "override_instructions",
                "secret_exfiltration",
                "tool_or_action_abuse",
                "policy_evasion",
            ):
                val = float(table.get(wall, 0.0))
                if val < 0.0 or val > 1.0:
                    raise ValueError(f"pi0.semantic.{key}.{wall} must be in [0,1]")

        boosts = semantic_cfg.get("boost_caps", {}) or {}
        if not isinstance(boosts, dict):
            raise ValueError("pi0.semantic.boost_caps must be a mapping")
        for wall in (
            "override_instructions",
            "secret_exfiltration",
            "tool_or_action_abuse",
            "policy_evasion",
        ):
            val = float(boosts.get(wall, 0.0))
            if val < 0.0:
                raise ValueError(f"pi0.semantic.boost_caps.{wall} must be >= 0")

        guard = semantic_cfg.get("guard_thresholds", {}) or {}
        if not isinstance(guard, dict):
            raise ValueError("pi0.semantic.guard_thresholds must be a mapping")
        for key in ("negation", "protect", "tutorial"):
            g = float(guard.get(key, 0.0))
            if g < 0.0 or g > 1.0:
                raise ValueError(f"pi0.semantic.guard_thresholds.{key} must be in [0,1]")

        guard_by_wall = semantic_cfg.get("guard_apply_by_wall", {}) or {}
        if guard_by_wall and not isinstance(guard_by_wall, dict):
            raise ValueError("pi0.semantic.guard_apply_by_wall must be a mapping")
        for wall in (
            "override_instructions",
            "secret_exfiltration",
            "tool_or_action_abuse",
            "policy_evasion",
        ):
            guards_for_wall = guard_by_wall.get(wall, None)
            if guards_for_wall is None:
                continue
            if not isinstance(guards_for_wall, list):
                raise ValueError(f"pi0.semantic.guard_apply_by_wall.{wall} must be a list")
            for guard_name in guards_for_wall:
                if str(guard_name) not in {"negation", "protect", "tutorial"}:
                    raise ValueError(
                        f"pi0.semantic.guard_apply_by_wall.{wall} values must be negation|protect|tutorial"
                    )

        prototypes = semantic_cfg.get("prototypes", {}) or {}
        for wall in (
            "override_instructions",
            "secret_exfiltration",
            "tool_or_action_abuse",
            "policy_evasion",
        ):
            pos = ((prototypes.get(wall, {}) or {}).get("positive") or [])
            if not isinstance(pos, list) or not pos:
                raise ValueError(f"pi0.semantic.prototypes.{wall}.positive must be a non-empty list")
        guards = (prototypes.get("guards", {}) or {})
        for key in ("negation", "protect", "tutorial"):
            vals = guards.get(key, []) or []
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"pi0.semantic.prototypes.guards.{key} must be a non-empty list")

    validate_projector_config(config)

    pitheta_train_cfg = config.get("pitheta_train", {}) or {}
    if pitheta_train_cfg:
        if int(pitheta_train_cfg.get("max_len", 256)) <= 0:
            raise ValueError("pitheta_train.max_len must be > 0")
        if int(pitheta_train_cfg.get("batch_size", 32)) <= 0:
            raise ValueError("pitheta_train.batch_size must be > 0")
        if float(pitheta_train_cfg.get("lr", 2e-4)) <= 0:
            raise ValueError("pitheta_train.lr must be > 0")
        if int(pitheta_train_cfg.get("epochs", 3)) <= 0:
            raise ValueError("pitheta_train.epochs must be > 0")
        lora_cfg = pitheta_train_cfg.get("lora", {}) or {}
        if lora_cfg:
            if int(lora_cfg.get("r", 16)) <= 0:
                raise ValueError("pitheta_train.lora.r must be > 0")
            if int(lora_cfg.get("alpha", 32)) <= 0:
                raise ValueError("pitheta_train.lora.alpha must be > 0")
            if float(lora_cfg.get("dropout", 0.05)) < 0:
                raise ValueError("pitheta_train.lora.dropout must be >= 0")
        loss_weights = pitheta_train_cfg.get("loss_weights", {}) or {}
        if loss_weights:
            if float(loss_weights.get("ordinal", 1.0)) <= 0:
                raise ValueError("pitheta_train.loss_weights.ordinal must be > 0")
            if float(loss_weights.get("polarity", 0.3)) < 0:
                raise ValueError("pitheta_train.loss_weights.polarity must be >= 0")
        labeling_cfg = pitheta_train_cfg.get("labeling", {}) or {}
        bins = list(labeling_cfg.get("ordinal_bins", [0.45, 1.10, 2.00]))
        if len(bins) != 3:
            raise ValueError("pitheta_train.labeling.ordinal_bins must contain 3 thresholds")
        prev_bin = -1e30
        for i, value in enumerate(bins):
            b = float(value)
            if b <= prev_bin:
                raise ValueError("pitheta_train.labeling.ordinal_bins must be strictly increasing")
            prev_bin = b
        active_floor_gold = int(labeling_cfg.get("active_floor_gold", 2))
        if active_floor_gold < 1 or active_floor_gold > 3:
            raise ValueError("pitheta_train.labeling.active_floor_gold must be in [1,3]")
        for key, expected_len in (("ordinal", 4), ("polarity", 3)):
            weight_block = (pitheta_train_cfg.get(key, {}) or {}).get("class_weights", None)
            if weight_block is None:
                continue
            if not isinstance(weight_block, list) or len(weight_block) != 4:
                raise ValueError(f"pitheta_train.{key}.class_weights must be a list of 4 lists")
            for wall_idx, wall_weights in enumerate(weight_block):
                if not isinstance(wall_weights, list) or len(wall_weights) != expected_len:
                    raise ValueError(
                        f"pitheta_train.{key}.class_weights[{wall_idx}] must have length {expected_len}"
                    )
                for weight in wall_weights:
                    if float(weight) <= 0:
                        raise ValueError(f"pitheta_train.{key}.class_weights values must be > 0")
        calibration_cfg = pitheta_train_cfg.get("calibration", {}) or {}
        if calibration_cfg:
            _ = bool(calibration_cfg.get("fit_temperature", True))
            split = str(calibration_cfg.get("temperature_split", "dev")).lower()
            if split not in {"dev", "holdout"}:
                raise ValueError("pitheta_train.calibration.temperature_split must be dev|holdout")
            out_path = str(calibration_cfg.get("temperature_output", "best/temperature_scaling.json")).strip()
            if not out_path:
                raise ValueError("pitheta_train.calibration.temperature_output must be non-empty")
        content_filter_cfg = pitheta_train_cfg.get("content_filter", {}) or {}
        if content_filter_cfg:
            if not isinstance(content_filter_cfg, dict):
                raise ValueError("pitheta_train.content_filter must be a mapping")
            mode = str(content_filter_cfg.get("mode", "off")).strip().lower()
            if mode not in {"off", "heuristic", "openai", "openai_then_heuristic"}:
                raise ValueError("pitheta_train.content_filter.mode must be off|heuristic|openai|openai_then_heuristic")
            _ = bool(content_filter_cfg.get("fail_closed", False))
            if not str(content_filter_cfg.get("api_key_env", "OPENAI_API_KEY")).strip():
                raise ValueError("pitheta_train.content_filter.api_key_env must be non-empty")
            if not str(content_filter_cfg.get("base_url", "https://api.openai.com/v1")).strip():
                raise ValueError("pitheta_train.content_filter.base_url must be non-empty")
            if not str(content_filter_cfg.get("model", "omni-moderation-latest")).strip():
                raise ValueError("pitheta_train.content_filter.model must be non-empty")
            if float(content_filter_cfg.get("timeout_sec", 20.0)) <= 0.0:
                raise ValueError("pitheta_train.content_filter.timeout_sec must be > 0")
            if int(content_filter_cfg.get("max_retries", 2)) < 0:
                raise ValueError("pitheta_train.content_filter.max_retries must be >= 0")
            if float(content_filter_cfg.get("backoff_sec", 0.75)) < 0.0:
                raise ValueError("pitheta_train.content_filter.backoff_sec must be >= 0")
            if float(content_filter_cfg.get("block_score_threshold", 0.0)) < 0.0:
                raise ValueError("pitheta_train.content_filter.block_score_threshold must be >= 0")
            apply_splits = content_filter_cfg.get("apply_splits", ["train", "dev", "holdout"])
            if not isinstance(apply_splits, list):
                raise ValueError("pitheta_train.content_filter.apply_splits must be a list")
            for split in apply_splits:
                if str(split).strip().lower() not in {"train", "dev", "holdout"}:
                    raise ValueError("pitheta_train.content_filter.apply_splits values must be train|dev|holdout")
            if "block_categories" in content_filter_cfg and not isinstance(content_filter_cfg.get("block_categories"), list):
                raise ValueError("pitheta_train.content_filter.block_categories must be a list when provided")

    pitheta_registry_cfg = config.get("pitheta_dataset_registry", {}) or {}
    if pitheta_registry_cfg:
        datasets = pitheta_registry_cfg.get("datasets", [])
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("pitheta_dataset_registry.datasets must be a non-empty list")
        sampling = pitheta_registry_cfg.get("sampling", {}) or {}
        temperature = float(sampling.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("pitheta_dataset_registry.sampling.temperature must be > 0")



def load_resolved_config(
    config_dir: Optional[str] = None,
    profile: str = "dev",
    cli_overrides: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> ConfigSnapshot:
    resolved: Dict[str, Any] = {}
    file_hashes: Dict[str, str] = {}

    use_filesystem = bool(config_dir)
    if use_filesystem:
        root = Path(str(config_dir))
        for name in _CONFIG_LAYER_ORDER:
            path = root / _CONFIG_LAYER_FILES[name]
            if path.exists():
                file_hashes[str(path.as_posix())] = _sha256_bytes(path.read_bytes())
            layer = _load_yaml(path)
            resolved = _deep_merge(resolved, layer)

        profile_path = root / "profiles" / f"{profile}.yml"
        if not profile_path.exists():
            _raise_profile_not_found(profile=profile)
        if profile_path.exists():
            file_hashes[str(profile_path.as_posix())] = _sha256_bytes(profile_path.read_bytes())
        resolved = _deep_merge(resolved, _load_yaml(profile_path))
    else:
        for name in _CONFIG_LAYER_ORDER:
            layer, source, digest = _load_bundled_yaml(_CONFIG_LAYER_FILES[name])
            if source is not None and digest is not None:
                file_hashes[source] = digest
            resolved = _deep_merge(resolved, layer)

        profile_layer, source, digest = _load_bundled_yaml("profiles", f"{profile}.yml")
        if source is None:
            profile_layer, source, digest = _load_enterprise_bundled_yaml("profiles", f"{profile}.yml")
        if source is None:
            _raise_profile_not_found(profile=profile)
        if source is not None and digest is not None:
            file_hashes[source] = digest
        resolved = _deep_merge(resolved, profile_layer)

    env_source = env or os.environ
    resolved = _apply_env_overrides(resolved, env_source)
    if cli_overrides:
        resolved = _deep_merge(resolved, cli_overrides)
    resolved = expand_omega_environment(resolved, env_source)

    validate_resolved_config(resolved)

    resolved_json = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    resolved_sha = _sha256_bytes(resolved_json)

    LOGGER.info(
        "config_snapshot",
        extra={
            "file_hashes": file_hashes,
            "resolved_sha256": resolved_sha,
        },
    )

    return ConfigSnapshot(resolved=resolved, file_hashes=file_hashes, resolved_sha256=resolved_sha)


def config_refs_from_snapshot(snapshot: ConfigSnapshot, code_commit: str = "unknown") -> Dict[str, str]:
    refs = {
        "code_commit": code_commit,
        "resolved_config_sha256": snapshot.resolved_sha256,
    }
    for path, digest in snapshot.file_hashes.items():
        base = Path(path).name.replace(".yml", "")
        refs[f"{base}_sha256"] = digest
    return refs
