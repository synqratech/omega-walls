"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib.resources as importlib_resources
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

_CONFIG_LAYER_FILES: Dict[str, str] = {
    "pi0": "pi0_defaults.yml",
    "pi0_semantic": "pi0_semantic.yml",
    "projector": "projector.yml",
    "omega": "omega_defaults.yml",
    "off_policy": "off_policy.yml",
    "source_policy": "source_policy.yml",
    "tools": "tools.yml",
    "retriever": "retriever.yml",
    "api": "api.yml",
    "monitoring": "monitoring.yml",
    "notifications": "notifications.yml",
    "bipia": "bipia.yml",
    "deepset": "deepset.yml",
    "pitheta_dataset_registry": "pitheta_dataset_registry.yml",
    "pitheta_train": "pitheta_train.yml",
    "release_gate": "release_gate.yml",
}
_CONFIG_LAYER_ORDER: Tuple[str, ...] = (
    "pi0",
    "pi0_semantic",
    "projector",
    "omega",
    "off_policy",
    "source_policy",
    "tools",
    "retriever",
    "api",
    "monitoring",
    "notifications",
    "bipia",
    "deepset",
    "pitheta_dataset_registry",
    "pitheta_train",
    "release_gate",
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any], env: Dict[str, str], prefix: str = "OMEGA__") -> Dict[str, Any]:
    """Apply env vars like OMEGA__OMEGA__EPSILON=0.2 to nested keys."""
    updated = dict(config)
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        cur: Dict[str, Any] = updated
        for part in path[:-1]:
            next_val = cur.get(part)
            if not isinstance(next_val, dict):
                next_val = {}
                cur[part] = next_val
            cur = next_val
        leaf = path[-1]
        parsed: Any = value
        for cast in (int, float):
            try:
                parsed = cast(value)
                break
            except ValueError:
                continue
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        cur[leaf] = parsed
    return updated


def validate_resolved_config(config: Dict[str, Any]) -> None:
    walls = config["omega"]["walls"]
    if walls != [
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    ]:
        raise ValueError("Wall ordering mismatch with v1 contract")

    gamma_omega = config["omega"]["attribution"]["gamma"]
    gamma_policy = config["off_policy"]["block"]["gamma"]
    if abs(float(gamma_omega) - float(gamma_policy)) > 1e-9:
        raise ValueError("gamma mismatch between omega.attribution and off_policy.block")

    enforcement_mode = str(config["off_policy"].get("enforcement_mode", "ENFORCE")).upper()
    if enforcement_mode not in {"ENFORCE", "LOG_ONLY"}:
        raise ValueError("off_policy.enforcement_mode must be ENFORCE or LOG_ONLY")
    control_outcome_cfg = (config.get("off_policy", {}) or {}).get("control_outcome", {}) or {}
    if control_outcome_cfg and not isinstance(control_outcome_cfg, dict):
        raise ValueError("off_policy.control_outcome must be a mapping")
    if isinstance(control_outcome_cfg, dict) and control_outcome_cfg:
        warn_cfg = control_outcome_cfg.get("warn", {}) or {}
        if warn_cfg and not isinstance(warn_cfg, dict):
            raise ValueError("off_policy.control_outcome.warn must be a mapping")
        if isinstance(warn_cfg, dict) and warn_cfg:
            _ = bool(warn_cfg.get("enabled", False))
            if not str(warn_cfg.get("target", "SESSION")).strip():
                raise ValueError("off_policy.control_outcome.warn.target must be non-empty")
            if float(warn_cfg.get("max_p_gte", 0.0)) < 0.0:
                raise ValueError("off_policy.control_outcome.warn.max_p_gte must be >= 0")
            if float(warn_cfg.get("sum_m_next_gte", 0.0)) < 0.0:
                raise ValueError("off_policy.control_outcome.warn.sum_m_next_gte must be >= 0")
        req_cfg = control_outcome_cfg.get("require_approval", {}) or {}
        if req_cfg and not isinstance(req_cfg, dict):
            raise ValueError("off_policy.control_outcome.require_approval must be a mapping")
        if isinstance(req_cfg, dict) and req_cfg:
            _ = bool(req_cfg.get("enabled", False))
            _ = bool(req_cfg.get("on_off", True))
            _ = bool(req_cfg.get("on_warn", True))
            tools = req_cfg.get("tools", [])
            if not isinstance(tools, list):
                raise ValueError("off_policy.control_outcome.require_approval.tools must be a list")
            if int(req_cfg.get("horizon_steps", 0)) < 0:
                raise ValueError("off_policy.control_outcome.require_approval.horizon_steps must be >= 0")
    incident_artifact_cfg = (config.get("off_policy", {}) or {}).get("incident_artifact", {}) or {}
    if incident_artifact_cfg and not isinstance(incident_artifact_cfg, dict):
        raise ValueError("off_policy.incident_artifact must be a mapping")
    if isinstance(incident_artifact_cfg, dict) and incident_artifact_cfg:
        _ = bool(incident_artifact_cfg.get("enabled", False))
        _ = bool(incident_artifact_cfg.get("include_timeline", True))
        emit_for = incident_artifact_cfg.get("emit_for_outcomes", [])
        if emit_for is not None and not isinstance(emit_for, list):
            raise ValueError("off_policy.incident_artifact.emit_for_outcomes must be a list")

    source_policy = config.get("source_policy", {})
    default_trust = source_policy.get("default_trust", "untrusted")
    valid_trust = {"trusted", "semi", "untrusted", "semi_trusted"}
    if default_trust not in valid_trust:
        raise ValueError("source_policy.default_trust must be trusted|semi|semi_trusted|untrusted")

    tools_cfg = config.get("tools", {})
    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).upper()
    if execution_mode not in {"ENFORCE", "DRY_RUN"}:
        raise ValueError("tools.execution_mode must be ENFORCE or DRY_RUN")
    arg_validation_cfg = tools_cfg.get("arg_validation", {}) or {}
    if arg_validation_cfg and not isinstance(arg_validation_cfg, dict):
        raise ValueError("tools.arg_validation must be a mapping")
    if isinstance(arg_validation_cfg, dict) and arg_validation_cfg:
        _ = bool(arg_validation_cfg.get("enabled", True))
        fail_mode = str(arg_validation_cfg.get("fail_mode", "deny")).strip().lower()
        if fail_mode != "deny":
            raise ValueError("tools.arg_validation.fail_mode must be deny")
        shell_patterns = arg_validation_cfg.get("shell_like_name_patterns", [])
        if shell_patterns is not None and not isinstance(shell_patterns, list):
            raise ValueError("tools.arg_validation.shell_like_name_patterns must be a list")
        net_cfg = arg_validation_cfg.get("network_post", {}) or {}
        wr_cfg = arg_validation_cfg.get("write_file", {}) or {}
        sh_cfg = arg_validation_cfg.get("shell_like", {}) or {}
        for key in ("max_payload_bytes", "max_headers", "max_header_key_chars", "max_header_value_chars"):
            if int(net_cfg.get(key, 1)) <= 0:
                raise ValueError(f"tools.arg_validation.network_post.{key} must be > 0")
        for key in ("max_filename_chars", "max_content_bytes"):
            if int(wr_cfg.get(key, 1)) <= 0:
                raise ValueError(f"tools.arg_validation.write_file.{key} must be > 0")
        if int(sh_cfg.get("max_command_chars", 1)) <= 0:
            raise ValueError("tools.arg_validation.shell_like.max_command_chars must be > 0")
        destructive_patterns = sh_cfg.get("destructive_patterns", [])
        if destructive_patterns is not None and not isinstance(destructive_patterns, list):
            raise ValueError("tools.arg_validation.shell_like.destructive_patterns must be a list")

    runtime_cfg = config.get("runtime", {}) or {}
    if runtime_cfg and not isinstance(runtime_cfg, dict):
        raise ValueError("runtime must be a mapping")
    guard_mode = str(runtime_cfg.get("guard_mode", "enforce")).strip().lower()
    if guard_mode not in {"enforce", "monitor"}:
        raise ValueError("runtime.guard_mode must be enforce|monitor")

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

    cross_session_cfg = config.get("off_policy", {}).get("cross_session", {})
    if cross_session_cfg:
        backend = str(cross_session_cfg.get("backend", "sqlite")).lower()
        if backend != "sqlite":
            raise ValueError("off_policy.cross_session.backend must be sqlite in v1")
        decay_mode = str(cross_session_cfg.get("decay", {}).get("mode", "exponential")).lower()
        if decay_mode != "exponential":
            raise ValueError("off_policy.cross_session.decay.mode must be exponential in v1")
        half_life = float(cross_session_cfg.get("decay", {}).get("half_life_steps", 120))
        if half_life <= 0:
            raise ValueError("off_policy.cross_session.decay.half_life_steps must be > 0")

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

    api_cfg = config.get("api", {}) or {}
    if api_cfg:
        host = str(api_cfg.get("host", "127.0.0.1")).strip()
        if not host:
            raise ValueError("api.host must be non-empty")
        port = int(api_cfg.get("port", 8080))
        if port <= 0 or port > 65535:
            raise ValueError("api.port must be in 1..65535")
        auth_cfg = api_cfg.get("auth", {}) or {}
        if auth_cfg and not isinstance(auth_cfg, dict):
            raise ValueError("api.auth must be a mapping")
        api_keys = auth_cfg.get("api_keys", [])
        if not isinstance(api_keys, list):
            raise ValueError("api.auth.api_keys must be a list")
        _ = bool(auth_cfg.get("require_hmac", True))
        hmac_secret_env = str(auth_cfg.get("hmac_secret_env", "OMEGA_API_HMAC_SECRET")).strip()
        if not hmac_secret_env:
            raise ValueError("api.auth.hmac_secret_env must be non-empty")
        hmac_headers = auth_cfg.get("hmac_headers", {}) or {}
        if hmac_headers and not isinstance(hmac_headers, dict):
            raise ValueError("api.auth.hmac_headers must be a mapping")
        for key in ("signature", "timestamp", "nonce"):
            if not str(hmac_headers.get(key, f"X-{key.title()}")).strip():
                raise ValueError(f"api.auth.hmac_headers.{key} must be non-empty")
        if int(auth_cfg.get("max_clock_skew_sec", 300)) <= 0:
            raise ValueError("api.auth.max_clock_skew_sec must be > 0")
        if int(auth_cfg.get("replay_nonce_ttl_sec", 600)) <= 0:
            raise ValueError("api.auth.replay_nonce_ttl_sec must be > 0")
        if int(auth_cfg.get("replay_cache_max_entries", 100000)) <= 0:
            raise ValueError("api.auth.replay_cache_max_entries must be > 0")
        security_cfg = api_cfg.get("security", {}) or {}
        if security_cfg and not isinstance(security_cfg, dict):
            raise ValueError("api.security must be a mapping")
        transport_mode = str(security_cfg.get("transport_mode", "proxy_tls")).strip().lower()
        if transport_mode not in {"proxy_tls", "disabled"}:
            raise ValueError("api.security.transport_mode must be proxy_tls|disabled")
        _ = bool(security_cfg.get("require_https", True))
        limits_cfg = api_cfg.get("limits", {}) or {}
        if limits_cfg and not isinstance(limits_cfg, dict):
            raise ValueError("api.limits must be a mapping")
        if int(limits_cfg.get("max_file_bytes", 20 * 1024 * 1024)) <= 0:
            raise ValueError("api.limits.max_file_bytes must be > 0")
        if int(limits_cfg.get("max_extracted_text_chars", 200_000)) <= 0:
            raise ValueError("api.limits.max_extracted_text_chars must be > 0")
        if int(limits_cfg.get("request_timeout_sec", 15)) <= 0:
            raise ValueError("api.limits.request_timeout_sec must be > 0")
        logging_cfg = api_cfg.get("logging", {}) or {}
        if logging_cfg and not isinstance(logging_cfg, dict):
            raise ValueError("api.logging must be a mapping")
        _ = bool(logging_cfg.get("enabled", True))
        _ = bool(logging_cfg.get("include_policy_trace", True))
        debug_cfg = api_cfg.get("debug", {}) or {}
        if debug_cfg and not isinstance(debug_cfg, dict):
            raise ValueError("api.debug must be a mapping")
        _ = bool(debug_cfg.get("enable_document_scan_report", False))
        if int(debug_cfg.get("max_report_chunks", 200)) <= 0:
            raise ValueError("api.debug.max_report_chunks must be > 0")
        chunk_cfg = api_cfg.get("chunk_pipeline", {}) or {}
        if chunk_cfg and not isinstance(chunk_cfg, dict):
            raise ValueError("api.chunk_pipeline must be a mapping")
        if isinstance(chunk_cfg, dict) and chunk_cfg:
            wall_thr = float(chunk_cfg.get("wall_trigger_threshold", 0.12))
            if wall_thr < 0.0 or wall_thr > 1.0:
                raise ValueError("api.chunk_pipeline.wall_trigger_threshold must be in [0,1]")
            for key in ("worst_weight", "synergy_weight", "confidence_weight"):
                if float(chunk_cfg.get(key, 0.0)) < 0.0:
                    raise ValueError(f"api.chunk_pipeline.{key} must be >= 0")
            for key in (
                "synergy_pair_bonus",
                "synergy_multiwall_bonus",
                "synergy_pattern_bonus",
                "confidence_margin",
                "confidence_support_threshold",
            ):
                if float(chunk_cfg.get(key, 0.0)) < 0.0:
                    raise ValueError(f"api.chunk_pipeline.{key} must be >= 0")
            if int(chunk_cfg.get("confidence_support_chunks", 1)) <= 0:
                raise ValueError("api.chunk_pipeline.confidence_support_chunks must be > 0")
            if int(chunk_cfg.get("top_chunks_limit", 1)) <= 0:
                raise ValueError("api.chunk_pipeline.top_chunks_limit must be > 0")
            synergy_pairs = chunk_cfg.get("synergy_pairs", [])
            if synergy_pairs is not None and not isinstance(synergy_pairs, list):
                raise ValueError("api.chunk_pipeline.synergy_pairs must be a list")
            if isinstance(synergy_pairs, list):
                for idx, pair in enumerate(synergy_pairs):
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError(f"api.chunk_pipeline.synergy_pairs[{idx}] must be [wall_a, wall_b]")
        policy_mapper_cfg = api_cfg.get("policy_mapper", {}) or {}
        if policy_mapper_cfg and not isinstance(policy_mapper_cfg, dict):
            raise ValueError("api.policy_mapper must be a mapping")
        if isinstance(policy_mapper_cfg, dict) and policy_mapper_cfg:
            for key in (
                "block_score_threshold",
                "quarantine_score_threshold",
                "quarantine_worst_threshold",
                "quarantine_synergy_threshold",
                "exfil_block_wall_threshold",
                "confidence_block_threshold",
            ):
                value = float(policy_mapper_cfg.get(key, 0.0))
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"api.policy_mapper.{key} must be in [0,1]")
            hgl_cfg = policy_mapper_cfg.get("hallucination_guard_lite", {}) or {}
            if hgl_cfg and not isinstance(hgl_cfg, dict):
                raise ValueError("api.policy_mapper.hallucination_guard_lite must be a mapping")
            if isinstance(hgl_cfg, dict) and hgl_cfg:
                _ = bool(hgl_cfg.get("enabled", False))
                bands = hgl_cfg.get("apply_when_source_trust", ["untrusted", "mixed"])
                if bands is not None and not isinstance(bands, list):
                    raise ValueError("api.policy_mapper.hallucination_guard_lite.apply_when_source_trust must be a list")
                valid_bands = {"trusted", "untrusted", "mixed"}
                for idx, band in enumerate(list(bands or [])):
                    norm = str(band).strip().lower()
                    if norm == "semi":
                        norm = "trusted"
                    if norm == "semi_trusted":
                        norm = "trusted"
                    if norm not in valid_bands:
                        raise ValueError(
                            "api.policy_mapper.hallucination_guard_lite.apply_when_source_trust"
                            f"[{idx}] must be trusted|untrusted|mixed"
                        )
                low_conf = float(hgl_cfg.get("low_confidence_lte", 0.35))
                if low_conf < 0.0 or low_conf > 1.0:
                    raise ValueError("api.policy_mapper.hallucination_guard_lite.low_confidence_lte must be in [0,1]")
                _ = bool(hgl_cfg.get("only_if_intended_allow", True))
                soft_q_cfg = hgl_cfg.get("soft_quarantine", {}) or {}
                if soft_q_cfg and not isinstance(soft_q_cfg, dict):
                    raise ValueError("api.policy_mapper.hallucination_guard_lite.soft_quarantine must be a mapping")
                if isinstance(soft_q_cfg, dict) and soft_q_cfg:
                    _ = bool(soft_q_cfg.get("enabled", False))
                    _ = bool(soft_q_cfg.get("mixed_only", True))
                    very_low = float(soft_q_cfg.get("very_low_confidence_lte", 0.20))
                    if very_low < 0.0 or very_low > 1.0:
                        raise ValueError(
                            "api.policy_mapper.hallucination_guard_lite.soft_quarantine.very_low_confidence_lte "
                            "must be in [0,1]"
                        )
                    pattern_synergy = float(soft_q_cfg.get("pattern_synergy_gte", 0.30))
                    if pattern_synergy < 0.0 or pattern_synergy > 1.0:
                        raise ValueError(
                            "api.policy_mapper.hallucination_guard_lite.soft_quarantine.pattern_synergy_gte "
                            "must be in [0,1]"
                        )
        att_cfg = api_cfg.get("attestation", {}) or {}
        if att_cfg and not isinstance(att_cfg, dict):
            raise ValueError("api.attestation must be a mapping")
        if bool(att_cfg.get("enabled", False)):
            fmt = str(att_cfg.get("format", "jws")).strip().lower()
            if fmt != "jws":
                raise ValueError("api.attestation.format must be jws")
            alg = str(att_cfg.get("alg", "RS256")).strip().upper()
            if alg != "RS256":
                raise ValueError("api.attestation.alg must be RS256")
            if not str(att_cfg.get("kid", "omega-attestation-v1")).strip():
                raise ValueError("api.attestation.kid must be non-empty")
            if not str(att_cfg.get("private_key_pem_env", "OMEGA_API_ATTESTATION_PRIVATE_KEY")).strip():
                raise ValueError("api.attestation.private_key_pem_env must be non-empty")
            if int(att_cfg.get("exp_sec", 300)) <= 0:
                raise ValueError("api.attestation.exp_sec must be > 0")

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

    bipia_cfg = config.get("bipia", {})
    if bipia_cfg:
        mode = str(bipia_cfg.get("mode_default", "sampled")).lower()
        if mode not in {"sampled", "full"}:
            raise ValueError("bipia.mode_default must be sampled|full")
        split = str(bipia_cfg.get("split_default", "test")).lower()
        if split != "test":
            raise ValueError("bipia.split_default must be test in v1")
        sampled = bipia_cfg.get("sampled", {})
        max_contexts = int(sampled.get("max_contexts_per_task", 20))
        max_attacks = int(sampled.get("max_attacks_per_task", 10))
        if max_contexts <= 0:
            raise ValueError("bipia.sampled.max_contexts_per_task must be > 0")
        if max_attacks <= 0:
            raise ValueError("bipia.sampled.max_attacks_per_task must be > 0")
        thresholds = bipia_cfg.get("thresholds", {}).get("sampled", {})
        for key in ("attack_off_rate_ge", "per_task_attack_off_rate_ge", "coverage_wall_any_ge"):
            if float(thresholds.get(key, 0.0)) < 0.0:
                raise ValueError(f"bipia.thresholds.sampled.{key} must be >= 0")

    deepset_cfg = config.get("deepset", {})
    if deepset_cfg:
        mode = str(deepset_cfg.get("mode_default", "full")).lower()
        if mode not in {"sampled", "full"}:
            raise ValueError("deepset.mode_default must be sampled|full")
        split = str(deepset_cfg.get("split_default", "test")).lower()
        if split not in {"train", "test"}:
            raise ValueError("deepset.split_default must be train|test")
        label_attack = int(deepset_cfg.get("label_attack_value", 1))
        if label_attack not in {0, 1}:
            raise ValueError("deepset.label_attack_value must be 0|1")
        sampled = deepset_cfg.get("sampled", {}) or {}
        max_samples = int(sampled.get("max_samples", 116))
        if max_samples <= 0:
            raise ValueError("deepset.sampled.max_samples must be > 0")
        thresholds = (deepset_cfg.get("thresholds", {}) or {}).get("report", {}) or {}
        for key in ("attack_off_rate_ge", "coverage_wall_any_attack_ge", "f1_ge"):
            val = float(thresholds.get(key, 0.0))
            if val < 0.0 or val > 1.0:
                raise ValueError(f"deepset.thresholds.report.{key} must be in [0,1]")
        benign_off = float(thresholds.get("benign_off_rate_le", 1.0))
        if benign_off < 0.0 or benign_off > 1.0:
            raise ValueError("deepset.thresholds.report.benign_off_rate_le must be in [0,1]")
        repro = deepset_cfg.get("reproducibility", {}) or {}
        if int(repro.get("seed_default", 41)) < 0:
            raise ValueError("deepset.reproducibility.seed_default must be >= 0")

    release_gate_cfg = config.get("release_gate", {})
    if release_gate_cfg:
        gates = release_gate_cfg.get("gates", [])
        if not isinstance(gates, list):
            raise ValueError("release_gate.gates must be a list")
        allowed_ops = {"eq", "ge", "le", "is_null", "not_null"}
        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("release_gate.gates entries must be mappings")
            gate_id = str(gate.get("id", "")).strip()
            metric = str(gate.get("metric", "")).strip()
            op = str(gate.get("op", "")).strip().lower()
            if not gate_id:
                raise ValueError("release_gate gate id must be non-empty")
            if not metric:
                raise ValueError(f"release_gate {gate_id} metric must be non-empty")
            if op not in allowed_ops:
                raise ValueError(f"release_gate {gate_id} op must be one of {sorted(allowed_ops)}")

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

    projector_cfg = config.get("projector", {}) or {}
    if projector_cfg:
        mode = str(projector_cfg.get("mode", "pi0")).lower()
        if mode not in {"pi0", "pitheta", "hybrid", "hybrid_api"}:
            raise ValueError("projector.mode must be pi0|pitheta|hybrid|hybrid_api")
        api_cfg = projector_cfg.get("api_perception", {}) or {}
        if api_cfg:
            enabled = str(api_cfg.get("enabled", "auto")).lower()
            if enabled not in {"auto", "true", "false"}:
                raise ValueError("projector.api_perception.enabled must be auto|true|false")
            if not str(api_cfg.get("model", "gpt-5")).strip():
                raise ValueError("projector.api_perception.model must be non-empty")
            if not str(api_cfg.get("base_url", "https://api.openai.com/v1")).strip():
                raise ValueError("projector.api_perception.base_url must be non-empty")
            if not str(api_cfg.get("api_key_env", "OPENAI_API_KEY")).strip():
                raise ValueError("projector.api_perception.api_key_env must be non-empty")
            if float(api_cfg.get("timeout_sec", 30.0)) <= 0.0:
                raise ValueError("projector.api_perception.timeout_sec must be > 0")
            if int(api_cfg.get("max_retries", 2)) < 0:
                raise ValueError("projector.api_perception.max_retries must be >= 0")
            if float(api_cfg.get("backoff_sec", 0.75)) < 0.0:
                raise ValueError("projector.api_perception.backoff_sec must be >= 0")
            if float(api_cfg.get("retry_backoff_max_sec", 2.0)) < 0.0:
                raise ValueError("projector.api_perception.retry_backoff_max_sec must be >= 0")
            if float(api_cfg.get("request_deadline_sec", 20.0)) <= 0.0:
                raise ValueError("projector.api_perception.request_deadline_sec must be > 0")
            if int(api_cfg.get("long_text_threshold_chars", 3000)) <= 0:
                raise ValueError("projector.api_perception.long_text_threshold_chars must be > 0")
            if int(api_cfg.get("long_text_max_retries", 1)) < 0:
                raise ValueError("projector.api_perception.long_text_max_retries must be >= 0")
            short_thr = int(api_cfg.get("short_text_threshold_chars", 1200))
            if short_thr <= 0:
                raise ValueError("projector.api_perception.short_text_threshold_chars must be > 0")
            _ = bool(api_cfg.get("short_prefer_chat_completions", True))
            _ = bool(api_cfg.get("short_chat_only", True))
            _ = bool(api_cfg.get("short_fast_path_enabled", True))
            _ = bool(api_cfg.get("short_fast_path_skip_on_pi0_hard", True))
            _ = bool(api_cfg.get("short_fast_path_skip_on_pi0_clean", True))
            hard_min = float(api_cfg.get("short_fast_path_hard_min_score", 0.55))
            clean_max = float(api_cfg.get("short_fast_path_clean_max_score", 0.0))
            if hard_min < 0.0 or hard_min > 1.0:
                raise ValueError("projector.api_perception.short_fast_path_hard_min_score must be in [0,1]")
            if clean_max < 0.0 or clean_max > 1.0:
                raise ValueError("projector.api_perception.short_fast_path_clean_max_score must be in [0,1]")
            if clean_max > hard_min:
                raise ValueError(
                    "projector.api_perception.short_fast_path_clean_max_score must be <= short_fast_path_hard_min_score"
                )
            _ = bool(api_cfg.get("prewarm_on_init", True))
            if float(api_cfg.get("transient_error_ttl_sec", 90.0)) < 0.0:
                raise ValueError("projector.api_perception.transient_error_ttl_sec must be >= 0")
            if float(api_cfg.get("responses_cooldown_sec", 60.0)) < 0.0:
                raise ValueError("projector.api_perception.responses_cooldown_sec must be >= 0")
            if not str(api_cfg.get("prompt_version", "api_hybrid_v1")).strip():
                raise ValueError("projector.api_perception.prompt_version must be non-empty")
            if "cache_path" in api_cfg and not str(api_cfg.get("cache_path", "")).strip():
                raise ValueError("projector.api_perception.cache_path must be non-empty when provided")
            if "error_log_path" in api_cfg and not str(api_cfg.get("error_log_path", "")).strip():
                raise ValueError("projector.api_perception.error_log_path must be non-empty when provided")
        pitheta_cfg = projector_cfg.get("pitheta", {}) or {}
        if pitheta_cfg:
            if int(pitheta_cfg.get("max_length", 256)) <= 0:
                raise ValueError("projector.pitheta.max_length must be > 0")
            if int(pitheta_cfg.get("batch_size", 8)) <= 0:
                raise ValueError("projector.pitheta.batch_size must be > 0")
            head_mode = str(pitheta_cfg.get("head_mode", "auto")).lower()
            if head_mode not in {"auto", "legacy", "ordinal"}:
                raise ValueError("projector.pitheta.head_mode must be auto|legacy|ordinal")
            conversion_mode = str(pitheta_cfg.get("conversion_mode", "expected")).lower()
            if conversion_mode not in {"expected", "argmax"}:
                raise ValueError("projector.pitheta.conversion_mode must be expected|argmax")
            pressure_map = list(pitheta_cfg.get("pressure_map", [0.0, 0.25, 0.6, 1.0]))
            if len(pressure_map) != 4:
                raise ValueError("projector.pitheta.pressure_map must have 4 values")
            last = -1.0
            for i, value in enumerate(pressure_map):
                v = float(value)
                if v < 0.0:
                    raise ValueError(f"projector.pitheta.pressure_map[{i}] must be >= 0")
                if i > 0 and v < last:
                    raise ValueError("projector.pitheta.pressure_map must be non-decreasing")
                last = v
            _ = bool(pitheta_cfg.get("require_calibration", True))
            if "calibration_file" in pitheta_cfg and not str(pitheta_cfg.get("calibration_file", "")).strip():
                raise ValueError("projector.pitheta.calibration_file must be non-empty when provided")
            thresholds = (pitheta_cfg.get("legacy", {}) or {}).get("wall_thresholds", pitheta_cfg.get("wall_thresholds", {})) or {}
            for wall in (
                "override_instructions",
                "secret_exfiltration",
                "tool_or_action_abuse",
                "policy_evasion",
            ):
                val = float(thresholds.get(wall, 0.5))
                if val < 0.0 or val > 1.0:
                    raise ValueError(f"projector.pitheta.wall_thresholds.{wall} must be in [0,1]")

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
        if source is not None and digest is not None:
            file_hashes[source] = digest
        resolved = _deep_merge(resolved, profile_layer)

    resolved = _apply_env_overrides(resolved, env or os.environ)
    if cli_overrides:
        resolved = _deep_merge(resolved, cli_overrides)

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

