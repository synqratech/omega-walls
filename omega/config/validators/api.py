from __future__ import annotations

from typing import Any, Dict
import ipaddress


_PRODUCTION_PROFILE_ENVS = {
    "prod",
    "production",
    "prod_api",
    "prod_vision",
    "prod_vision_local_ocr",
    "prod_enterprise",
    "prod_vision_enterprise",
}


def validate_api_config(config: Dict[str, Any]) -> None:
    api_cfg = config.get("api", {}) or {}
    if not api_cfg:
        return

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
    if any(not isinstance(value, str) for value in api_keys):
        raise ValueError("api.auth.api_keys values must be strings")
    api_key_env = str(auth_cfg.get("api_key_env", "OMEGA_API_KEYS")).strip()
    if not api_key_env:
        raise ValueError("api.auth.api_key_env must be non-empty")
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
    if transport_mode not in {"proxy_tls", "direct_tls", "disabled"}:
        raise ValueError("api.security.transport_mode must be proxy_tls|direct_tls|disabled")
    require_https = bool(security_cfg.get("require_https", True))
    trusted_proxy_cidrs = security_cfg.get("trusted_proxy_cidrs", [])
    if not isinstance(trusted_proxy_cidrs, list):
        raise ValueError("api.security.trusted_proxy_cidrs must be a list")
    for idx, cidr in enumerate(trusted_proxy_cidrs):
        try:
            ipaddress.ip_network(str(cidr), strict=False)
        except ValueError as exc:
            raise ValueError(f"api.security.trusted_proxy_cidrs[{idx}] is invalid") from exc

    profile_env = str((config.get("profiles", {}) or {}).get("env", "")).strip().lower()
    if profile_env in _PRODUCTION_PROFILE_ENVS:
        weak_keys = {
            "dev-api-key",
            "test-api-key",
            "quickstart-api-key",
            "changeme",
            "change-me",
            "secret",
        }
        configured = [str(value).strip() for value in api_keys if str(value).strip()]
        bad = [value for value in configured if value.lower() in weak_keys or len(value) < 24]
        if bad:
            raise ValueError("production API cannot use development, placeholder, or short credentials")
        runtime_cfg = api_cfg.get("runtime", {}) or {}
        if bool(runtime_cfg.get("allow_request_override", False)):
            raise ValueError("production api.runtime.allow_request_override must be false")
        if not bool(auth_cfg.get("require_hmac", True)):
            raise ValueError("production api.auth.require_hmac must be true")
        if transport_mode not in {"proxy_tls", "direct_tls"} or not require_https:
            raise ValueError("production API requires proxy_tls/direct_tls and HTTPS")
        if transport_mode == "proxy_tls" and not trusted_proxy_cidrs:
            raise ValueError("production proxy_tls requires explicit trusted_proxy_cidrs")
    limits_cfg = api_cfg.get("limits", {}) or {}
    if limits_cfg and not isinstance(limits_cfg, dict):
        raise ValueError("api.limits must be a mapping")
    if int(limits_cfg.get("max_file_bytes", 20 * 1024 * 1024)) <= 0:
        raise ValueError("api.limits.max_file_bytes must be > 0")
    if int(limits_cfg.get("max_extracted_text_chars", 200_000)) <= 0:
        raise ValueError("api.limits.max_extracted_text_chars must be > 0")
    if int(limits_cfg.get("request_timeout_sec", 15)) <= 0:
        raise ValueError("api.limits.request_timeout_sec must be > 0")
    for key, default in (
        ("max_request_body_bytes", 24 * 1024 * 1024),
        ("max_multipart_files", 1),
        ("max_multipart_fields", 16),
        ("max_multipart_part_bytes", 20 * 1024 * 1024),
    ):
        if int(limits_cfg.get(key, default)) <= 0:
            raise ValueError(f"api.limits.{key} must be > 0")
    if int(limits_cfg.get("max_multipart_part_bytes", 20 * 1024 * 1024)) > int(limits_cfg.get("max_request_body_bytes", 24 * 1024 * 1024)):
        raise ValueError("api.limits.max_multipart_part_bytes cannot exceed max_request_body_bytes")
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
    incident_export_cfg = api_cfg.get("incident_export", {}) or {}
    if incident_export_cfg and not isinstance(incident_export_cfg, dict):
        raise ValueError("api.incident_export must be a mapping")
    if isinstance(incident_export_cfg, dict) and incident_export_cfg:
        _ = bool(incident_export_cfg.get("enabled", False))
        if not str(incident_export_cfg.get("contract_version", "1.0")).strip():
            raise ValueError("api.incident_export.contract_version must be non-empty")
        default_env = str(incident_export_cfg.get("default_environment", "staging")).strip().lower()
        if default_env not in {"dev", "staging", "prod"}:
            raise ValueError("api.incident_export.default_environment must be dev|staging|prod")
        if int(incident_export_cfg.get("retention_days", 30)) <= 0:
            raise ValueError("api.incident_export.retention_days must be > 0")
        store_cfg = incident_export_cfg.get("store", {}) or {}
        if store_cfg and not isinstance(store_cfg, dict):
            raise ValueError("api.incident_export.store must be a mapping")
        if not str(store_cfg.get("sqlite_path", "artifacts/state/incident_export.db")).strip():
            raise ValueError("api.incident_export.store.sqlite_path must be non-empty")
        auth_cfg_ie = incident_export_cfg.get("auth", {}) or {}
        if auth_cfg_ie and not isinstance(auth_cfg_ie, dict):
            raise ValueError("api.incident_export.auth must be a mapping")
        if not str(auth_cfg_ie.get("key_store_path", "artifacts/state/incident_export_keys.db")).strip():
            raise ValueError("api.incident_export.auth.key_store_path must be non-empty")
        if not str(auth_cfg_ie.get("required_scope", "incidents:read")).strip():
            raise ValueError("api.incident_export.auth.required_scope must be non-empty")
        rl_cfg = incident_export_cfg.get("rate_limit", {}) or {}
        if rl_cfg and not isinstance(rl_cfg, dict):
            raise ValueError("api.incident_export.rate_limit must be a mapping")
        if int(rl_cfg.get("rpm", 60)) <= 0:
            raise ValueError("api.incident_export.rate_limit.rpm must be > 0")
        if int(rl_cfg.get("burst", 10)) <= 0:
            raise ValueError("api.incident_export.rate_limit.burst must be > 0")
        cors_cfg = incident_export_cfg.get("cors", {}) or {}
        if cors_cfg and not isinstance(cors_cfg, dict):
            raise ValueError("api.incident_export.cors must be a mapping")
        allowed_origins = cors_cfg.get("allowed_origins", [])
        if allowed_origins is not None and not isinstance(allowed_origins, list):
            raise ValueError("api.incident_export.cors.allowed_origins must be a list")
    incident_replay_cfg = api_cfg.get("incident_replay", {}) or {}
    if incident_replay_cfg and not isinstance(incident_replay_cfg, dict):
        raise ValueError("api.incident_replay must be a mapping")
    if isinstance(incident_replay_cfg, dict) and incident_replay_cfg:
        _ = bool(incident_replay_cfg.get("enabled", False))
        if not str(incident_replay_cfg.get("contract_version", "1.0.0")).strip():
            raise ValueError("api.incident_replay.contract_version must be non-empty")
        if int(incident_replay_cfg.get("download_ttl_hours", 24)) <= 0:
            raise ValueError("api.incident_replay.download_ttl_hours must be > 0")
        if int(incident_replay_cfg.get("job_ttl_hours", 72)) <= 0:
            raise ValueError("api.incident_replay.job_ttl_hours must be > 0")
        max_steps = int(incident_replay_cfg.get("max_steps", 50))
        if max_steps <= 0 or max_steps > 50:
            raise ValueError("api.incident_replay.max_steps must be in 1..50")
        store_cfg = incident_replay_cfg.get("store", {}) or {}
        if store_cfg and not isinstance(store_cfg, dict):
            raise ValueError("api.incident_replay.store must be a mapping")
        if not str(store_cfg.get("sqlite_path", "artifacts/state/incident_replay.db")).strip():
            raise ValueError("api.incident_replay.store.sqlite_path must be non-empty")
        package_cfg = incident_replay_cfg.get("package_storage", {}) or {}
        if package_cfg and not isinstance(package_cfg, dict):
            raise ValueError("api.incident_replay.package_storage must be a mapping")
        if not str(package_cfg.get("path", "artifacts/replay/packages")).strip():
            raise ValueError("api.incident_replay.package_storage.path must be non-empty")
        if not str(package_cfg.get("encryption_key_env", "OMEGA_REPLAY_ENCRYPTION_KEY")).strip():
            raise ValueError("api.incident_replay.package_storage.encryption_key_env must be non-empty")
        worker_cfg = incident_replay_cfg.get("worker", {}) or {}
        if worker_cfg and not isinstance(worker_cfg, dict):
            raise ValueError("api.incident_replay.worker must be a mapping")
        if int(worker_cfg.get("max_concurrent_jobs", 4)) <= 0:
            raise ValueError("api.incident_replay.worker.max_concurrent_jobs must be > 0")
        auth_cfg_ir = incident_replay_cfg.get("auth", {}) or {}
        if auth_cfg_ir and not isinstance(auth_cfg_ir, dict):
            raise ValueError("api.incident_replay.auth must be a mapping")
        scopes_cfg = auth_cfg_ir.get("required_scopes", {}) or {}
        if scopes_cfg and not isinstance(scopes_cfg, dict):
            raise ValueError("api.incident_replay.auth.required_scopes must be a mapping")
        if not str(scopes_cfg.get("read", "incidents:replay:read")).strip():
            raise ValueError("api.incident_replay.auth.required_scopes.read must be non-empty")
        if not str(scopes_cfg.get("raw", "incidents:replay:raw")).strip():
            raise ValueError("api.incident_replay.auth.required_scopes.raw must be non-empty")
