"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfigSnapshot:
    resolved: Dict[str, Any]
    file_hashes: Dict[str, str]
    resolved_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_bytes()
    parsed = yaml.safe_load(content) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return parsed


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

    source_policy = config.get("source_policy", {})
    default_trust = source_policy.get("default_trust", "untrusted")
    valid_trust = {"trusted", "semi", "untrusted", "semi_trusted"}
    if default_trust not in valid_trust:
        raise ValueError("source_policy.default_trust must be trusted|semi|semi_trusted|untrusted")

    tools_cfg = config.get("tools", {})
    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).upper()
    if execution_mode not in {"ENFORCE", "DRY_RUN"}:
        raise ValueError("tools.execution_mode must be ENFORCE or DRY_RUN")

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
        if mode not in {"pi0", "pitheta", "hybrid"}:
            raise ValueError("projector.mode must be pi0|pitheta|hybrid")
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
    config_dir: str = "config",
    profile: str = "dev",
    cli_overrides: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> ConfigSnapshot:
    root = Path(config_dir)
    files = {
        "pi0": root / "pi0_defaults.yml",
        "pi0_semantic": root / "pi0_semantic.yml",
        "projector": root / "projector.yml",
        "omega": root / "omega_defaults.yml",
        "off_policy": root / "off_policy.yml",
        "source_policy": root / "source_policy.yml",
        "tools": root / "tools.yml",
        "retriever": root / "retriever.yml",
        "api": root / "api.yml",
        "bipia": root / "bipia.yml",
        "deepset": root / "deepset.yml",
        "pitheta_dataset_registry": root / "pitheta_dataset_registry.yml",
        "pitheta_train": root / "pitheta_train.yml",
        "release_gate": root / "release_gate.yml",
        "profile": root / "profiles" / f"{profile}.yml",
    }

    resolved: Dict[str, Any] = {}
    file_hashes: Dict[str, str] = {}

    for name in (
        "pi0",
        "pi0_semantic",
        "projector",
        "omega",
        "off_policy",
        "source_policy",
        "tools",
        "retriever",
        "api",
        "bipia",
        "deepset",
        "pitheta_dataset_registry",
        "pitheta_train",
        "release_gate",
        "profile",
    ):
        path = files[name]
        if path.exists():
            file_hashes[str(path.as_posix())] = _sha256_bytes(path.read_bytes())
        layer = _load_yaml(path)
        resolved = _deep_merge(resolved, layer)

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

