from __future__ import annotations

from typing import Any, Dict

from .common import WALLS_V1_ORDER
from omega.projector.api_hybrid import normalization as _provider_norm


_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _validate_provider_capabilities(
    *, capabilities: Any, provider: str, path: str
) -> None:
    if capabilities in (None, {}):
        return
    if not isinstance(capabilities, dict):
        raise ValueError(f"{path} must be a mapping")
    allowed = {
        "text",
        "image",
        "supported_image_mime_types",
        "max_image_bytes",
        "max_images",
    }
    unknown = sorted(set(capabilities) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {','.join(unknown)}")
    for key in ("text", "image"):
        if key in capabilities and type(capabilities.get(key)) is not bool:
            raise ValueError(f"{path}.{key} must be boolean")
    image_enabled = bool(
        capabilities.get("image", provider in {"openai", "anthropic", "local_vision"})
    )
    if image_enabled and provider not in {
        "openai",
        "anthropic",
        "openai_compat",
        "local_vision",
    }:
        raise ValueError(f"{path}.image=true is not supported for provider {provider}")
    mimes = capabilities.get("supported_image_mime_types")
    if mimes is not None:
        if not isinstance(mimes, list) or not mimes:
            raise ValueError(
                f"{path}.supported_image_mime_types must be a non-empty list"
            )
        normalized = [str(x).strip().lower() for x in mimes]
        if any(x not in _IMAGE_MIMES for x in normalized):
            raise ValueError(
                f"{path}.supported_image_mime_types contains unsupported mime"
            )
        if not image_enabled:
            raise ValueError(f"{path}.supported_image_mime_types requires image=true")
    if (
        "max_image_bytes" in capabilities
        and int(capabilities.get("max_image_bytes", 0)) <= 0
    ):
        raise ValueError(f"{path}.max_image_bytes must be > 0")
    if "max_images" in capabilities and int(capabilities.get("max_images", 0)) <= 0:
        raise ValueError(f"{path}.max_images must be > 0")


def validate_projector_config(config: Dict[str, Any]) -> None:
    projector_cfg = config.get("projector", {}) or {}
    if not projector_cfg:
        return

    mode = str(projector_cfg.get("mode", "pi0")).lower()
    if mode not in {"pi0", "pitheta", "hybrid", "hybrid_api"}:
        raise ValueError("projector.mode must be pi0|pitheta|hybrid|hybrid_api")
    api_cfg = projector_cfg.get("api_perception", {}) or {}
    if api_cfg:
        semantic_mode_raw = api_cfg.get("semantic_mode", None)
        semantic_mode = (
            str(semantic_mode_raw).strip().lower()
            if semantic_mode_raw is not None
            else ""
        )
        if semantic_mode and semantic_mode not in {
            "rules_only",
            "hybrid_cloud",
            "hybrid_redacted",
            "local_semantic",
            "rules_plus_ocr",
        }:
            raise ValueError(
                "projector.api_perception.semantic_mode must be rules_only|hybrid_cloud|hybrid_redacted|local_semantic|rules_plus_ocr"
            )
        api_cfg_effective = _provider_norm.apply_provider_preset(dict(api_cfg))
        provider = str(api_cfg_effective.get("provider", "openai")).strip().lower()
        if provider not in {"openai", "anthropic", "openai_compat", "local_vision"}:
            raise ValueError(
                "projector.api_perception.provider must be openai|anthropic|openai_compat|local_vision"
            )
        provider_options = api_cfg_effective.get("provider_options", {}) or {}
        if provider_options and not isinstance(provider_options, dict):
            raise ValueError(
                "projector.api_perception.provider_options must be a mapping"
            )
        capabilities = (
            provider_options.get("capabilities", {})
            if isinstance(provider_options, dict)
            else {}
        )
        _validate_provider_capabilities(
            capabilities=capabilities,
            provider=provider,
            path="projector.api_perception.provider_options.capabilities",
        )
        if "allow_raw_image_outbound" in provider_options and not isinstance(
            provider_options.get("allow_raw_image_outbound"), bool
        ):
            raise ValueError(
                "projector.api_perception.provider_options.allow_raw_image_outbound must be boolean"
            )
        if (
            "hybrid_redacted_allow_raw_image_outbound" in provider_options
            and not isinstance(
                provider_options.get("hybrid_redacted_allow_raw_image_outbound"), bool
            )
        ):
            raise ValueError(
                "projector.api_perception.provider_options.hybrid_redacted_allow_raw_image_outbound must be boolean"
            )
        if (
            "blob_ttl_sec" in provider_options
            and float(provider_options.get("blob_ttl_sec", 0.0)) <= 0.0
        ):
            raise ValueError(
                "projector.api_perception.provider_options.blob_ttl_sec must be > 0"
            )
        if (
            "blob_max_total_bytes" in provider_options
            and int(provider_options.get("blob_max_total_bytes", 0)) <= 0
        ):
            raise ValueError(
                "projector.api_perception.provider_options.blob_max_total_bytes must be > 0"
            )
        if (
            "blob_max_records" in provider_options
            and int(provider_options.get("blob_max_records", 0)) <= 0
        ):
            raise ValueError(
                "projector.api_perception.provider_options.blob_max_records must be > 0"
            )
        visual_egress = (
            provider_options.get("visual_egress", {})
            if isinstance(provider_options, dict)
            else {}
        )
        if visual_egress:
            if not isinstance(visual_egress, dict):
                raise ValueError(
                    "projector.api_perception.provider_options.visual_egress must be a mapping"
                )
            if (
                "enabled" in visual_egress
                and type(visual_egress.get("enabled")) is not bool
            ):
                raise ValueError(
                    "projector.api_perception.provider_options.visual_egress.enabled must be boolean"
                )
            if str(visual_egress.get("default_action", "allow")) not in {
                "allow",
                "deny",
            }:
                raise ValueError(
                    "projector.api_perception.provider_options.visual_egress.default_action must be allow|deny"
                )
            for key in ("providers", "tenants"):
                if key in visual_egress and not isinstance(
                    visual_egress.get(key), dict
                ):
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.{key} must be a mapping"
                    )
            for provider_id, row in dict(
                visual_egress.get("providers", {}) or {}
            ).items():
                if not isinstance(row, dict):
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.providers.{provider_id} must be a mapping"
                    )
                unknown = sorted(set(row) - {"external", "region"})
                if unknown:
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.providers.{provider_id} contains unknown fields: {','.join(unknown)}"
                    )
                if "external" in row and type(row.get("external")) is not bool:
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.providers.{provider_id}.external must be boolean"
                    )
                if not str(row.get("region", "global")).strip():
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.providers.{provider_id}.region must be non-empty"
                    )
            for tenant_id, row in dict(visual_egress.get("tenants", {}) or {}).items():
                if not isinstance(row, dict):
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.tenants.{tenant_id} must be a mapping"
                    )
                unknown = sorted(
                    set(row)
                    - {
                        "allow_external",
                        "allowed_providers",
                        "allowed_regions",
                        "require_region_match",
                        "require_data_region",
                    }
                )
                if unknown:
                    raise ValueError(
                        f"projector.api_perception.provider_options.visual_egress.tenants.{tenant_id} contains unknown fields: {','.join(unknown)}"
                    )
                for flag in (
                    "allow_external",
                    "require_region_match",
                    "require_data_region",
                ):
                    if flag in row and type(row.get(flag)) is not bool:
                        raise ValueError(
                            f"projector.api_perception.provider_options.visual_egress.tenants.{tenant_id}.{flag} must be boolean"
                        )
                for list_field in ("allowed_providers", "allowed_regions"):
                    if list_field in row and not isinstance(row.get(list_field), list):
                        raise ValueError(
                            f"projector.api_perception.provider_options.visual_egress.tenants.{tenant_id}.{list_field} must be a list"
                        )
        local_vision = (
            provider_options.get("local_vision", {})
            if isinstance(provider_options, dict)
            else {}
        )
        allow_loopback_http = False
        if local_vision:
            if not isinstance(local_vision, dict):
                raise ValueError(
                    "projector.api_perception.provider_options.local_vision must be a mapping"
                )
            backend = str(local_vision.get("backend", "ocr_pi0")).strip().lower()
            if backend not in {"ocr_pi0", "openai_compatible"}:
                raise ValueError(
                    "projector.api_perception.provider_options.local_vision.backend must be ocr_pi0|openai_compatible"
                )
            if backend == "openai_compatible":
                from urllib.parse import urlparse

                parsed = urlparse(str(api_cfg_effective.get("base_url", "")))
                if parsed.scheme not in {"http", "https"} or str(
                    parsed.hostname or ""
                ).lower() not in {"localhost", "127.0.0.1", "::1"}:
                    raise ValueError(
                        "local_vision openai_compatible base_url must use a loopback host"
                    )
                allow_loopback_http = True
        if (
            "allow_legacy_inline_image_meta" in provider_options
            and type(provider_options.get("allow_legacy_inline_image_meta")) is not bool
        ):
            raise ValueError(
                "projector.api_perception.provider_options.allow_legacy_inline_image_meta must be boolean"
            )
        enabled = str(api_cfg_effective.get("enabled", "auto")).lower()
        if enabled not in {"auto", "true", "false"}:
            raise ValueError("projector.api_perception.enabled must be auto|true|false")
        if not str(api_cfg_effective.get("model", "gpt-5")).strip():
            raise ValueError("projector.api_perception.model must be non-empty")
        default_base_url = (
            "https://api.anthropic.com/v1"
            if provider == "anthropic"
            else "https://api.openai.com/v1"
        )
        base_url = _provider_norm.canonical_base_url(api_cfg_effective.get("base_url", default_base_url))
        if not str(base_url).strip():
            raise ValueError("projector.api_perception.base_url must be non-empty")
        allowed_base_urls = _provider_norm.normalize_allowed_base_urls(
            api_cfg_effective.get("allowed_base_urls", ())
        )
        _provider_norm.enforce_provider_endpoint_policy(
            base_url=base_url,
            allowed_base_urls=allowed_base_urls,
            allow_http_private_gateway=bool(api_cfg_effective.get("allow_http_private_gateway", False)),
            allow_loopback_http=allow_loopback_http,
        )
        _ = _provider_norm.normalize_extra_headers(api_cfg_effective.get("extra_headers", {}))
        if "allow_redirects" in api_cfg_effective and type(api_cfg_effective.get("allow_redirects")) is not bool:
            raise ValueError("projector.api_perception.allow_redirects must be boolean")
        for file_field in ("api_key_file", "api_key_file_env"):
            if file_field in api_cfg_effective and not str(api_cfg_effective.get(file_field, "")).strip():
                raise ValueError(f"projector.api_perception.{file_field} must be non-empty when provided")
        default_api_key_env = (
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        )
        if not str(api_cfg_effective.get("api_key_env", default_api_key_env)).strip():
            raise ValueError("projector.api_perception.api_key_env must be non-empty")
        if float(api_cfg_effective.get("timeout_sec", 30.0)) <= 0.0:
            raise ValueError("projector.api_perception.timeout_sec must be > 0")
        if int(api_cfg_effective.get("max_retries", 2)) < 0:
            raise ValueError("projector.api_perception.max_retries must be >= 0")
        if float(api_cfg_effective.get("backoff_sec", 0.75)) < 0.0:
            raise ValueError("projector.api_perception.backoff_sec must be >= 0")
        if float(api_cfg_effective.get("retry_backoff_max_sec", 2.0)) < 0.0:
            raise ValueError(
                "projector.api_perception.retry_backoff_max_sec must be >= 0"
            )
        if float(api_cfg_effective.get("request_deadline_sec", 20.0)) <= 0.0:
            raise ValueError(
                "projector.api_perception.request_deadline_sec must be > 0"
            )
        if int(api_cfg_effective.get("long_text_threshold_chars", 3000)) <= 0:
            raise ValueError(
                "projector.api_perception.long_text_threshold_chars must be > 0"
            )
        if int(api_cfg_effective.get("long_text_max_retries", 1)) < 0:
            raise ValueError(
                "projector.api_perception.long_text_max_retries must be >= 0"
            )
        short_thr = int(api_cfg_effective.get("short_text_threshold_chars", 1200))
        if short_thr <= 0:
            raise ValueError(
                "projector.api_perception.short_text_threshold_chars must be > 0"
            )
        _ = bool(api_cfg_effective.get("short_prefer_chat_completions", True))
        _ = bool(api_cfg_effective.get("short_chat_only", True))
        _ = bool(api_cfg_effective.get("short_fast_path_enabled", True))
        _ = bool(api_cfg_effective.get("short_fast_path_skip_on_pi0_hard", True))
        _ = bool(api_cfg_effective.get("short_fast_path_skip_on_pi0_clean", True))
        hard_min = float(api_cfg_effective.get("short_fast_path_hard_min_score", 0.55))
        clean_max = float(api_cfg_effective.get("short_fast_path_clean_max_score", 0.0))
        if hard_min < 0.0 or hard_min > 1.0:
            raise ValueError(
                "projector.api_perception.short_fast_path_hard_min_score must be in [0,1]"
            )
        if clean_max < 0.0 or clean_max > 1.0:
            raise ValueError(
                "projector.api_perception.short_fast_path_clean_max_score must be in [0,1]"
            )
        if clean_max > hard_min:
            raise ValueError(
                "projector.api_perception.short_fast_path_clean_max_score must be <= short_fast_path_hard_min_score"
            )
        _ = bool(api_cfg_effective.get("prewarm_on_init", True))
        if float(api_cfg_effective.get("transient_error_ttl_sec", 90.0)) < 0.0:
            raise ValueError(
                "projector.api_perception.transient_error_ttl_sec must be >= 0"
            )
        if float(api_cfg_effective.get("responses_cooldown_sec", 60.0)) < 0.0:
            raise ValueError(
                "projector.api_perception.responses_cooldown_sec must be >= 0"
            )
        if not str(api_cfg_effective.get("prompt_version", "api_hybrid_v1")).strip():
            raise ValueError(
                "projector.api_perception.prompt_version must be non-empty"
            )
        if "cache_path" in api_cfg and not str(api_cfg_effective.get("cache_path", "")).strip():
            raise ValueError(
                "projector.api_perception.cache_path must be non-empty when provided"
            )
        if (
            "error_log_path" in api_cfg
            and not str(api_cfg_effective.get("error_log_path", "")).strip()
        ):
            raise ValueError(
                "projector.api_perception.error_log_path must be non-empty when provided"
            )
        if (
            "image_region_pass_enabled" in api_cfg
            and type(api_cfg_effective.get("image_region_pass_enabled")) is not bool
        ):
            raise ValueError(
                "projector.api_perception.image_region_pass_enabled must be boolean"
            )
        region_cfg = api_cfg_effective.get("image_region_pass", {}) or {}
        if region_cfg and not isinstance(region_cfg, dict):
            raise ValueError(
                "projector.api_perception.image_region_pass must be a mapping"
            )
        if isinstance(region_cfg, dict) and region_cfg:
            allowed_region = {
                "enabled",
                "trigger_mode",
                "pressure_abs_max",
                "confidence_max",
                "max_tiles",
                "overlap_ratio",
                "include_center_crop",
            }
            unknown_region = sorted(set(region_cfg) - allowed_region)
            if unknown_region:
                raise ValueError(
                    "projector.api_perception.image_region_pass contains unknown fields: "
                    + ",".join(unknown_region)
                )
            if "enabled" in region_cfg and type(region_cfg.get("enabled")) is not bool:
                raise ValueError(
                    "projector.api_perception.image_region_pass.enabled must be boolean"
                )
            trigger_mode = (
                str(region_cfg.get("trigger_mode", "uncertain")).strip().lower()
            )
            if trigger_mode not in {"zero", "uncertain", "always"}:
                raise ValueError(
                    "projector.api_perception.image_region_pass.trigger_mode must be zero|uncertain|always"
                )
            pressure_abs_max = float(region_cfg.get("pressure_abs_max", 0.12))
            if pressure_abs_max < 0.0:
                raise ValueError(
                    "projector.api_perception.image_region_pass.pressure_abs_max must be >= 0"
                )
            confidence_max = float(region_cfg.get("confidence_max", 0.80))
            if confidence_max < 0.0 or confidence_max > 1.0:
                raise ValueError(
                    "projector.api_perception.image_region_pass.confidence_max must be in [0,1]"
                )
            max_tiles = int(region_cfg.get("max_tiles", 5))
            if max_tiles <= 0 or max_tiles > 16:
                raise ValueError(
                    "projector.api_perception.image_region_pass.max_tiles must be in [1,16]"
                )
            overlap_ratio = float(region_cfg.get("overlap_ratio", 0.08))
            if overlap_ratio < 0.0 or overlap_ratio > 0.5:
                raise ValueError(
                    "projector.api_perception.image_region_pass.overlap_ratio must be in [0,0.5]"
                )
            if (
                "include_center_crop" in region_cfg
                and type(region_cfg.get("include_center_crop")) is not bool
            ):
                raise ValueError(
                    "projector.api_perception.image_region_pass.include_center_crop must be boolean"
                )
        orch_cfg = api_cfg_effective.get("orchestrator", {}) or {}
        if orch_cfg and not isinstance(orch_cfg, dict):
            raise ValueError("projector.api_perception.orchestrator must be a mapping")
        if isinstance(orch_cfg, dict) and orch_cfg:
            _ = bool(orch_cfg.get("enabled", False))
            if not str(orch_cfg.get("master_key_env", "OMEGA_MASTER_KEY")).strip():
                raise ValueError(
                    "projector.api_perception.orchestrator.master_key_env must be non-empty"
                )
            store_cfg = orch_cfg.get("store", {}) or {}
            if store_cfg and not isinstance(store_cfg, dict):
                raise ValueError(
                    "projector.api_perception.orchestrator.store must be a mapping"
                )
            if not str(
                store_cfg.get("sqlite_path", "artifacts/state/provider_orchestrator.db")
            ).strip():
                raise ValueError(
                    "projector.api_perception.orchestrator.store.sqlite_path must be non-empty"
                )
            fallback_cfg = orch_cfg.get("fallback", {}) or {}
            if fallback_cfg and not isinstance(fallback_cfg, dict):
                raise ValueError(
                    "projector.api_perception.orchestrator.fallback must be a mapping"
                )
            mode = str(fallback_cfg.get("mode", "rule_only")).strip().lower()
            if mode not in {"rule_only", "fail_closed"}:
                raise ValueError(
                    "projector.api_perception.orchestrator.fallback.mode must be rule_only|fail_closed"
                )
            threshold_cfg = fallback_cfg.get("threshold", {}) or {}
            if threshold_cfg and not isinstance(threshold_cfg, dict):
                raise ValueError(
                    "projector.api_perception.orchestrator.fallback.threshold must be a mapping"
                )
            if int(threshold_cfg.get("errors", 3)) <= 0:
                raise ValueError(
                    "projector.api_perception.orchestrator.fallback.threshold.errors must be > 0"
                )
            if int(threshold_cfg.get("window_sec", 60)) <= 0:
                raise ValueError(
                    "projector.api_perception.orchestrator.fallback.threshold.window_sec must be > 0"
                )
            recovery_cfg = orch_cfg.get("recovery", {}) or {}
            if recovery_cfg and not isinstance(recovery_cfg, dict):
                raise ValueError(
                    "projector.api_perception.orchestrator.recovery must be a mapping"
                )
            interval = int(recovery_cfg.get("healthcheck_interval_sec", 180))
            if interval < 120 or interval > 300:
                raise ValueError(
                    "projector.api_perception.orchestrator.recovery.healthcheck_interval_sec must be in [120,300]"
                )
            alerts_cfg = orch_cfg.get("alerts", {}) or {}
            if alerts_cfg and not isinstance(alerts_cfg, dict):
                raise ValueError(
                    "projector.api_perception.orchestrator.alerts must be a mapping"
                )
            if int(alerts_cfg.get("cooldown_sec", 900)) <= 0:
                raise ValueError(
                    "projector.api_perception.orchestrator.alerts.cooldown_sec must be > 0"
                )
            providers_cfg = orch_cfg.get("providers", [])
            if providers_cfg and not isinstance(providers_cfg, list):
                raise ValueError(
                    "projector.api_perception.orchestrator.providers must be a list"
                )
            if isinstance(providers_cfg, list):
                for idx, row in enumerate(providers_cfg):
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"projector.api_perception.orchestrator.providers[{idx}] must be a mapping"
                        )
                    if not str(row.get("id", "")).strip():
                        raise ValueError(
                            f"projector.api_perception.orchestrator.providers[{idx}].id must be non-empty"
                        )
                    ptype = str(row.get("type", "")).strip().lower()
                    if ptype not in {
                        "openai",
                        "anthropic",
                        "openai_compat",
                        "local_vision",
                    }:
                        raise ValueError(
                            f"projector.api_perception.orchestrator.providers[{idx}].type must be openai|anthropic|openai_compat|local_vision"
                        )
                    if "priority" in row and int(row.get("priority", 0)) < 0:
                        raise ValueError(
                            f"projector.api_perception.orchestrator.providers[{idx}].priority must be >= 0"
                        )
                    _validate_provider_capabilities(
                        capabilities=row.get("capabilities", {}),
                        provider=ptype,
                        path=f"projector.api_perception.orchestrator.providers[{idx}].capabilities",
                    )
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
            raise ValueError(
                "projector.pitheta.conversion_mode must be expected|argmax"
            )
        pressure_map = list(pitheta_cfg.get("pressure_map", [0.0, 0.25, 0.6, 1.0]))
        if len(pressure_map) != 4:
            raise ValueError("projector.pitheta.pressure_map must have 4 values")
        last = -1.0
        for i, value in enumerate(pressure_map):
            v = float(value)
            if v < 0.0:
                raise ValueError(f"projector.pitheta.pressure_map[{i}] must be >= 0")
            if i > 0 and v < last:
                raise ValueError(
                    "projector.pitheta.pressure_map must be non-decreasing"
                )
            last = v
        _ = bool(pitheta_cfg.get("require_calibration", True))
        if (
            "calibration_file" in pitheta_cfg
            and not str(pitheta_cfg.get("calibration_file", "")).strip()
        ):
            raise ValueError(
                "projector.pitheta.calibration_file must be non-empty when provided"
            )
        thresholds = (pitheta_cfg.get("legacy", {}) or {}).get(
            "wall_thresholds", pitheta_cfg.get("wall_thresholds", {})
        ) or {}
        for wall in WALLS_V1_ORDER:
            val = float(thresholds.get(wall, 0.5))
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"projector.pitheta.wall_thresholds.{wall} must be in [0,1]"
                )
