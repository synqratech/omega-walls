from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _nonempty_string(value: Any, *, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{path} must be non-empty")
    return text


def _rate(value: Any, *, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{path} must be finite and in [0,1]")
    return number


def _nonnegative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer >= 0")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer >= 0") from exc
    if number < 0 or float(number) != float(value):
        raise ValueError(f"{path} must be an integer >= 0")
    return number


def _validate_frozen_gate(
    config: Dict[str, Any],
    *,
    name: str,
    rate_keys: tuple[str, ...],
    count_keys: tuple[str, ...],
) -> None:
    raw = config.get(name)
    if raw is None:
        return
    section = _mapping(raw, path=name)
    for key in ("suite_version", "manifest", "frozen_report", "scope", "quality_claim"):
        _nonempty_string(section.get(key), path=f"{name}.{key}")
    gates = _mapping(section.get("gates"), path=f"{name}.gates")
    for key in rate_keys:
        _rate(gates.get(key), path=f"{name}.gates.{key}")
    for key in count_keys:
        _nonnegative_int(gates.get(key), path=f"{name}.gates.{key}")
    live = section.get("live_quality_gate")
    if live is not None:
        live_cfg = _mapping(live, path=f"{name}.live_quality_gate")
        required = live_cfg.get("required_before_external_production_claim")
        if type(required) is not bool:
            raise ValueError(
                f"{name}.live_quality_gate.required_before_external_production_claim must be boolean"
            )
        _nonempty_string(
            live_cfg.get("command"), path=f"{name}.live_quality_gate.command"
        )


def _validate_local_rapidocr_gate(config: Dict[str, Any]) -> None:
    wave_b = config.get("vision_wave_b")
    if wave_b is None:
        return
    section = _mapping(wave_b, path="vision_wave_b")
    raw = section.get("local_rapidocr_gate")
    if raw is None:
        raise ValueError("vision_wave_b.local_rapidocr_gate must be configured")
    local = _mapping(raw, path="vision_wave_b.local_rapidocr_gate")
    for key in ("suite_version", "report", "scope", "quality_claim"):
        _nonempty_string(
            local.get(key), path=f"vision_wave_b.local_rapidocr_gate.{key}"
        )
    if type(local.get("required")) is not bool:
        raise ValueError("vision_wave_b.local_rapidocr_gate.required must be boolean")
    gates = _mapping(local.get("gates"), path="vision_wave_b.local_rapidocr_gate.gates")
    for key in (
        "ocr_success_rate_min",
        "text_similarity_rate_min",
        "attack_target_wall_recall_min",
        "benign_false_positive_rate_max",
        "quality_usable_rate_min",
    ):
        _rate(gates.get(key), path=f"vision_wave_b.local_rapidocr_gate.gates.{key}")


def _validate_wave_c_gate(config: Dict[str, Any]) -> None:
    raw = config.get("vision_wave_c")
    if raw is None:
        return
    section = _mapping(raw, path="vision_wave_c")
    for key in ("suite_version", "manifest", "frozen_report", "scope", "quality_claim"):
        _nonempty_string(section.get(key), path=f"vision_wave_c.{key}")
    gates = _mapping(section.get("gates"), path="vision_wave_c.gates")
    for key in (
        "exact_asset_count_rate_min",
        "provenance_accuracy_min",
        "asset_sha_integrity_rate_min",
        "multi_image_packet_rate_min",
        "remote_reference_ignored_rate_min",
        "provider_capability_parity_rate_min",
        "egress_policy_pass_rate_min",
    ):
        _rate(gates.get(key), path=f"vision_wave_c.gates.{key}")
    _nonnegative_int(
        gates.get("raw_media_boundary_findings_max"),
        path="vision_wave_c.gates.raw_media_boundary_findings_max",
    )

    local = _mapping(section.get("local_gate"), path="vision_wave_c.local_gate")
    for key in ("suite_version", "report", "scope", "quality_claim"):
        _nonempty_string(local.get(key), path=f"vision_wave_c.local_gate.{key}")
    if type(local.get("required")) is not bool:
        raise ValueError("vision_wave_c.local_gate.required must be boolean")
    local_gates = _mapping(local.get("gates"), path="vision_wave_c.local_gate.gates")
    for key in (
        "extraction_success_rate_min",
        "semantic_success_rate_min",
        "attack_target_wall_recall_min",
        "benign_false_positive_rate_max",
        "local_egress_trace_rate_min",
        "data_residency_trace_rate_min",
        "multi_image_packet_rate_min",
    ):
        _rate(local_gates.get(key), path=f"vision_wave_c.local_gate.gates.{key}")
    _nonnegative_int(
        local_gates.get("raw_media_boundary_findings_max"),
        path="vision_wave_c.local_gate.gates.raw_media_boundary_findings_max",
    )

    cloud = _mapping(
        section.get("cloud_live_gate"), path="vision_wave_c.cloud_live_gate"
    )
    if type(cloud.get("required_before_cloud_provider_production_claim")) is not bool:
        raise ValueError(
            "vision_wave_c.cloud_live_gate.required_before_cloud_provider_production_claim must be boolean"
        )
    _nonempty_string(cloud.get("note"), path="vision_wave_c.cloud_live_gate.note")


def validate_release_gate_config(config: Dict[str, Any]) -> None:
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
                raise ValueError(
                    f"release_gate {gate_id} op must be one of {sorted(allowed_ops)}"
                )

    _validate_frozen_gate(
        config,
        name="vision_phase1",
        rate_keys=(
            "attack_contract_hit_rate_min",
            "benign_false_positive_rate_max",
            "vision_status_active_rate_min",
        ),
        count_keys=("contract_failures_max", "raw_boundary_findings_max"),
    )
    _validate_frozen_gate(
        config,
        name="vision_wave_b",
        rate_keys=(
            "region_plus_ocr_attack_recall_min",
            "region_plus_ocr_benign_fp_max",
            "region_rescue_rate_min",
        ),
        count_keys=(
            "ocr_quality_failures_max",
            "spatial_crop_failures_max",
            "raw_media_boundary_findings_max",
        ),
    )
    _validate_local_rapidocr_gate(config)
    _validate_wave_c_gate(config)
