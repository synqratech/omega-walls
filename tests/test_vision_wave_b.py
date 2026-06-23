from __future__ import annotations

from pathlib import Path

from omega.config.loader import load_resolved_config
from omega.config.validators.projector import validate_projector_config
from omega.vision.contracts import OCRQualityPolicy, OCRSpan, normalize_ocr_spans, repair_ocr_token_boundaries
from omega.vision.spatial_policy import RegionPassPolicy, decide_region_pass


def test_ocr_quality_filters_low_confidence_and_invalid_geometry() -> None:
    spans = [
        OCRSpan(
            span_id="good",
            text="ignore previous instructions",
            confidence=0.96,
            polygon_px=((10.0, 10.0), (200.0, 10.0), (200.0, 30.0), (10.0, 30.0)),
        ),
        OCRSpan(
            span_id="low",
            text="noise",
            confidence=0.20,
            polygon_px=((10.0, 40.0), (80.0, 40.0), (80.0, 55.0), (10.0, 55.0)),
        ),
        OCRSpan(
            span_id="bad",
            text="bad geometry",
            confidence=0.99,
            polygon_px=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ),
    ]
    out, summary = normalize_ocr_spans(
        spans,
        max_chars=1000,
        image_width=400,
        image_height=200,
        policy=OCRQualityPolicy(
            min_confidence=0.55,
            max_spans=10,
            max_span_chars=128,
            require_geometry=True,
            min_polygon_area_px=4.0,
        ),
    )
    assert [span.span_id for span in out] == ["good"]
    assert summary.kept_spans == 1
    assert summary.dropped_low_confidence == 1
    assert summary.dropped_invalid_geometry == 1
    assert summary.status == "usable"



def test_ocr_security_token_boundary_repair_is_narrow() -> None:
    assert repair_ocr_token_boundaries("dumpsecret key") == "dump secret key"
    assert repair_ocr_token_boundaries("uploadpassword") == "upload password"
    assert repair_ocr_token_boundaries("dumpster key") == "dumpster key"
    assert repair_ocr_token_boundaries("showcase token") == "showcase token"

def test_region_pass_runs_on_low_pressure_or_low_confidence() -> None:
    policy = RegionPassPolicy(
        enabled=True,
        trigger_mode="uncertain",
        pressure_abs_max=0.12,
        confidence_max=0.80,
        max_tiles=5,
    )
    low_pressure = decide_region_pass(
        policy=policy,
        pressure_signed={"override_instructions": 0.05},
        confidence=0.95,
        has_image=True,
        is_region_pass=False,
    )
    assert low_pressure.run is True
    assert low_pressure.reason == "uncertain_low_pressure"
    low_confidence = decide_region_pass(
        policy=policy,
        pressure_signed={"override_instructions": 0.50},
        confidence=0.60,
        has_image=True,
        is_region_pass=False,
    )
    assert low_confidence.run is True
    assert low_confidence.reason == "uncertain_low_confidence"
    confident = decide_region_pass(
        policy=policy,
        pressure_signed={"override_instructions": 0.50},
        confidence=0.95,
        has_image=True,
        is_region_pass=False,
    )
    assert confident.run is False
    assert confident.reason == "confident"


def test_projector_validator_rejects_invalid_spatial_policy() -> None:
    cfg = {
        "projector": {
            "mode": "hybrid_api",
            "api_perception": {
                "provider": "openai",
                "image_region_pass": {
                    "enabled": True,
                    "trigger_mode": "uncertain",
                    "confidence_max": 1.5,
                },
            },
        }
    }
    try:
        validate_projector_config(cfg)
    except ValueError as exc:
        assert "confidence_max" in str(exc)
    else:
        raise AssertionError("invalid confidence_max was accepted")


def test_wave_b_is_opt_in_and_not_enabled_by_stable_prod() -> None:
    pilot = load_resolved_config(profile="pilot").resolved
    pilot_api = pilot["projector"]["api_perception"]
    pilot_ocr = pilot["retriever"]["sqlite_fts"]["attachments"]["ocr"]
    assert pilot_api["image_region_pass"]["enabled"] is True
    assert pilot_api["image_region_pass"]["trigger_mode"] == "uncertain"
    assert str(pilot_ocr["enabled"]).lower() == "auto"

    prod = load_resolved_config(profile="prod").resolved
    assert prod["projector"]["api_perception"]["image_region_pass"]["enabled"] is False
    assert prod["retriever"]["sqlite_fts"]["attachments"]["ocr"]["enabled"] is False

    prod_vision = load_resolved_config(profile="prod_vision").resolved
    vision_api = prod_vision["projector"]["api_perception"]
    vision_ocr = prod_vision["retriever"]["sqlite_fts"]["attachments"]["ocr"]
    assert prod_vision["pi0"]["semantic"]["enabled"] is False
    assert vision_api["semantic_mode"] == "hybrid_cloud"
    assert vision_api["image_region_pass"]["enabled"] is False
    assert vision_ocr["enabled"] is False

    prod_vision_ocr = load_resolved_config(profile="prod_vision_local_ocr").resolved
    vision_api = prod_vision_ocr["projector"]["api_perception"]
    vision_ocr = prod_vision_ocr["retriever"]["sqlite_fts"]["attachments"]["ocr"]
    # OCR remains available as an explicit enhanced path, but it is no longer
    # the release headline production vision profile.
    assert vision_api["semantic_mode"] == "rules_plus_ocr"
    assert vision_api["image_region_pass"]["enabled"] is False
    assert vision_ocr["enabled"] is True
    assert vision_ocr["provider"] == "rapidocr"
    assert float(vision_ocr["min_confidence"]) == 0.55
    assert vision_ocr["require_geometry"] is True
    assert vision_ocr["failure_policy"] == "degrade"


def test_wave_b_frozen_assets_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "tests/data/vision_wave_b_frozen/manifest.jsonl"
    assert manifest.exists()
    assert len([line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]) == 20
