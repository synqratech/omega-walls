from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.eval_wainject_image_balanced_partial import (
    BENIGN_CATEGORY_TARGETS,
    MALICIOUS_CATEGORIES,
    _allocate_proportional_targets,
    _build_backend_command,
    _opaque_staged_name,
    _select_benign_samples,
    _select_malicious_samples,
    _validate_backend_quality_total,
    _validate_selected_total,
    build_balanced_report,
    main,
)


def _write_images(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        (root / f"{idx:03d}.png").write_bytes(f"img-{idx}".encode("utf-8"))


def _mk_backend_report(*, total: int, allow: int, quarantine: int, block: int, ocr: int, vision: int, semantic_failed: int, latency: float):
    return {
        "scenarios": [
            {
                "phase_metrics": {
                    "quality_fresh": {
                        "total": total,
                        "verdict_counts": {
                            "allow": allow,
                            "quarantine": quarantine,
                            "block": block,
                        },
                        "ocr_success_count": ocr,
                        "vision_active_count": vision,
                        "semantic_failed_count": semantic_failed,
                        "semantic_latency_active_count": total,
                        "provider_call_count": {"active_count": total, "total": total, "avg": float(total), "max": total},
                        "cache_hit_last_request_count": 0,
                        "latency_ms": {"avg": latency},
                    }
                }
            }
        ]
    }


def test_allocate_proportional_targets_redistributes_when_bucket_is_small():
    counts = {
        "EIA": 40,
        "popup": 20,
        "VPI": 1,
        "VWA_adv_embedded_img": 20,
        "VWA_adv_screenshot": 20,
        "wasp": 10,
        "WebInject": 40,
    }

    allocation = _allocate_proportional_targets(counts, total=50)

    assert sum(allocation.values()) == 50
    assert allocation["VPI"] <= counts["VPI"]
    assert allocation["EIA"] >= 1
    assert allocation["WebInject"] >= 1


def test_select_benign_samples_uses_fixed_25_per_category(tmp_path: Path):
    benign_root = tmp_path / "benign"
    for category, target in BENIGN_CATEGORY_TARGETS.items():
        _write_images(benign_root / category, target + 3)

    selected = _select_benign_samples(benign_root, seed=41)

    assert len(selected) == 50
    assert sum(1 for sample in selected if sample.category == "embedded_img") == 25
    assert sum(1 for sample in selected if sample.category == "screenshot") == 25
    assert all(sample.staged_name.startswith("sample_") for sample in selected)
    assert all("embedded_img" not in sample.staged_name for sample in selected)
    assert all("screenshot" not in sample.staged_name for sample in selected)
    assert all("_b_" not in sample.staged_name for sample in selected)


def test_select_malicious_samples_returns_balanced_total_with_allocation(tmp_path: Path):
    malicious_root = tmp_path / "malicious"
    counts = {
        "EIA": 30,
        "popup": 10,
        "VPI": 2,
        "VWA_adv_embedded_img": 15,
        "VWA_adv_screenshot": 17,
        "wasp": 4,
        "WebInject": 35,
    }
    for category in MALICIOUS_CATEGORIES:
        _write_images(malicious_root / category, counts[category])

    selected = _select_malicious_samples(malicious_root, seed=41, total=50)

    assert len(selected) == 50
    picked_by_category = {category: sum(1 for sample in selected if sample.category == category) for category in MALICIOUS_CATEGORIES}
    assert picked_by_category["VPI"] <= counts["VPI"]
    assert picked_by_category["wasp"] <= counts["wasp"]
    assert sum(picked_by_category.values()) == 50
    assert all(sample.staged_name.startswith("sample_") for sample in selected)
    assert all(sample.category not in sample.staged_name for sample in selected)
    assert all("_m_" not in sample.staged_name for sample in selected)


def test_build_backend_command_uses_backend_mode_and_disables_dedupe(tmp_path: Path):
    command = _build_backend_command(
        profile="prod_vision_local_ocr",
        root=tmp_path,
        max_samples=50,
        artifacts_root=tmp_path / "artifacts",
        mode="vision_plus_ocr",
        ocr_provider="rapidocr",
        pi0_semantic_enabled="true",
        pi0_semantic_device="auto",
        api_runtime_mode="",
        attachment_visual_enabled="",
        cache_namespace="run_benign",
    )

    assert "vision_plus_ocr" in command
    assert "hybrid_api" not in command
    assert "--dedupe-sha256" in command
    assert "false" in command
    assert "--cache-namespace" in command
    assert "run_benign" in command


def test_build_backend_command_can_request_prod_default_stateless_visual(tmp_path: Path):
    command = _build_backend_command(
        profile="sensitive_rules_only",
        root=tmp_path,
        max_samples=50,
        artifacts_root=tmp_path / "artifacts",
        mode="vision_single",
        ocr_provider="rapidocr",
        pi0_semantic_enabled="false",
        pi0_semantic_device="auto",
        api_runtime_mode="stateless",
        attachment_visual_enabled="true",
        cache_namespace="run_live",
    )

    assert "vision_single" in command
    assert "--api-runtime-mode" in command
    assert "stateless" in command
    assert "--attachment-visual-enabled" in command
    assert "--pi0-semantic-enabled" in command
    assert "false" in command
    assert "run_live" in command


def test_validate_selected_total_fails_when_contract_is_not_met():
    with pytest.raises(ValueError, match="expected exactly 50"):
        _validate_selected_total([], expected_total=50, label="malicious")


def test_validate_backend_quality_total_fails_on_denominator_mismatch():
    with pytest.raises(ValueError, match="quality_fresh.total=49"):
        _validate_backend_quality_total(
            _mk_backend_report(
                total=49,
                allow=49,
                quarantine=0,
                block=0,
                ocr=49,
                vision=49,
                semantic_failed=0,
                latency=100.0,
            ),
            expected_total=50,
            label="benign",
        )


def test_opaque_staged_name_is_label_generic():
    benign_name = _opaque_staged_name(index=1, suffix=".png")
    malicious_name = _opaque_staged_name(index=2, suffix=".jpg")

    assert benign_name == "sample_000001.png"
    assert malicious_name == "sample_000002.jpg"
    assert "benign" not in benign_name
    assert "malicious" not in malicious_name


def test_build_balanced_report_uses_quality_fresh_and_label_aware_rates():
    benign_samples = [type("S", (), {"category": "embedded_img", "staged_name": "sample_000001.png", "source_path": Path("a.png")})()]
    malicious_samples = [type("S", (), {"category": "EIA", "staged_name": "sample_000001.png", "source_path": Path("b.png")})()]
    benign_run = {
        "report_path": "artifacts/benign/report.json",
        "report": {
            **_mk_backend_report(
                total=1,
                allow=0,
                quarantine=1,
                block=0,
                ocr=1,
                vision=1,
                semantic_failed=0,
                latency=120.0,
            ),
            "scenarios": [
                {
                    "semantic_status_after_fresh": {
                        "api_semantic_mode": "rules_plus_ocr",
                        "api_provider": "local_vision",
                        "enabled_mode": "true",
                        "active": False,
                        "attempted": False,
                    },
                    "phase_metrics": {
                        "quality_fresh": {
                            "total": 1,
                            "verdict_counts": {"allow": 0, "quarantine": 1, "block": 0},
                            "ocr_success_count": 1,
                            "vision_active_count": 1,
                            "semantic_failed_count": 0,
                            "semantic_latency_active_count": 1,
                            "provider_call_count": {"active_count": 1, "total": 1, "avg": 1.0, "max": 1},
                            "latency_ms": {"avg": 120.0},
                        }
                    },
                }
            ],
        },
    }
    malicious_run = {
        "report_path": "artifacts/malicious/report.json",
        "report": {
            **_mk_backend_report(
                total=1,
                allow=0,
                quarantine=0,
                block=1,
                ocr=1,
                vision=1,
                semantic_failed=1,
                latency=150.0,
            ),
            "scenarios": [
                {
                    "semantic_status_after_fresh": {
                        "api_semantic_mode": "rules_plus_ocr",
                        "api_provider": "local_vision",
                        "enabled_mode": "true",
                        "active": False,
                        "attempted": False,
                    },
                    "phase_metrics": {
                        "quality_fresh": {
                            "total": 1,
                            "verdict_counts": {"allow": 0, "quarantine": 0, "block": 1},
                            "ocr_success_count": 1,
                            "vision_active_count": 1,
                            "semantic_failed_count": 1,
                            "semantic_latency_active_count": 1,
                            "provider_call_count": {"active_count": 1, "total": 1, "avg": 1.0, "max": 1},
                            "latency_ms": {"avg": 150.0},
                        }
                    },
                }
            ],
        },
    }

    report = build_balanced_report(
        profile="prod_vision_local_ocr",
        seed=41,
        mode="vision_plus_ocr",
        ocr_provider="rapidocr",
        pi0_semantic_enabled="true",
        pi0_semantic_device="auto",
        api_runtime_mode="stateful",
        attachment_visual_enabled="true",
        cache_replay_expected=False,
        benign_samples=benign_samples,
        malicious_samples=malicious_samples,
        benign_run=benign_run,
        malicious_run=malicious_run,
    )

    assert report["summary"]["attack_detect_rate"] == 1.0
    assert report["summary"]["benign_false_positive_rate"] == 1.0
    assert report["summary"]["benign_allow_rate"] == 0.0
    assert report["summary"]["malicious_allow_rate"] == 0.0
    assert report["summary"]["operational_health"]["ocr_success_count"] == 2
    assert report["summary"]["operational_health"]["provider_call_active_count"] == 2
    assert report["mode"] == "vision_plus_ocr"
    assert report["api_runtime_mode"] == "stateful"


def test_build_balanced_report_marks_invalid_headline_when_audit_gates_fail():
    samples = [type("S", (), {"category": "embedded_img", "staged_name": "sample_000001.png", "source_path": Path("a.png")})()]
    report_payload = {
        "scenarios": [
            {
                "semantic_status_after_fresh": {
                    "api_semantic_mode": "rules_only",
                    "api_provider": "openai",
                    "enabled_mode": "false",
                    "active": False,
                    "attempted": True,
                },
                "phase_metrics": {
                    "quality_fresh": {
                        "total": 1,
                        "verdict_counts": {"allow": 0, "quarantine": 1, "block": 0},
                        "ocr_success_count": 0,
                        "vision_active_count": 0,
                        "semantic_failed_count": 0,
                        "semantic_latency_active_count": 0,
                        "provider_call_count": {"active_count": 0, "total": 0, "avg": 0.0, "max": 0},
                        "latency_ms": {"avg": 50.0},
                    }
                },
            }
        ]
    }
    report = build_balanced_report(
        profile="pilot",
        seed=41,
        mode="vision_single",
        ocr_provider="rapidocr",
        pi0_semantic_enabled="false",
        pi0_semantic_device="auto",
        api_runtime_mode="stateless",
        attachment_visual_enabled="true",
        cache_replay_expected=False,
        benign_samples=samples,
        malicious_samples=samples,
        benign_run={"report_path": "a.json", "report": report_payload},
        malicious_run={"report_path": "b.json", "report": report_payload},
    )

    assert report["audit"]["quality_gates"]["passed"] is False
    failed = {item["name"] for item in report["audit"]["quality_gates"]["checks"] if not item["passed"]}
    assert "api_semantic_mode_hybrid_cloud" in failed
    assert "vision_active_full_coverage" in failed
    assert "provider_or_cache_full_coverage" in failed


def test_main_returns_nonzero_when_quality_gates_fail(tmp_path: Path):
    benign_root = tmp_path / "data" / "WAInjectBench" / "image" / "benign"
    malicious_root = tmp_path / "data" / "WAInjectBench" / "image" / "malicious"
    for category, target in BENIGN_CATEGORY_TARGETS.items():
        _write_images(benign_root / category, target)
    for category in MALICIOUS_CATEGORIES:
        _write_images(malicious_root / category, 8)

    invalid_report = {
        "scenarios": [
            {
                "semantic_status_after_fresh": {
                    "api_semantic_mode": "rules_only",
                    "api_provider": "openai",
                    "enabled_mode": "false",
                    "active": False,
                    "attempted": True,
                },
                "phase_metrics": {
                    "quality_fresh": {
                        "total": 50,
                        "verdict_counts": {"allow": 0, "quarantine": 50, "block": 0},
                        "ocr_success_count": 0,
                        "vision_active_count": 0,
                        "semantic_failed_count": 0,
                        "semantic_latency_active_count": 0,
                        "provider_call_count": {"active_count": 0, "total": 0, "avg": 0.0, "max": 0},
                        "cache_hit_last_request_count": 0,
                        "latency_ms": {"avg": 10.0},
                    }
                },
            }
        ]
    }

    with (
        patch("scripts.eval_wainject_image_balanced_partial.ROOT", tmp_path),
        patch("scripts.eval_wainject_image_balanced_partial._run_backend", return_value={"report_path": "x.json", "report": invalid_report}),
        patch.object(
            sys,
            "argv",
            [
                "eval_wainject_image_balanced_partial.py",
                "--profile",
                "pilot",
                "--root",
                "data/WAInjectBench/image",
                "--mode",
                "vision_single",
                "--api-runtime-mode",
                "stateless",
                "--attachment-visual-enabled",
                "true",
                "--pi0-semantic-enabled",
                "false",
            ],
        ),
    ):
        assert main() == 1
