from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.eval_wainject_image_ocr_slice import (
    ImageEvalRow,
    ImageSample,
    _build_runtime,
    _deep_merge_dict,
    _opaque_run_namespace,
    _mode_overrides,
    _scan_image,
    collect_image_samples,
    summarize_phase_metrics,
)


def _mk_row(
    *,
    phase: str,
    verdict: str,
    latency_ms: float,
    provider_call_count: int,
    cache_hit_last_request: bool,
    is_worker_first_request: bool,
    name: str = "1.png",
) -> ImageEvalRow:
    return ImageEvalRow(
        mode="vision_single",
        repeat=1,
        phase=phase,
        worker_id=0,
        worker_ordinal=(1 if is_worker_first_request else 2),
        is_worker_first_request=is_worker_first_request,
        name=name,
        sha256="ab" * 32,
        verdict=verdict,
        risk_score=0,
        latency_ms=latency_ms,
        semantic_status="ok",
        vision_status="vision_semantic_active",
        ocr_status="disabled",
        ocr_provider="rapidocr",
        ocr_quality_status="none",
        ocr_kept_spans=0,
        ocr_dropped_low_confidence=0,
        ocr_geometry_coverage_ratio=0.0,
        ocr_active_walls=[],
        ocr_adjudication_result="not_attempted",
        region_trigger_reason="disabled",
        region_variant_count=0,
        provider_call_count=provider_call_count,
        retry_count=0,
        cache_hit_last_request=cache_hit_last_request,
        vision_fallback_used=False,
        second_pass_attempted=False,
        second_pass_result="not_attempted",
        first_pass_latency_ms=latency_ms,
        second_pass_latency_ms=None,
        semantic_latency_ms=latency_ms,
        token_usage={},
        reasons=[],
    )


def test_collect_image_samples_dedupes_by_sha256(tmp_path: Path):
    (tmp_path / "1.png").write_bytes(b"same-image")
    (tmp_path / "2.png").write_bytes(b"same-image")
    (tmp_path / "3.png").write_bytes(b"other-image")

    samples, duplicates = collect_image_samples(tmp_path, max_samples=10, dedupe_by_sha256=True)

    assert [sample.path.name for sample in samples] == ["1.png", "3.png"]
    assert duplicates == [
        {
            "sha256": samples[0].sha256,
            "kept": "1.png",
            "dropped": ["2.png"],
        }
    ]


def test_summarize_phase_metrics_splits_cold_warm_and_cache():
    rows = [
        _mk_row(
            phase="fresh",
            verdict="allow",
            latency_ms=100.0,
            provider_call_count=1,
            cache_hit_last_request=False,
            is_worker_first_request=True,
            name="cold.png",
        ),
        _mk_row(
            phase="fresh",
            verdict="block",
            latency_ms=80.0,
            provider_call_count=1,
            cache_hit_last_request=False,
            is_worker_first_request=False,
            name="warm.png",
        ),
        _mk_row(
            phase="cache_replay",
            verdict="block",
            latency_ms=5.0,
            provider_call_count=0,
            cache_hit_last_request=True,
            is_worker_first_request=False,
            name="cache.png",
        ),
    ]

    summary = summarize_phase_metrics(rows)

    assert summary["quality_fresh"]["total"] == 2
    assert summary["quality_fresh"]["detected_rate"] == 0.5
    assert summary["quality_fresh"]["provider_call_count"]["active_count"] == 2
    assert summary["quality_fresh"]["provider_call_count"]["total"] == 2
    assert summary["quality_fresh"]["semantic_latency_active_count"] == 2
    assert summary["cold_first_request"]["total"] == 1
    assert summary["cold_first_request"]["latency_ms"]["p50"] == 100.0
    assert summary["warm_fresh_provider_calls"]["total"] == 1
    assert summary["warm_fresh_provider_calls"]["latency_ms"]["p50"] == 80.0
    assert summary["cache_hit_replay"]["total"] == 1
    assert summary["cache_hit_replay"]["latency_ms"]["p50"] == 5.0


def test_scan_image_includes_stateful_session_id(tmp_path: Path):
    image_path = tmp_path / "1.png"
    image_path.write_bytes(b"fake-image")
    sample = ImageSample(path=image_path, sha256="ab" * 32)
    captured = {}

    def _fake_scan_request(runtime, payload):
        _ = runtime
        captured.update(payload)
        return {
            "verdict": "allow",
            "risk_score": 0,
            "policy_trace": {},
            "reasons": [],
        }

    with patch("scripts.eval_wainject_image_ocr_slice.api_server._scan_request", side_effect=_fake_scan_request):
        row = _scan_image(
            runtime=object(),
            sample=sample,
            idx=7,
            run_namespace="abc123def0",
            mode_label="vision_plus_ocr",
            repeat=1,
            phase="fresh",
            worker_id=0,
            worker_ordinal=1,
        )

    assert captured["tenant_id"] == "wainject-image-eval-abc123def0-007"
    assert captured["session_id"] == "img-eval-abc123def0-r01-w00-s007"
    assert captured["request_id"].startswith("abc123def0-vision_plus_ocr-fresh-r01-w00-007-")
    assert row.name == "1.png"


def test_mode_overrides_do_not_force_provider():
    overrides = _mode_overrides("vision_plus_ocr", ocr_provider="rapidocr")
    api_cfg = ((overrides.get("projector") or {}).get("api_perception") or {})

    assert "provider" not in api_cfg


def test_deep_merge_dict_preserves_nested_visual_and_ocr_settings():
    merged = _deep_merge_dict(
        {
            "retriever": {
                "sqlite_fts": {
                    "attachments": {
                        "visual": {"enabled": "true"},
                    }
                }
            }
        },
        {
            "retriever": {
                "sqlite_fts": {
                    "attachments": {
                        "ocr": {"enabled": "false", "provider": "rapidocr"},
                    }
                }
            }
        },
    )

    attachments = (((merged.get("retriever") or {}).get("sqlite_fts") or {}).get("attachments") or {})
    assert ((attachments.get("visual") or {}).get("enabled")) == "true"
    assert ((attachments.get("ocr") or {}).get("enabled")) == "false"
    assert ((attachments.get("ocr") or {}).get("provider")) == "rapidocr"


def test_build_runtime_passes_stateless_visual_overrides_to_loader():
    captured = {}

    def _fake_load_resolved_config(*, profile, cli_overrides):
        captured["profile"] = profile
        captured["cli_overrides"] = cli_overrides

        class _Snapshot:
            resolved = {"ok": True}

        return _Snapshot()

    with (
        patch("scripts.eval_wainject_image_ocr_slice.load_resolved_config", side_effect=_fake_load_resolved_config),
        patch("scripts.eval_wainject_image_ocr_slice.api_server._make_runtime", return_value=object()),
    ):
        _build_runtime(
            profile="sensitive_rules_only",
            mode="vision_single",
            ocr_provider="rapidocr",
            pi0_semantic_enabled="false",
            pi0_semantic_device="auto",
            cache_namespace="prod_default_smoke",
            api_runtime_mode="stateless",
            attachment_visual_enabled="true",
        )

    overrides = captured["cli_overrides"]
    attachments = (((overrides.get("retriever") or {}).get("sqlite_fts") or {}).get("attachments") or {})
    assert captured["profile"] == "sensitive_rules_only"
    assert (((overrides.get("api") or {}).get("runtime") or {}).get("mode")) == "stateless"
    assert ((attachments.get("visual") or {}).get("enabled")) == "true"
    assert ((attachments.get("ocr") or {}).get("enabled")) == "false"


def test_opaque_run_namespace_is_stable_and_blind():
    ns = _opaque_run_namespace("staging/benign_w01")
    assert len(ns) == 10
    assert ns != "staging/benign_w01"
