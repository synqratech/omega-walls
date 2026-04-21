from __future__ import annotations

from scripts.build_comparative_report import build_comparative_payload


def test_build_comparative_payload_sections():
    contour = {
        "artifacts": {"manifest_json": "artifacts/post_patch_contour/run/manifest.json"},
        "reports": {
            "rule_cycle": {"status": "OK", "summary": {"attack_off_rate": 0.75}},
            "strict_pi": {"status": "ok", "summary": {"attack_off_rate": 0.94}},
            "attachment": {"status": "ok", "summary": {"attack_off_rate": 0.8}},
        },
    }
    pint = {
        "dataset_ready": True,
        "summary": {"pint_score_pct": 88.0, "attack_off_rate": 0.9, "benign_off_rate": 0.0},
        "external_benchmark": {
            "source_url": "https://github.com/lakeraai/pint-benchmark",
            "source_date": "2026-03-08",
            "metric_mapping": {"benchmark_metric": "PINT score"},
            "public_scoreboard": [{"system": "Lakera Guard", "pint_score_pct": 92.5461}],
        },
    }
    wainject = {
        "dataset_ready": True,
        "comparability_status": "partial_comparison",
        "comparability_reason": "full_run_complete_benign_malicious_splits",
        "summary": {"attack_off_rate": 0.82, "benign_off_rate": 0.0},
        "external_benchmark": {
            "source_url": "https://github.com/Norrrrrrr-lyn/WAInjectBench",
            "source_date": "2026-03-08",
            "metric_mapping": {"external_baseline_table_available": False},
        },
    }
    payload = build_comparative_payload(run_id="cmp_test", contour_manifest=contour, pint_report=pint, wainject_report=wainject)
    assert payload["run_id"] == "cmp_test"
    assert len(payload["direct_comparison"]) == 1
    assert payload["direct_comparison"][0]["benchmark"] == "PINT"
    assert len(payload["partial_comparison"]) == 1
    assert payload["partial_comparison"][0]["benchmark"] == "WAInjectBench"


def test_build_comparative_payload_uses_wainject_status_from_report():
    contour = {"artifacts": {"manifest_json": "x"}, "reports": {}}
    pint = {
        "dataset_ready": False,
        "summary": {},
        "external_benchmark": {"source_url": "https://github.com/lakeraai/pint-benchmark", "source_date": "2026-03-08", "metric_mapping": {}},
    }
    # dataset_ready intentionally true, but comparability_status forced to non_comparable
    wainject = {
        "dataset_ready": True,
        "comparability_status": "non_comparable",
        "comparability_reason": "subsampled_run_max_samples",
        "summary": {"attack_off_rate": 0.7, "benign_off_rate": 0.0},
        "external_benchmark": {
            "source_url": "https://github.com/Norrrrrrr-lyn/WAInjectBench",
            "source_date": "2026-03-08",
            "metric_mapping": {"external_baseline_table_available": False},
        },
    }
    payload = build_comparative_payload(run_id="cmp_noncomp", contour_manifest=contour, pint_report=pint, wainject_report=wainject)
    assert len(payload["partial_comparison"]) == 0
    assert any(x.get("benchmark") == "WAInjectBench" for x in payload["non_comparable"])
    assert any("subsampled_run_max_samples" in str(x) for x in payload["methodology_limits"])


def test_build_comparative_payload_adds_promptshield_as_non_comparable():
    contour = {"artifacts": {"manifest_json": "x"}, "reports": {}}
    pint = {
        "dataset_ready": False,
        "summary": {},
        "external_benchmark": {"source_url": "https://github.com/lakeraai/pint-benchmark", "source_date": "2026-03-08", "metric_mapping": {}},
    }
    wainject = {
        "dataset_ready": True,
        "comparability_status": "partial_comparison",
        "summary": {"attack_off_rate": 0.8, "benign_off_rate": 0.01},
        "external_benchmark": {
            "source_url": "https://github.com/Norrrrrrr-lyn/WAInjectBench",
            "source_date": "2026-03-08",
            "metric_mapping": {"external_baseline_table_available": False},
        },
    }
    promptshield = {
        "comparability_status": "non_comparable",
        "comparability_reason": "no_benchmark_maintainer_detector_leaderboard",
        "summary": {"attack_off_rate": 0.4, "benign_off_rate": 0.1},
        "external_benchmark": {"source_url": None, "source_date": "2026-03-12", "metric_mapping": {}},
    }
    payload = build_comparative_payload(
        run_id="cmp_promptshield_noncomp",
        contour_manifest=contour,
        pint_report=pint,
        wainject_report=wainject,
        promptshield_report=promptshield,
    )
    assert any(x.get("benchmark") == "PromptShield" for x in payload["non_comparable"])
    assert any("PromptShield marked non-comparable" in str(x) for x in payload["methodology_limits"])
