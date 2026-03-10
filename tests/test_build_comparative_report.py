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
