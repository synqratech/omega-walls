from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import jsonschema
import pytest

from scripts import build_enterprise_attack_dashboard as dashboard
from scripts import build_enterprise_incident_backlog as incident
from scripts import build_enterprise_weekly_report as weekly


def _write_run_artifact(run_dir: Path, run_id: str, rows: list[dict], *, attack_sessions: int, benign_sessions: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "summary_all": {
            "sessions_total": int(attack_sessions + benign_sessions),
            "attack_sessions": int(attack_sessions),
            "benign_sessions": int(benign_sessions),
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_rows() -> list[dict]:
    return [
        # TP session: attack detected.
        {
            "session_id": "s_atk_tp",
            "bucket": "core",
            "family": "cocktail",
            "turn_id": 1,
            "label_turn": "attack",
            "label_session": "attack",
            "off": True,
            "max_p": 1.0,
        },
        # FN session: attack missed.
        {
            "session_id": "s_atk_fn",
            "bucket": "cross_session",
            "family": "distributed_wo_explicit",
            "turn_id": 1,
            "label_turn": "attack",
            "label_session": "attack",
            "off": False,
            "max_p": 0.8,
        },
        # TN session: benign not blocked.
        {
            "session_id": "s_ben_tn",
            "bucket": "core",
            "family": "benign_long_context",
            "turn_id": 1,
            "label_turn": "benign",
            "label_session": "benign",
            "off": False,
            "max_p": 0.0,
        },
        # FP session: benign blocked.
        {
            "session_id": "s_ben_fp",
            "bucket": "core",
            "family": "benign_long_context",
            "turn_id": 1,
            "label_turn": "benign",
            "label_session": "benign",
            "off": True,
            "max_p": 0.0,
        },
    ]


def _make_tmp_dir(prefix: str) -> Path:
    base = Path("enterprise/monitoring/out/.tmp_enterprise_tests")
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{prefix}{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_dashboard_aggregation_and_clusters_and_schema():
    tmp_path = _make_tmp_dir("dashboard_")
    try:
        run_dir = tmp_path / "run_one"
        _write_run_artifact(
            run_dir,
            run_id="agentdojo_stateful_mini_eval_20260403T010101Z",
            rows=_sample_rows(),
            attack_sessions=2,
            benign_sessions=2,
        )

        runs = dashboard.discover_stateful_runs(tmp_path)
        assert len(runs) == 1

        outcomes = dashboard.load_session_outcomes(runs[0].rows_jsonl, runs[0].run_id)
        feed = dashboard.build_dashboard_feed(
            runs=runs,
            outcomes=outcomes,
            window_days=7,
            baseline={"status": "provided", "attack_off_rate": 0.75, "benign_off_rate": 0.40},
        )

        assert feed["kpi"]["confusion"]["tp"] == 1
        assert feed["kpi"]["confusion"]["fn"] == 1
        assert feed["kpi"]["confusion"]["fp"] == 1
        assert feed["kpi"]["confusion"]["tn"] == 1
        assert feed["kpi"]["attack_off_rate"] == pytest.approx(0.5)
        assert feed["kpi"]["benign_off_rate"] == pytest.approx(0.5)

        fn_clusters = feed["mismatch_clusters"]["fn_clusters"]
        assert len(fn_clusters) >= 1
        assert fn_clusters[0]["cluster_key"].startswith("distributed_wo_explicit|cross_session|")

        schema_path = Path("schemas/enterprise_attack_dashboard_feed_v1.json")
        dashboard.validate_feed_schema(feed, schema_path)

        bad_feed = dict(feed)
        bad_feed.pop("kpi", None)
        with pytest.raises(jsonschema.ValidationError):
            dashboard.validate_feed_schema(bad_feed, schema_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_incident_backlog_severity_assignment():
    feed = {
        "kpi": {
            "attack_volume": {"attack_sessions": 250, "attack_turn_events": 400},
            "attack_off_rate": 0.60,
            "benign_off_rate": 0.03,
        },
        "mismatch_clusters": {
            "fn_clusters": [
                {
                    "cluster_key": "cocktail|core|no_off:maxp>0.20",
                    "family": "cocktail",
                    "bucket": "core",
                    "reason_signature": "no_off:maxp>0.20",
                    "count": 12,
                    "sample_session_ids": ["s1", "s2"],
                    "sample_run_ids": ["r1"],
                }
            ],
            "fp_clusters": [
                {
                    "cluster_key": "benign_long_context|core|false_off:turn:1",
                    "family": "benign_long_context",
                    "bucket": "core",
                    "reason_signature": "false_off:turn:1",
                    "count": 2,
                    "sample_session_ids": ["s3"],
                    "sample_run_ids": ["r1"],
                }
            ],
        },
        "alerts": [
            {
                "id": "alert_attack_off_rate_breach",
                "type": "attack_off_rate_breach",
                "metric": "attack_off_rate",
                "message": "attack_off_rate is below baseline by more than 1.0 p.p.",
                "value": 0.6,
                "baseline": 0.8,
                "delta_pp": -20.0,
            }
        ],
    }

    backlog = incident.build_incident_backlog(
        dashboard_feed=feed,
        dashboard_feed_path="enterprise/monitoring/out/dashboard_feed.json",
        high_volume_attack_sessions=100,
        sev2_cluster_count_threshold=8,
    )
    incidents = backlog["incidents"]
    assert incidents
    assert any(x["severity"] == "SEV-1" and x["category"] == "metric_breach" for x in incidents)
    assert any(x["category"] == "fn_cluster" for x in incidents)
    assert all("owner" in x and "sla" in x and "escalation_path" in x for x in incidents)


def test_weekly_report_baseline_missing_and_provided():
    feed = {
        "kpi": {
            "attack_volume": {"attack_sessions": 100, "attack_turn_events": 180},
            "family_split": [{"family": "cocktail", "attack_sessions": 80, "share": 0.8}],
            "attack_off_rate": 0.82,
            "benign_off_rate": 0.02,
            "confusion": {"tp": 82, "fp": 2, "tn": 98, "fn": 18, "attack_sessions": 100, "benign_sessions": 100},
        },
        "mismatch_clusters": {"fn_clusters": [], "fp_clusters": []},
        "alerts": [],
    }
    backlog = {"incidents": []}

    summary_missing = weekly.build_weekly_summary(
        dashboard_feed=feed,
        incident_backlog=backlog,
        baseline={"status": "missing"},
        week_token="2026_14",
        dashboard_feed_path="dashboard.json",
        incident_backlog_path="backlog.json",
    )
    assert summary_missing["baseline_regression"]["baseline_status"] == "missing"

    summary_with_baseline = weekly.build_weekly_summary(
        dashboard_feed=feed,
        incident_backlog=backlog,
        baseline={"status": "provided", "source": "baseline.json", "attack_off_rate": 0.90, "benign_off_rate": 0.01},
        week_token="2026_14",
        dashboard_feed_path="dashboard.json",
        incident_backlog_path="backlog.json",
    )
    reg = summary_with_baseline["baseline_regression"]
    assert reg["baseline_status"] == "provided"
    assert reg["attack_off_rate_delta_pp"] == pytest.approx(-8.0)
    assert reg["benign_off_rate_delta_pp"] == pytest.approx(1.0)
    assert reg["alert_attack_off_rate_breach"] is True
    assert reg["alert_benign_off_rate_breach"] is True

    md = weekly.render_weekly_markdown(summary_with_baseline)
    assert "## Executive Summary" in md
    assert "## KPI Trends" in md
    assert "## Baseline Regression" in md
    assert "## Critical Cases" in md
    assert "## Top Mismatch Clusters" in md
    assert "## Next Week Plan" in md


def test_weekly_baseline_path_fallback():
    tmp_path = _make_tmp_dir("baseline_")
    try:
        weekly_dir = tmp_path / "weekly"
        weekly_dir.mkdir(parents=True, exist_ok=True)
        old = weekly_dir / "weekly_security_report_2026_10.json"
        new = weekly_dir / "weekly_security_report_2026_11.json"
        old.write_text(json.dumps({"kpi": {"attack_off_rate": 0.7, "benign_off_rate": 0.02}}), encoding="utf-8")
        new.write_text(json.dumps({"kpi": {"attack_off_rate": 0.8, "benign_off_rate": 0.01}}), encoding="utf-8")

        chosen = weekly._resolve_baseline_path(explicit_path=None, weekly_dir=weekly_dir, exclude_path=None)
        assert chosen is not None
        assert chosen.name == "weekly_security_report_2026_11.json"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_integration_smoke_real_artifacts():
    runs_root = Path("artifacts/agentdojo_prepilot_blind_20260330_193616")
    if not runs_root.exists():
        pytest.skip("artifact-driven integration smoke requires local artifacts/agentdojo_prepilot_blind_20260330_193616")

    tmp_path = _make_tmp_dir("integration_")
    try:
        runs = dashboard.discover_stateful_runs(runs_root)
        if not runs:
            pytest.skip("no valid run dirs with report.json + rows.jsonl")

        outcomes = []
        for run in runs:
            outcomes.extend(dashboard.load_session_outcomes(run.rows_jsonl, run.run_id))

        feed = dashboard.build_dashboard_feed(
            runs=runs,
            outcomes=outcomes,
            window_days=3650,
            baseline={"status": "missing"},
        )
        dashboard.validate_feed_schema(feed, Path("schemas/enterprise_attack_dashboard_feed_v1.json"))

        feed_path = tmp_path / "dashboard_feed.json"
        html_path = tmp_path / "dashboard.html"
        feed_path.write_text(json.dumps(feed, ensure_ascii=False, indent=2), encoding="utf-8")
        html_path.write_text(dashboard.render_dashboard_html(feed), encoding="utf-8")
        assert feed_path.exists()
        assert html_path.exists()

        backlog = incident.build_incident_backlog(
            dashboard_feed=feed,
            dashboard_feed_path=str(feed_path),
            high_volume_attack_sessions=100,
            sev2_cluster_count_threshold=8,
        )
        backlog_path = tmp_path / "incident_backlog.json"
        backlog_path.write_text(json.dumps(backlog, ensure_ascii=False, indent=2), encoding="utf-8")
        assert backlog_path.exists()

        summary = weekly.build_weekly_summary(
            dashboard_feed=feed,
            incident_backlog=backlog,
            baseline={"status": "missing"},
            week_token="2026_14",
            dashboard_feed_path=str(feed_path),
            incident_backlog_path=str(backlog_path),
        )
        md = weekly.render_weekly_markdown(summary)
        summary_path = tmp_path / "weekly_security_report_2026_14.json"
        md_path = tmp_path / "weekly_security_report_2026_14.md"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(md, encoding="utf-8")

        assert summary_path.exists()
        assert md_path.exists()
        assert "## Executive Summary" in md
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
