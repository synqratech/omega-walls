from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

from fastapi.testclient import TestClient


SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "owl-telemetry"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from owl_telemetry.app import create_app  # noqa: E402
from owl_telemetry.settings import Settings  # noqa: E402


def _instance_id(seed: str) -> str:
    return (seed * 64)[:64]


def _batch(
    *,
    instance_id: str,
    batch_id: str,
    version: str = "0.1.5",
    attack_type: str = "prompt_injection",
    attack_count: int = 1,
    signature_hash: str = "a" * 64,
) -> dict:
    return {
        "schema_version": "telemetry_batch_v1",
        "batch_id": batch_id,
        "instance": {
            "instance_id": instance_id,
            "core_version": version,
            "tier": "oss",
            "deployment_mode": "sidecar",
            "os_arch": "linux-x86_64",
            "python_version": "3.11.9",
        },
        "window": "24h",
        "product_metrics": {
            "uptime_bucket": "1-6",
            "modules_enabled": ["telemetry", "orchestrator"],
            "fallback_events_count": 0,
            "config_changes_count": 0,
            "last_sync_status": "ok",
        },
        "security_analytics": {
            "attack_type_counts": {attack_type: attack_count},
            "policy_triggers": {"reason_sum": attack_count},
            "tool_abuse_categories": {"financial": attack_count},
            "risk_score_buckets": [0.9],
            "fp_reports_count": 0,
            "enforcement_actions": {"blocked": attack_count},
        },
        "attack_signatures": [
            {
                "pattern_hash": signature_hash,
                "rule_id": "rule-x",
                "accumulation_steps": 3,
                "provenance_type": "tool_output",
                "count": attack_count,
            }
        ],
    }


def _settings(tmp_path: Path, *, limit: int = 10) -> Settings:
    return Settings(
        service_port=8000,
        database_path=str((tmp_path / "analytics.db").as_posix()),
        api_key="test-key",
        discord_webhook_url="",
        anon_salt="salt",
        rate_limit_requests=limit,
        rate_limit_window=60,
        weekly_report_cron="0 9 * * 1",
        timezone="UTC",
        log_level="INFO",
        test_mode=True,
        burst_alert_min_events=5,
        burst_alert_window_minutes=10,
        burst_alert_cooldown_minutes=10,
        schema_path=str((SERVICE_ROOT / "schemas" / "telemetry_batch_v1.json").as_posix()),
    )


def _headers() -> dict[str, str]:
    return {"X-API-Key": "test-key", "Content-Type": "application/json"}


def test_collect_accept_duplicate_and_stats(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        payload = _batch(instance_id=_instance_id("a"), batch_id="batch-0001")
        r1 = client.post("/v1/collect", headers=_headers(), json=payload)
        assert r1.status_code == 202
        r2 = client.post("/v1/collect", headers=_headers(), json=payload)
        assert r2.status_code == 409
        stats = client.get("/api/v1/stats/summary", params={"window": "7d"})
        assert stats.status_code == 200
        body = stats.json()
        assert int(body["total_instances"]) == 1
        assert int(body["total_events"]) >= 1


def test_collect_auth_schema_and_rate_limit(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, limit=2))
    with TestClient(app) as client:
        payload = _batch(instance_id=_instance_id("b"), batch_id="batch-1001")
        assert client.post("/v1/collect", json=payload).status_code == 401

        bad = _batch(instance_id="short", batch_id="batch-bad-1")
        assert client.post("/v1/collect", headers=_headers(), json=bad).status_code == 400

        ok1 = client.post("/v1/collect", headers=_headers(), json=_batch(instance_id=_instance_id("b"), batch_id="batch-1002"))
        ok2 = client.post("/v1/collect", headers=_headers(), json=_batch(instance_id=_instance_id("b"), batch_id="batch-1003"))
        limited = client.post("/v1/collect", headers=_headers(), json=_batch(instance_id=_instance_id("b"), batch_id="batch-1004"))
        assert ok1.status_code == 202
        assert ok2.status_code == 202
        assert limited.status_code == 429


def test_forbidden_keys_are_stripped_before_write(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        payload = _batch(instance_id=_instance_id("c"), batch_id="batch-2001")
        payload["session_id"] = "forbidden-top"
        payload["product_metrics"]["hostname"] = "forbidden-host"
        payload["security_analytics"]["prompt"] = "forbidden-prompt"
        r = client.post("/v1/collect", headers=_headers(), json=payload)
        assert r.status_code == 202

    conn = sqlite3.connect(str((tmp_path / "analytics.db").as_posix()))
    row = conn.execute("SELECT payload_json FROM batches LIMIT 1").fetchone()
    conn.close()
    stored = json.loads(row[0])

    def _has_forbidden_key(value: object) -> bool:
        forbidden = {"session_id", "hostname", "prompt"}
        if isinstance(value, dict):
            for k, v in value.items():
                if str(k).lower() in forbidden:
                    return True
                if _has_forbidden_key(v):
                    return True
            return False
        if isinstance(value, list):
            return any(_has_forbidden_key(item) for item in value)
        return False

    assert not _has_forbidden_key(stored)


def test_alert_dedupe_for_new_attack_pattern(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    sig = "b" * 64
    with TestClient(app) as client:
        r1 = client.post(
            "/v1/collect",
            headers=_headers(),
            json=_batch(instance_id=_instance_id("d"), batch_id=f"b-{uuid4().hex[:8]}", signature_hash=sig),
        )
        r2 = client.post(
            "/v1/collect",
            headers=_headers(),
            json=_batch(instance_id=_instance_id("d"), batch_id=f"b-{uuid4().hex[:8]}", signature_hash=sig),
        )
        assert r1.status_code == 202
        assert r2.status_code == 202

    conn = sqlite3.connect(str((tmp_path / "analytics.db").as_posix()))
    rows = conn.execute(
        "SELECT event_type, COUNT(*) FROM alerts_log GROUP BY event_type ORDER BY event_type"
    ).fetchall()
    conn.close()
    counts = {str(k): int(v) for k, v in rows}
    assert counts.get("new_instance", 0) == 1
    assert counts.get("new_attack_pattern", 0) == 1


def test_weekly_digest_endpoint_writes_aggregate(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        client.post("/v1/collect", headers=_headers(), json=_batch(instance_id=_instance_id("e"), batch_id="week-0001"))
        res = client.post("/internal/weekly-digest-now")
        assert res.status_code == 200
        body = res.json()
        assert "week_start" in body
        assert "summary" in body

    conn = sqlite3.connect(str((tmp_path / "analytics.db").as_posix()))
    row = conn.execute("SELECT COUNT(*) FROM weekly_aggregates").fetchone()
    conn.close()
    assert int(row[0]) >= 1
