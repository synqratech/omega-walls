from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def _run_json_script(rel_path: str, *, timeout_sec: int = 60) -> tuple[Dict[str, Any], float, str]:
    script_path = ROOT / rel_path
    assert script_path.exists(), f"missing script: {script_path}"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = float(time.perf_counter() - t0)
    if proc.returncode != 0:
        raise AssertionError(
            f"{rel_path} failed with code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{rel_path} did not output valid JSON.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"{rel_path} JSON output must be an object.")
    return payload, elapsed, proc.stderr


def test_monitor_quickstart_example() -> None:
    payload, elapsed, _stderr = _run_json_script("examples/reliability_quickstart/monitor_quickstart_demo.py")
    assert elapsed < 60.0
    assert payload.get("status") == "ok"
    assert str(payload.get("events_path", "")).strip()
    samples = payload.get("samples", {})
    assert isinstance(samples, dict)
    attack = samples.get("attack", {})
    assert isinstance(attack, dict)
    assert str(attack.get("actual_action", "")).upper() == "ALLOW"
    assert str(attack.get("intended_action", "")).upper() != "ALLOW"
    report = payload.get("report", {})
    assert isinstance(report, dict)
    assert int(report.get("total_checks", 0)) >= 2


def test_explain_timeline_example() -> None:
    payload, elapsed, _stderr = _run_json_script("examples/reliability_quickstart/explain_timeline_demo.py")
    assert elapsed < 60.0
    assert payload.get("status") == "ok"
    assert str(payload.get("session_id", "")).strip()
    assert int(payload.get("timeline_len", 0)) >= 3
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert int(summary.get("events_count", 0)) >= 3
    mttd = payload.get("mttd", {})
    assert isinstance(mttd, dict)
    assert "first_non_allow_index" in mttd


def test_workflow_continuity_example() -> None:
    payload, elapsed, _stderr = _run_json_script("examples/reliability_quickstart/workflow_continuity_demo.py")
    assert elapsed < 60.0
    assert payload.get("status") == "ok"
    routes = payload.get("routes", [])
    assert isinstance(routes, list)
    assert len(routes) >= 3
    route_labels = {str(row.get("route", "")) for row in routes if isinstance(row, dict)}
    assert "ALLOW" in route_labels
    assert bool(route_labels & {"REDACT_AND_CONTINUE", "ESCALATE"})
    for row in routes:
        if not isinstance(row, dict):
            continue
        assert str(row.get("actual_action", "")).upper() == "ALLOW"
