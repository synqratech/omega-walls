from __future__ import annotations

import json
from pathlib import Path
import uuid

import scripts.check_fw001_perf_gate as gate


class _Proc:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = int(returncode)
        self.stdout = str(stdout)
        self.stderr = ""


def _workspace_tmp(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "fw001_perf_gate_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_baseline(path: Path, short: float, medium: float, large: float) -> None:
    payload = {
        "schema_version": "fw001_perf_baseline_v1",
        "python": "3.13",
        "profile": "dev",
        "metric": "omega_rule_only_pi0.p95_ms",
        "benchmark_args": {"short_chars": 280, "medium_chars": 4200, "large_chars": 12000, "repeats": 4},
        "p95_ms": {"short": short, "medium": medium, "large": large},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_candidate_report(path: Path, short: float, medium: float, large: float) -> None:
    payload = {
        "modes": {
            "omega_rule_only_pi0": {
                "per_size": {
                    "short": {"p95_ms": short},
                    "medium": {"p95_ms": medium},
                    "large": {"p95_ms": large},
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_perf_gate_passes_when_overhead_within_threshold(monkeypatch):
    tmp_dir = _workspace_tmp("pass")
    baseline = tmp_dir / "baseline.json"
    report = tmp_dir / "report.json"
    _write_baseline(baseline, short=100.0, medium=200.0, large=300.0)
    _write_candidate_report(report, short=110.0, medium=225.0, large=330.0)

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        return _Proc(0, json.dumps({"status": "ok", "report_json": str(report)}))

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)
    rc = gate.main(
        [
            "--baseline-file",
            str(baseline),
            "--artifacts-root",
            str(tmp_dir / "artifacts"),
            "--perf-overhead-max",
            "0.15",
            "--strict",
        ]
    )
    assert rc == gate.EXIT_PASS


def test_perf_gate_strict_fails_when_any_size_exceeds_threshold(monkeypatch):
    tmp_dir = _workspace_tmp("fail")
    baseline = tmp_dir / "baseline.json"
    report = tmp_dir / "report.json"
    _write_baseline(baseline, short=100.0, medium=100.0, large=100.0)
    _write_candidate_report(report, short=120.0, medium=114.0, large=115.1)

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        return _Proc(0, json.dumps({"status": "ok", "report_json": str(report)}))

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)
    rc = gate.main(
        [
            "--baseline-file",
            str(baseline),
            "--artifacts-root",
            str(tmp_dir / "artifacts"),
            "--perf-overhead-max",
            "0.15",
            "--strict",
        ]
    )
    assert rc == gate.EXIT_FAIL


def test_perf_gate_returns_error_on_missing_or_malformed_baseline():
    tmp_dir = _workspace_tmp("error")
    baseline = tmp_dir / "baseline_bad.json"
    baseline.write_text(json.dumps({"schema_version": "fw001_perf_baseline_v1"}), encoding="utf-8")
    rc = gate.main(["--baseline-file", str(baseline), "--strict"])
    assert rc == gate.EXIT_ERROR
