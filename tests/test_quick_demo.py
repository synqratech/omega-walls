from __future__ import annotations

import json
from pathlib import Path
import subprocess
import uuid

import scripts.quick_demo as quick_demo


def _cp(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=rc, stdout=stdout, stderr=stderr)


def _workspace_tmp(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "quick_demo_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_quick_demo_missing_api_key_returns_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = quick_demo.main(["--mode", "hybrid_api"])
    captured = capsys.readouterr()
    assert rc == quick_demo.EXIT_MISSING_KEY
    assert "OPENAI_API_KEY" in captured.err


def test_quick_demo_summary_and_semantic_warning(monkeypatch, capsys):
    tmp_path = _workspace_tmp("summary")
    pack = tmp_path / "pack.jsonl"
    pack.write_text("", encoding="utf-8")
    artifacts_root = tmp_path / "artifacts"
    run_dir = artifacts_root / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary_all": {
                    "session_attack_off_rate": 0.75,
                    "session_benign_off_rate": 0.10,
                    "tp": 3,
                },
                "cocktail_metrics": {
                    "mssr_core": 1.0,
                    "mssr_cross_primary": 0.5,
                },
                "projector": {
                    "core_runtime": {"semantic_active": False},
                    "cross_runtime": {"semantic_active": True},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def _fake_run(argv):
        _ = argv
        payload = {"artifacts": {"report_json": str(report_path)}}
        return _cp(0, stdout=json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(quick_demo, "_run_cmd", _fake_run)
    rc = quick_demo.main(
        [
            "--mode",
            "pi0",
            "--pack",
            str(pack),
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    captured = capsys.readouterr()
    assert rc == quick_demo.EXIT_OK
    assert "session_attack_off_rate: 0.7500" in captured.out
    assert "session_benign_off_rate: 0.1000" in captured.out
    assert "mssr_core: 1.0000" in captured.out
    assert "mssr_cross_primary: 0.5000" in captured.out
    assert "blocked behavior observed: yes" in captured.out
    assert "WARNING: semantic fallback active" in captured.out


def test_quick_demo_missing_report_returns_3(monkeypatch, capsys):
    tmp_path = _workspace_tmp("missing_report")
    pack = tmp_path / "pack.jsonl"
    pack.write_text("", encoding="utf-8")
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    def _fake_run(argv):
        _ = argv
        payload = {"artifacts": {"report_json": str(tmp_path / "missing_report.json")}}
        return _cp(0, stdout=json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(quick_demo, "_run_cmd", _fake_run)
    rc = quick_demo.main(
        [
            "--mode",
            "pi0",
            "--pack",
            str(pack),
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    captured = capsys.readouterr()
    assert rc == quick_demo.EXIT_RUNTIME_ERROR
    assert "report.json not found" in captured.err


def test_quick_demo_agentdojo_runs_mode_calls_builder_and_eval(monkeypatch):
    tmp_path = _workspace_tmp("agentdojo_runs")
    artifacts_root = tmp_path / "artifacts"
    report_path = artifacts_root / "demo_run" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "summary_all": {
                    "session_attack_off_rate": 1.0,
                    "session_benign_off_rate": 0.0,
                    "tp": 1,
                },
                "cocktail_metrics": {
                    "mssr_core": 1.0,
                    "mssr_cross_primary": 1.0,
                },
                "projector": {
                    "core_runtime": {"semantic_active": True},
                    "cross_runtime": {"semantic_active": True},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(argv):
        calls.append(list(argv))
        cmd = " ".join(argv)
        if "build_agentdojo_cocktail_mini_pack.py" in cmd:
            out_idx = argv.index("--out") + 1
            out_pack = Path(argv[out_idx])
            out_pack.parent.mkdir(parents=True, exist_ok=True)
            out_pack.write_text("", encoding="utf-8")
            payload = {"status": "ok"}
            return _cp(0, stdout=json.dumps(payload, ensure_ascii=False))
        payload = {"artifacts": {"report_json": str(report_path)}}
        return _cp(0, stdout=json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(quick_demo, "_run_cmd", _fake_run)
    rc = quick_demo.main(
        [
            "--mode",
            "pi0",
            "--dataset-source",
            "agentdojo_runs",
            "--artifacts-root",
            str(artifacts_root),
            "--agentdojo-runs-root",
            "data/AgentDojo/runs",
        ]
    )

    assert rc == quick_demo.EXIT_OK
    assert len(calls) == 2
    assert "build_agentdojo_cocktail_mini_pack.py" in " ".join(calls[0])
    assert "eval_agentdojo_stateful_mini.py" in " ".join(calls[1])


def test_quick_demo_pi0_integration_smoke(capsys):
    tmp_path = _workspace_tmp("integration")
    rc = quick_demo.main(
        [
            "--mode",
            "pi0",
            "--pack",
            "tests/data/session_benchmark/agentdojo_cocktail_mini_smoke_v1.jsonl",
            "--artifacts-root",
            str(tmp_path / "quick_demo_artifacts"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == quick_demo.EXIT_OK
    assert "Quick Demo Summary" in captured.out
    assert "session_attack_off_rate" in captured.out
