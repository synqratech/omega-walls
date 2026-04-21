from __future__ import annotations

import csv
import json
from pathlib import Path
import uuid

import scripts.run_benchmark as bench


def _workspace_tmp(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "run_benchmark_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _latest_report(artifacts_root: Path) -> Path:
    reports = sorted(artifacts_root.rglob("report.json"))
    assert reports
    return reports[-1]


def _suite_payload(name: str) -> dict:
    if name == "run_eval":
        return {
            "hard_negatives": {"count": 50, "fp": 0},
            "canonical_positives": {
                "count": 40,
                "overall_hit": 0.95,
                "per_wall": {
                    "override_instructions": 1.0,
                    "secret_exfiltration": 0.9,
                    "tool_or_action_abuse": 1.0,
                    "policy_evasion": 0.95,
                },
            },
        }
    if name == "run_attack_layer_cycle":
        return {
            "status": "ok",
            "summary": {
                "attack_off_rate": 0.8,
                "benign_off_rate": 0.1,
                "utility_preservation": 0.9,
            },
            "gate": {"passed": True, "checks": [{"id": "ALV1-X", "status": "PASS"}]},
        }
    if name == "eval_support_stateful_vs_stateless":
        return {
            "metrics": {
                "overall": {
                    "stateful_target": {
                        "session_attack_off_rate": 0.8,
                        "session_benign_off_rate": 0.1,
                        "late_detect_rate": 0.2,
                        "steps_to_off": {"median": 2.0, "p90": 4.0},
                    },
                    "baseline_d_bare_llm_detector": {
                        "session_attack_off_rate": 0.7,
                        "session_benign_off_rate": 0.12,
                        "late_detect_rate": 0.25,
                        "steps_to_off": {"median": 3.0, "p90": 4.5},
                    },
                    "baseline_a_per_turn_stateless": {
                        "session_attack_off_rate": 0.72,
                        "session_benign_off_rate": 0.11,
                        "late_detect_rate": 0.21,
                        "steps_to_off": {"median": 3.0, "p90": 4.0},
                    },
                }
            },
            "market_ready": {
                "variants": {
                    "stateful_target": {"enforcement_scope": "native_policy_actions"},
                    "baseline_d_bare_llm_detector": {"enforcement_scope": "detector_only_not_comparable"},
                    "baseline_a_per_turn_stateless": {"enforcement_scope": "native_policy_actions"},
                }
            },
        }
    raise AssertionError(f"unexpected suite: {name}")


def test_build_dataset_manifest_core_profile():
    manifest = bench.build_dataset_manifest(
        registry_path=Path("config/benchmark_datasets.yml"),
        dataset_profile="core_oss_v1",
        seed=41,
        runtime_mode="pi0",
        config_snapshot_sha256="abc123",
    )
    assert manifest["dataset_profile"] == "core_oss_v1"
    assert manifest["dataset_count"] >= 7
    for row in manifest["datasets"]:
        assert row["source_url"]
        assert row["sha256"]
        assert Path(row["path"]).exists()


def test_scorecard_writer_is_deterministic():
    tmp = _workspace_tmp("scorecard")
    rows = [
        {
            "run_id": "r1",
            "suite": "support_compare",
            "variant": "stateful_target",
            "metric": "attack_off_rate",
            "value": 0.8,
            "unit": "rate",
            "direction": "higher_is_better",
            "threshold": "",
            "status": "ok",
            "comparable_scope": "comparable",
        },
        {
            "run_id": "r1",
            "suite": "support_compare",
            "variant": "baseline_d_bare_llm_detector",
            "metric": "attack_off_rate",
            "value": 0.7,
            "unit": "rate",
            "direction": "higher_is_better",
            "threshold": "",
            "status": "ok",
            "comparable_scope": "detector_only_not_comparable",
        },
    ]
    a = tmp / "a.csv"
    b = tmp / "b.csv"
    bench._write_scorecard(a, rows)  # noqa: SLF001
    bench._write_scorecard(b, rows)  # noqa: SLF001
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    with a.open("r", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert len(parsed) == 2


def test_main_smoke_with_mocked_suites(monkeypatch):
    tmp = _workspace_tmp("main_smoke")
    artifacts_root = tmp / "artifacts"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _fake_run(name: str, argv, out_dir: Path):  # noqa: ANN001
        _ = (argv, out_dir)
        return {
            "name": name,
            "argv": ["python", name],
            "exit_code": 0,
            "duration_sec": 0.01,
            "stdout_file": str((tmp / f"{name}.stdout.txt").resolve()),
            "stderr_file": str((tmp / f"{name}.stderr.txt").resolve()),
            "parse_error": None,
            "report": _suite_payload(name),
        }

    monkeypatch.setattr(bench, "_run_suite_command", _fake_run)
    rc = bench.main(
        [
            "--dataset-profile",
            "core_oss_v1",
            "--mode",
            "pi0",
            "--artifacts-root",
            str(artifacts_root),
            "--allow-skip-baseline-d",
        ]
    )
    assert rc == bench.EXIT_OK
    report = json.loads(_latest_report(artifacts_root).read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["normalized"]["headline_compare"]["available"] is True
    scorecard = Path(report["artifacts"]["scorecard_csv"])
    assert scorecard.exists()


def test_main_strict_fails_on_gate(monkeypatch):
    tmp = _workspace_tmp("main_strict_fail")
    artifacts_root = tmp / "artifacts"

    def _fake_run(name: str, argv, out_dir: Path):  # noqa: ANN001
        _ = (argv, out_dir)
        payload = _suite_payload(name)
        if name == "run_attack_layer_cycle":
            payload["gate"] = {"passed": False, "checks": [{"id": "ALV1-X", "status": "FAIL"}]}
        return {
            "name": name,
            "argv": ["python", name],
            "exit_code": 0,
            "duration_sec": 0.01,
            "stdout_file": str((tmp / f"{name}.stdout.txt").resolve()),
            "stderr_file": str((tmp / f"{name}.stderr.txt").resolve()),
            "parse_error": None,
            "report": payload,
        }

    monkeypatch.setattr(bench, "_run_suite_command", _fake_run)
    rc = bench.main(
        [
            "--dataset-profile",
            "core_oss_v1",
            "--mode",
            "pi0",
            "--strict",
            "--artifacts-root",
            str(artifacts_root),
            "--allow-skip-baseline-d",
        ]
    )
    assert rc == bench.EXIT_FAILED
    report = json.loads(_latest_report(artifacts_root).read_text(encoding="utf-8"))
    assert report["status"] == "failed_suite"
    assert report["failed_suite"] == "strict_gate"


def test_main_hybrid_api_skips_baseline_d_without_key(monkeypatch):
    tmp = _workspace_tmp("main_skip_d")
    artifacts_root = tmp / "artifacts"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _fake_run(name: str, argv, out_dir: Path):  # noqa: ANN001
        _ = out_dir
        if name == "eval_support_stateful_vs_stateless":
            payload = _suite_payload(name)
            overall = payload["metrics"]["overall"]
            overall.pop("baseline_d_bare_llm_detector", None)
            payload["market_ready"]["variants"].pop("baseline_d_bare_llm_detector", None)
        else:
            payload = _suite_payload(name)
        return {
            "name": name,
            "argv": list(argv),
            "exit_code": 0,
            "duration_sec": 0.01,
            "stdout_file": str((tmp / f"{name}.stdout.txt").resolve()),
            "stderr_file": str((tmp / f"{name}.stderr.txt").resolve()),
            "parse_error": None,
            "report": payload,
        }

    monkeypatch.setattr(bench, "_run_suite_command", _fake_run)
    rc = bench.main(
        [
            "--dataset-profile",
            "core_oss_v1",
            "--mode",
            "hybrid_api",
            "--artifacts-root",
            str(artifacts_root),
            "--allow-skip-baseline-d",
        ]
    )
    assert rc == bench.EXIT_OK
    report = json.loads(_latest_report(artifacts_root).read_text(encoding="utf-8"))
    assert report["status"] == "partial_ok"
    assert "baseline_d_skipped_due_to_missing_api_key" in report["notes"]
