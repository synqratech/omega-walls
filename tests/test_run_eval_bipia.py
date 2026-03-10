from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _extract_json_blob(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end >= start
    return json.loads(text[start : end + 1])


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_run_eval_bipia_enforced_pass_path():
    root = Path(__file__).resolve().parent.parent
    out_file = _mk_local_tmp("run-eval-bipia") / "bipia_report.json"

    env = dict(os.environ)
    env["BIPIA_COMMIT"] = "deadbeef"
    env["OMEGA__BIPIA__THRESHOLDS__SAMPLED__TASKS_EVALUATED_EQ"] = "5"
    env["OMEGA__BIPIA__THRESHOLDS__SAMPLED__ATTACK_OFF_RATE_GE"] = "0.0"
    env["OMEGA__BIPIA__THRESHOLDS__SAMPLED__BENIGN_OFF_RATE_EQ"] = "0.0"
    env["OMEGA__BIPIA__THRESHOLDS__SAMPLED__PER_TASK_ATTACK_OFF_RATE_GE"] = "0.0"
    env["OMEGA__BIPIA__THRESHOLDS__SAMPLED__COVERAGE_WALL_ANY_GE"] = "0.0"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--enforce-bipia",
            "--bipia-benchmark-root",
            "tests/data/bipia_fixture/benchmark",
            "--bipia-mode",
            "sampled",
            "--bipia-max-contexts-per-task",
            "1",
            "--bipia-seed",
            "41",
            "--whitebox-max-samples",
            "5",
            "--bipia-json-output",
            str(out_file.as_posix()),
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0
    payload = _extract_json_blob(proc.stdout)
    assert payload["bipia"]["status"] == "ok"
    assert payload["bipia"]["manifest_path"] is not None
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"


def test_run_eval_bipia_enforced_fails_without_commit_env():
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.pop("BIPIA_COMMIT", None)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--enforce-bipia",
            "--bipia-benchmark-root",
            "tests/data/bipia_fixture/benchmark",
            "--bipia-mode",
            "sampled",
            "--bipia-max-contexts-per-task",
            "1",
            "--whitebox-max-samples",
            "5",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode != 0
    assert "bipia unavailable" in (proc.stdout + proc.stderr)
