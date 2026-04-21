from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


def _extract_json_blob(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end >= start
    return json.loads(text[start : end + 1])


def _mk_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_smoke_langchain_strict():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_community")

    root = Path(__file__).resolve().parent.parent
    output = _mk_tmp("framework-smoke") / "langchain_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_langchain.py",
            "--strict",
            "--output",
            str(output.as_posix()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _extract_json_blob(proc.stdout)
    assert payload["framework"] == "langchain"
    assert not payload["failures"]
    assert float(payload["summary"]["gateway_coverage"]) >= 1.0
    assert int(payload["summary"]["orphan_executions"]) == 0


def test_smoke_llamaindex_strict():
    pytest.importorskip("llama_index.core")

    root = Path(__file__).resolve().parent.parent
    output = _mk_tmp("framework-smoke") / "llamaindex_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_llamaindex.py",
            "--strict",
            "--output",
            str(output.as_posix()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _extract_json_blob(proc.stdout)
    assert payload["framework"] == "llamaindex"
    assert not payload["failures"]
    assert float(payload["summary"]["gateway_coverage"]) >= 1.0
    assert int(payload["summary"]["orphan_executions"]) == 0


def test_run_framework_smokes_strict():
    root = Path(__file__).resolve().parent.parent
    output_dir = _mk_tmp("framework-smokes-bundle")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_framework_smokes.py",
            "--strict",
            "--output-dir",
            str(output_dir.as_posix()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _extract_json_blob(proc.stdout)
    assert payload["status"] == "ok"
    assert int(payload["framework_count"]) == 6
    assert set(payload["frameworks"].keys()) == {
        "langchain_guard",
        "langgraph_guard",
        "llamaindex_guard",
        "haystack_guard",
        "autogen_guard",
        "crewai_guard",
    }
    assert float(payload["metrics"]["min_gateway_coverage"]) >= 1.0
    assert int(payload["metrics"]["total_orphans"]) == 0
