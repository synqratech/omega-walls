from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


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


def test_smoke_langgraph_guard_strict() -> None:
    root = Path(__file__).resolve().parent.parent
    output = _mk_tmp("framework-smoke") / "langgraph_guard_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_langgraph_guard.py",
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
    assert payload["framework"] == "langgraph_guard"
    assert not payload["failures"]
    assert bool(payload["summary"]["blocked_graph_input_seen"])
    assert bool(payload["summary"]["blocked_tool_seen"])
    assert float(payload["summary"]["gateway_coverage"]) >= 1.0
    assert int(payload["summary"]["orphan_executions"]) == 0

