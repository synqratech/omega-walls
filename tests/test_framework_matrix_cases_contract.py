from __future__ import annotations

import json
from pathlib import Path


def test_workflow_cases_fixture_has_exactly_ten_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "data" / "framework_matrix" / "workflow_cases_v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    cases = payload.get("cases")
    assert isinstance(cases, list)
    assert len(cases) == 10
    ids = [str(row.get("case_id", "")) for row in cases if isinstance(row, dict)]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_workflow_cases_cover_required_categories() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "data" / "framework_matrix" / "workflow_cases_v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    categories = {str(row.get("category", "")) for row in cases if isinstance(row, dict)}
    assert {"allow_input", "blocked_input", "blocked_tool", "require_approval", "allow_tool", "outage_fail_closed"} <= categories

