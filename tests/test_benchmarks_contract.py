from __future__ import annotations

import json
from pathlib import Path

from scripts.check_benchmarks_contract import validate


ROOT = Path(__file__).resolve().parents[1]


def test_benchmarks_contract_structure_and_snapshots() -> None:
    report = validate(smoke_exec=False, require_source_reports=False)
    assert report["status"] == "ok"


def test_benchmark_index_references_existing_snapshots() -> None:
    index_path = ROOT / "benchmarks" / "results" / "benchmark_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks", [])
    assert isinstance(rows, list) and len(rows) == 4
    for row in rows:
        rel = str(row.get("snapshot_json", "")).strip()
        assert rel
        assert (ROOT / rel).exists(), f"missing snapshot: {rel}"
