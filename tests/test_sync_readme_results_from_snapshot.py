from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_readme_results_from_snapshot import build_results_block


ROOT = Path(__file__).resolve().parents[1]


def test_build_results_block_contains_frozen_runs_and_model_scope() -> None:
    snapshot = json.loads((ROOT / "docs/public_results_snapshot.json").read_text(encoding="utf-8"))
    block = build_results_block(snapshot)
    assert "benchmark_20260417T094612Z_a2865dc41147" in block
    assert "support_family_eval_compare_20260408T210609Z" in block
    assert "gpt-5.4-mini" in block
    assert "not claimed" in block.lower()
