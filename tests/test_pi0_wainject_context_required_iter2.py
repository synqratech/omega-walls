from __future__ import annotations

from omega.interfaces.contracts_v1 import WALLS_V1
from tests.helpers import load_jsonl, mk_item


def _idx() -> dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def test_pi0_wainject_context_required_iter2_casepack(projector):
    rows = load_jsonl("tests/data/wainject_context_required_iter2_cases.jsonl")
    blocking = [r for r in rows if str(r.get("expected_mode", "block")) == "block"]
    observe = [r for r in rows if str(r.get("expected_mode", "")) == "observe"]
    positives = [r for r in blocking if not bool(r.get("expected_all_zero", False))]
    negatives = [r for r in blocking if bool(r.get("expected_all_zero", False))]

    assert len(positives) >= 12
    assert len(negatives) >= 8
    assert len(observe) >= 4

    wall_idx = _idx()
    for row in positives:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        for wall in row.get("expected_nonzero", []):
            assert proj.v[wall_idx[wall]] > 0.0, row["id"]

    for row in negatives:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        assert float(proj.v.sum()) == 0.0, row["id"]


def test_pi0_wainject_context_required_iter2_observe_smoke(projector):
    rows = load_jsonl("tests/data/wainject_context_required_iter2_cases.jsonl")
    observe = [r for r in rows if str(r.get("expected_mode", "")) == "observe"]
    detected = 0
    for row in observe:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        if float(proj.v.sum()) > 0.0:
            detected += 1
    total = len(observe)
    rate = (detected / total) if total else 0.0
    assert total >= 4
    assert 0.0 <= rate <= 1.0

