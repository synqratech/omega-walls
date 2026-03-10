from __future__ import annotations

from omega.interfaces.contracts_v1 import WALLS_V1
from tests.helpers import load_jsonl, mk_item


def _idx() -> dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def test_pi0_open_prompt_injection_iter2_casepack(projector):
    rows = load_jsonl("tests/data/open_prompt_injection_iter2_cases.jsonl")
    positives = [r for r in rows if not bool(r.get("expected_all_zero", False))]
    hard_negatives = [r for r in rows if bool(r.get("expected_all_zero", False))]
    assert len(positives) >= 12
    assert len(hard_negatives) >= 8

    wall_idx = _idx()
    for row in positives:
        proj = projector.project(mk_item(row["id"], row["text"]))
        for wall in row["expected_nonzero"]:
            assert proj.v[wall_idx[wall]] > 0.0, row["id"]

    for row in hard_negatives:
        proj = projector.project(mk_item(row["id"], row["text"]))
        assert float(proj.v.sum()) == 0.0, row["id"]
