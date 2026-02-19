from __future__ import annotations

from omega.interfaces.contracts_v1 import WALLS_V1
from tests.helpers import load_jsonl, mk_item


def test_pi0_hard_negatives_all_zero(projector):
    rows = load_jsonl("tests/data/hard_negatives_50.jsonl")
    assert len(rows) == 50

    for row in rows:
        item = mk_item(row["id"], row["text"])
        proj = projector.project(item)
        assert float(proj.v.sum()) == 0.0, row["id"]
        assert max(proj.evidence.polarity) <= 0, row["id"]


def test_pi0_positives_hit_rates(projector):
    rows = load_jsonl("tests/data/redteam_pos_20.jsonl") + load_jsonl("tests/data/redteam_obf_20.jsonl")
    assert len(rows) == 40

    idx = {w: i for i, w in enumerate(WALLS_V1)}
    targeted = {w: 0 for w in WALLS_V1}
    hits = {w: 0 for w in WALLS_V1}
    all_hits = 0

    for row in rows:
        item = mk_item(row["id"], row["text"])
        proj = projector.project(item)
        local_ok = True
        for wall in row["expected_nonzero"]:
            targeted[wall] += 1
            if proj.v[idx[wall]] > 0:
                hits[wall] += 1
            else:
                local_ok = False
        if local_ok:
            all_hits += 1

    overall = all_hits / len(rows)
    assert overall >= 0.95
    for wall in WALLS_V1:
        rate = hits[wall] / targeted[wall] if targeted[wall] else 1.0
        assert rate >= 0.90, f"{wall} rate={rate}"
