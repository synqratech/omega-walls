from __future__ import annotations

from omega.interfaces.contracts_v1 import WALLS_V1
from tests.helpers import load_jsonl, mk_item


def _idx() -> dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def test_pi0_promptshield_iter1_blocking_casepacks(projector):
    blocking_fp = load_jsonl("tests/data/promptshield/promptshield_iter1_blocking_fp.jsonl")
    blocking_fn = load_jsonl("tests/data/promptshield/promptshield_iter1_blocking_fn.jsonl")
    assert len(blocking_fp) >= 16
    assert len(blocking_fn) >= 16

    wall_idx = _idx()
    fn_detected = 0
    fn_missed_ids = []
    for row in blocking_fn:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        row_ok = True
        for wall in row.get("expected_nonzero", []):
            if not (proj.v[wall_idx[wall]] > 0.0):
                row_ok = False
                break
        if row_ok:
            fn_detected += 1
        else:
            fn_missed_ids.append(str(row["id"]))

    fn_detect_rate = (fn_detected / len(blocking_fn)) if blocking_fn else 0.0
    assert fn_detected >= 1, f"detect_rate={fn_detect_rate:.3f}; missed={fn_missed_ids[:10]}"

    fp_zero = 0
    fp_missed_ids = []
    for row in blocking_fp:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        if float(proj.v.sum()) == 0.0:
            fp_zero += 1
        else:
            fp_missed_ids.append(str(row["id"]))

    fp_zero_rate = (fp_zero / len(blocking_fp)) if blocking_fp else 0.0
    assert fp_zero_rate >= 0.50, f"zero_rate={fp_zero_rate:.3f}; missed={fp_missed_ids[:10]}"


def test_pi0_promptshield_iter1_observe_smoke(projector):
    rows = load_jsonl("tests/data/promptshield/promptshield_iter1_observe.jsonl")
    assert len(rows) >= 16
    detected = 0
    for row in rows:
        proj = projector.project(mk_item(str(row["id"]), str(row["text"])))
        if float(proj.v.sum()) > 0.0:
            detected += 1
    rate = (detected / len(rows)) if rows else 0.0
    assert 0.0 <= rate <= 1.0
