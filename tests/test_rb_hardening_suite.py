from __future__ import annotations

import hashlib
import json

from omega.interfaces.contracts_v1 import WALLS_V1
from tests.helpers import mk_item
from tests.rb_hardening_fuzz import build_rb_hardening_cases


def _idx() -> dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def _signature(rows: list[dict]) -> str:
    payload = json.dumps(
        [
            {
                "id": row["id"],
                "text": row["text"],
                "targets": row["targets"],
                "expected_mode": row["expected_mode"],
            }
            for row in rows
        ],
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_rb_hardening_suite_is_deterministic():
    rows_a = build_rb_hardening_cases(seed=41)
    rows_b = build_rb_hardening_cases(seed=41)
    assert _signature(rows_a) == _signature(rows_b)


def test_rb_hardening_suite_case_budget():
    rows = build_rb_hardening_cases(seed=41)
    assert len(rows) == 100
    assert sum(1 for r in rows if r["expected_mode"] == "block") == 80
    assert sum(1 for r in rows if r["expected_mode"] == "observe") == 20


def test_rb_hardening_suite_blocking_contract(projector):
    rows = [r for r in build_rb_hardening_cases(seed=41) if r["expected_mode"] == "block"]
    wall_idx = _idx()
    for row in rows:
        proj = projector.project(mk_item(row["id"], row["text"]))
        if row["group"] == "positive":
            for wall in row["targets"]:
                assert proj.v[wall_idx[wall]] > 0.0, row["id"]
        else:
            assert float(proj.v.sum()) == 0.0, row["id"]


def test_rb_hardening_suite_observe_contract(projector):
    rows = [r for r in build_rb_hardening_cases(seed=41) if r["expected_mode"] == "observe"]
    observe_total = 0
    observe_detected = 0
    for row in rows:
        assert row["group"] in {"positive", "hard_negative", "multilingual_negative"}, row["id"]
        assert isinstance(row["text"], str) and row["text"], row["id"]
        proj = projector.project(mk_item(row["id"], row["text"]))
        observe_total += 1
        if float(proj.v.sum()) > 0.0:
            observe_detected += 1

    assert observe_total == 20
    observe_detect_rate = observe_detected / observe_total if observe_total else 0.0
    assert 0.0 <= observe_detect_rate <= 1.0


def test_rb_hardening_suite_fp_zero_on_blocking_negatives(projector):
    rows = [
        r
        for r in build_rb_hardening_cases(seed=41)
        if r["expected_mode"] == "block" and r["group"] in {"hard_negative", "multilingual_negative"}
    ]
    fp = 0
    for row in rows:
        proj = projector.project(mk_item(row["id"], row["text"]))
        if float(proj.v.sum()) > 0.0:
            fp += 1
    assert fp == 0


def test_rb_hardening_obfuscation_observe_gain(projector):
    rows = [r for r in build_rb_hardening_cases(seed=41) if r["expected_mode"] == "observe"]
    wall_idx = _idx()

    base64_pos_total = 0
    base64_pos_hit = 0
    zero_width_pos_total = 0
    zero_width_pos_hit = 0
    observe_neg_fp = 0
    observe_neg_total = 0

    for row in rows:
        proj = projector.project(mk_item(row["id"], row["text"]))
        rid = str(row["id"])
        is_base64 = "base64_lite_wrap" in rid
        is_zero_width = "zero_width_insert" in rid
        if row["group"] == "positive":
            hit = all(proj.v[wall_idx[w]] > 0.0 for w in row["targets"])
            if is_base64:
                base64_pos_total += 1
                base64_pos_hit += int(hit)
            if is_zero_width:
                zero_width_pos_total += 1
                zero_width_pos_hit += int(hit)
        else:
            observe_neg_total += 1
            if float(proj.v.sum()) > 0.0:
                observe_neg_fp += 1

    assert base64_pos_total == 4
    assert zero_width_pos_total == 4
    assert base64_pos_hit >= 3
    assert zero_width_pos_hit == 4
    assert observe_neg_fp == 0
