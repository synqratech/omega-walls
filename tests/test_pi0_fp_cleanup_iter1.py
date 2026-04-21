from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, OmegaState
from tests.helpers import load_jsonl, mk_item


def _load_session_pack_rows() -> dict[str, list[dict]]:
    rows_by_sid: dict[str, list[dict]] = defaultdict(list)
    path = Path("tests/data/session_benchmark/session_pack_seed41_v1.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows_by_sid[str(row["session_id"])].append(row)
    return rows_by_sid


def test_fp_cleanup_iter1_fixed_pack_all_zero(projector):
    rows = load_jsonl("tests/data/fp_cleanup_iter1_cases.jsonl")
    assert len(rows) == 47
    for row in rows:
        assert bool(row.get("expected_all_zero", False)) is True
        text = str(row["text"])
        out = projector.project(mk_item(str(row["id"]), text))
        assert float(out.v.sum()) == 0.0, str(row["id"])


def test_fp_cleanup_iter1_session_runtime_no_off_for_4_benign_sessions(projector, omega_core, off_policy):
    rows_by_sid = _load_session_pack_rows()
    target_sids = [
        "sess-core-ben-017",
        "sess-core-ben-072",
        "sess-core-ben-083",
        "sess-core-ben-087",
    ]

    for sid in target_sids:
        turns = sorted(rows_by_sid[sid], key=lambda r: int(r["turn_id"]))
        state = OmegaState(session_id=sid, m=np.zeros(4, dtype=float), step=0)
        for row in turns:
            turn_id = int(row["turn_id"])
            text = str(row["text"])
            item = ContentItem(
                doc_id=f"{sid}:turn:{turn_id:03d}",
                source_id=f"session_benchmark:{row['bucket']}:{row['family']}",
                source_type="other",
                trust="semi",
                text=text,
            )
            proj = projector.project(item)
            step = omega_core.step(state, [item], [proj])
            decision = off_policy.select_actions(step, [item])
            assert decision.off is False, f"{sid}:{turn_id}"
