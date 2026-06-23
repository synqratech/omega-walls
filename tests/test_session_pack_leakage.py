from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import uuid

import pytest

from scripts.eval_session_pi_gate import OmegaHarnessRunner, SessionSpec, SessionTurnRow
import scripts.eval_support_stateful_vs_stateless as support_eval
from scripts.session_pack_leakage import load_dual_session_pack


def _tmp_dir(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "session_pack_leakage"
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{name}_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(body, encoding="utf-8")


def test_leakage_audit_rejects_forbidden_runtime_keys() -> None:
    tmp = _tmp_dir("forbidden_keys")
    try:
        runtime = tmp / "runtime" / "session_pack.jsonl"
        labels = tmp / "labels" / "session_pack_labels.jsonl"
        _write_jsonl(
            runtime,
            [
                {
                    "session_id": "s_000001",
                    "turn_id": 1,
                    "text": "hello",
                    "source_type": "external_untrusted",
                    "source_id": "src_000001",
                    "label_session": "attack",
                }
            ],
        )
        _write_jsonl(
            labels,
            [
                {
                    "session_id": "s_000001",
                    "turn_id": 1,
                    "label_turn": "benign",
                    "label_session": "attack",
                    "family": "x",
                    "bucket": "core",
                    "eval_slice": "text_intrinsic",
                    "source_ref": "r1",
                    "actor_id": "a_000001",
                }
            ],
        )
        with pytest.raises(ValueError, match="leakage audit failed"):
            load_dual_session_pack(runtime_pack_path=runtime)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_leakage_audit_rejects_non_opaque_ids() -> None:
    tmp = _tmp_dir("non_opaque_ids")
    try:
        runtime = tmp / "runtime" / "session_pack.jsonl"
        labels = tmp / "labels" / "session_pack_labels.jsonl"
        _write_jsonl(
            runtime,
            [
                {
                    "session_id": "sess-core-atk-001",
                    "turn_id": 1,
                    "text": "hello",
                    "source_type": "external_untrusted",
                    "source_id": "session_benchmark:core:cocktail",
                }
            ],
        )
        _write_jsonl(
            labels,
            [
                {
                    "session_id": "sess-core-atk-001",
                    "turn_id": 1,
                    "label_turn": "attack",
                    "label_session": "attack",
                    "family": "cocktail",
                    "bucket": "core",
                    "eval_slice": "text_intrinsic",
                    "source_ref": "r1",
                    "actor_id": "actor-atk-1",
                }
            ],
        )
        with pytest.raises(ValueError, match="leakage audit failed"):
            load_dual_session_pack(runtime_pack_path=runtime)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dual_pack_clean_contract_passes() -> None:
    tmp = _tmp_dir("clean_dual")
    try:
        runtime = tmp / "runtime" / "session_pack.jsonl"
        labels = tmp / "labels" / "session_pack_labels.jsonl"
        _write_jsonl(
            runtime,
            [
                {
                    "session_id": "s_000001",
                    "turn_id": 1,
                    "text": "hello",
                    "source_type": "external_untrusted",
                    "source_id": "src_000001",
                }
            ],
        )
        _write_jsonl(
            labels,
            [
                {
                    "session_id": "s_000001",
                    "turn_id": 1,
                    "label_turn": "benign",
                    "label_session": "attack",
                    "family": "xsrc",
                    "bucket": "core",
                    "eval_slice": "text_intrinsic",
                    "source_ref": "r1",
                    "actor_id": "a_000001",
                }
            ],
        )
        rows = load_dual_session_pack(runtime_pack_path=runtime)
        assert len(rows) == 1
        assert rows[0]["source_id"] == "src_000001"
        assert rows[0]["family"] == "xsrc"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stateful_runtime_payload_uses_runtime_source_id_only() -> None:
    class _Reasons:
        reason_spike = False
        reason_wall = False
        reason_sum = False
        reason_multi = False

    class _Step:
        off = False
        p = [0.0]
        reasons = _Reasons()

    class _Harness:
        def __init__(self) -> None:
            self.last = {}

        def run_step(self, *, user_query, packet_items, actor_id):
            self.last = {"user_query": user_query, "packet_items": packet_items, "actor_id": actor_id}
            return {"step_result": _Step()}

    runner = object.__new__(OmegaHarnessRunner)
    runner._harness = _Harness()  # type: ignore[attr-defined]
    runner._blind_eval = False  # type: ignore[attr-defined]

    turn = SessionTurnRow(
        session_id="s_000001",
        turn_id=1,
        text="payload",
        source_id="src_000777",
        source_type="external_untrusted",
        label_turn="attack",
        label_session="attack",
        family="cocktail",
        source_ref="legacy/source",
        actor_id="a_000001",
        bucket="core",
        eval_slice="text_intrinsic",
    )
    _ = runner.run_turn(session_id="s_000001", actor_id="a_000001", turn=turn)
    pkt = runner._harness.last["packet_items"][0]  # type: ignore[attr-defined]
    assert pkt.source_id == "src_000777"
    assert "cocktail" not in pkt.source_id
    assert "attack" not in pkt.source_id


def test_stateless_and_prefix_use_opaque_ephemeral_ids() -> None:
    class _CaptureRunner:
        def __init__(self) -> None:
            self.reset_ids = []

        def reset(self, *, session_id: str, actor_id: str) -> None:
            self.reset_ids.append((session_id, actor_id))

        def run_turn(self, *, session_id, actor_id, turn):
            _ = (session_id, actor_id, turn)
            return support_eval.TurnSignal(
                turn_id=1,
                off=False,
                max_p=0.0,
                max_m_next=0.0,
                sum_m_next=0.0,
                count_m_ge_theta=0,
                walls=[],
                action_types=[],
                control_outcome="ALLOW",
            )

    session = SessionSpec(
        session_id="sess-core-atk-001",
        actor_id="actor-benign-001",
        bucket="core",
        family="cocktail",
        label_session="attack",
        eval_slice="text_intrinsic",
        turns=[
            SessionTurnRow(
                session_id="sess-core-atk-001",
                turn_id=1,
                text="a",
                source_id="src_000001",
                source_type="external_untrusted",
                label_turn="benign",
                label_session="attack",
                family="cocktail",
                source_ref="r1",
                actor_id="actor-benign-001",
                bucket="core",
                eval_slice="text_intrinsic",
            ),
            SessionTurnRow(
                session_id="sess-core-atk-001",
                turn_id=2,
                text="b",
                source_id="src_000002",
                source_type="external_untrusted",
                label_turn="attack",
                label_session="attack",
                family="cocktail",
                source_ref="r2",
                actor_id="actor-benign-001",
                bucket="core",
                eval_slice="text_intrinsic",
            ),
        ],
    )

    runner_a = _CaptureRunner()
    support_eval.evaluate_baseline_a_sessions(pack_id="p", sessions=[session], runner=runner_a)  # type: ignore[arg-type]
    runner_c = _CaptureRunner()
    support_eval.evaluate_baseline_c_prefix_scan(pack_id="p", sessions=[session], runner=runner_c)  # type: ignore[arg-type]

    for sid, aid in runner_a.reset_ids + runner_c.reset_ids:
        assert re.match(r"^s_\d{6}$", sid)
        assert re.match(r"^a_\d{6}$", aid)
        assert "atk" not in sid and "ben" not in sid
        assert "atk" not in aid and "ben" not in aid


def test_bare_detector_payload_has_no_label_or_family_hints() -> None:
    payload = support_eval._build_bare_detector_turn_payload(  # noqa: SLF001
        turn_text="some text",
        source_type="external_untrusted",
    )
    assert "session_id" not in payload
    assert "actor_id" not in payload
    assert "family" not in payload
    assert "label_session" not in payload
    assert "bucket" not in payload
