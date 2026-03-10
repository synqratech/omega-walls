from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from scripts.eval_session_pi_gate import (
    SessionSpec,
    SessionTurnRow,
    _baseline_compare,
    evaluate_pack_with_runner,
    evaluate_sessions,
    summarize_outcomes,
)


@dataclass
class _Runner:
    off_map: Dict[str, int]
    reset_calls: List[tuple[str, str]]
    run_calls: List[tuple[str, int]]

    def __init__(self, off_map: Dict[str, int] | None = None) -> None:
        self.off_map = dict(off_map or {})
        self.reset_calls = []
        self.run_calls = []

    def reset(self, *, session_id: str, actor_id: str) -> None:
        self.reset_calls.append((session_id, actor_id))

    def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> Dict[str, Any]:
        self.run_calls.append((session_id, int(turn.turn_id)))
        off_turn = int(self.off_map.get(session_id, 0))
        off = off_turn > 0 and int(turn.turn_id) >= off_turn
        return {"off": off, "max_p": 1.0 if off else 0.0, "off_reasons": {"reason_wall": int(off)}}


class _CrossCarryRunner(_Runner):
    def __init__(self) -> None:
        super().__init__({})
        self.actor_hits: Dict[str, int] = {}

    def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> Dict[str, Any]:
        self.run_calls.append((session_id, int(turn.turn_id)))
        if str(turn.label_turn) == "attack":
            self.actor_hits[actor_id] = int(self.actor_hits.get(actor_id, 0)) + 1
        off = int(self.actor_hits.get(actor_id, 0)) >= 2
        return {"off": off, "max_p": 1.0 if off else 0.0, "off_reasons": {"reason_sum": int(off)}}


def _mk_session(
    session_id: str,
    *,
    actor: str,
    bucket: str,
    family: str,
    label_session: str,
    turns: List[str],
    eval_slice: str = "text_intrinsic",
) -> SessionSpec:
    rows: List[SessionTurnRow] = []
    for idx, label_turn in enumerate(turns, start=1):
        rows.append(
            SessionTurnRow(
                session_id=session_id,
                turn_id=idx,
                text=f"{session_id}:{idx}:{label_turn}",
                label_turn=label_turn,
                label_session=label_session,
                family=family,
                source_ref="test",
                actor_id=actor,
                bucket=bucket,
                eval_slice=eval_slice,
            )
        )
    return SessionSpec(
        session_id=session_id,
        actor_id=actor,
        bucket=bucket,
        family=family,
        label_session=label_session,
        eval_slice=eval_slice,
        turns=rows,
    )


def test_evaluate_sessions_no_reset_within_session():
    sessions = [
        _mk_session("s1", actor="a1", bucket="core", family="cocktail", label_session="attack", turns=["attack", "attack", "attack"]),
        _mk_session("s2", actor="a2", bucket="core", family="benign_long_context", label_session="benign", turns=["benign", "benign", "benign"]),
    ]
    runner = _Runner(off_map={"s1": 2})
    outcomes, _ = evaluate_sessions(sessions=sessions, runner=runner)

    assert len(outcomes) == 2
    assert len(runner.reset_calls) == 2
    assert len(runner.run_calls) == 6
    assert outcomes[0].first_off_turn == 2
    assert outcomes[1].first_off_turn is None


def test_summarize_outcomes_time_to_off_late_and_never_detected():
    sessions = [
        _mk_session("a1", actor="x1", bucket="core", family="cocktail", label_session="attack", turns=["attack", "attack", "attack", "attack"]),
        _mk_session("a2", actor="x2", bucket="core", family="distributed_wo_explicit", label_session="attack", turns=["attack", "attack", "attack", "attack"]),
        _mk_session("a3", actor="x3", bucket="core", family="roleplay_escalation", label_session="attack", turns=["attack", "attack", "attack", "attack"]),
        _mk_session("b1", actor="y1", bucket="core", family="benign_long_context", label_session="benign", turns=["benign", "benign", "benign"]),
    ]
    runner = _Runner(off_map={"a1": 2, "a2": 4})
    outcomes, _ = evaluate_sessions(sessions=sessions, runner=runner)
    summary = summarize_outcomes(outcomes)

    assert summary["session_attack_off_rate"] == 2 / 3
    assert summary["session_benign_off_rate"] == 0.0
    assert summary["time_to_off"]["median"] == 3.0
    assert summary["first_off_turn_histogram"]["2"] == 1
    assert summary["first_off_turn_histogram"]["4"] == 1
    assert summary["first_off_turn_histogram"]["never"] == 1
    assert summary["late_detect_rate"] == 1 / 3
    assert summary["never_detected_rate_by_family"]["roleplay_escalation"]["never_detected_rate"] == 1.0


def test_evaluate_pack_with_cross_session_slice_separate():
    rows: List[SessionTurnRow] = []
    rows.extend(_mk_session("core-a", actor="core-1", bucket="core", family="distributed_wo_explicit", label_session="attack", turns=["attack", "attack"]).turns)
    rows.extend(_mk_session("core-b", actor="core-2", bucket="core", family="benign_long_context", label_session="benign", turns=["benign", "benign"]).turns)
    rows.extend(_mk_session("xs-a", actor="actor-z", bucket="cross_session", family="distributed_wo_explicit", label_session="attack", turns=["attack", "attack"]).turns)
    rows.extend(_mk_session("xs-b", actor="actor-z", bucket="cross_session", family="distributed_wo_explicit", label_session="attack", turns=["attack", "attack"]).turns)

    core_runner = _Runner(off_map={})
    cross_runner = _CrossCarryRunner()
    out = evaluate_pack_with_runner(rows=rows, core_runner=core_runner, cross_runner=cross_runner)

    assert out["core"]["summary_all"]["session_attack_off_rate"] == 0.0
    assert out["core"]["summary_text_intrinsic"]["session_attack_off_rate"] == 0.0
    assert out["cross_session"]["summary"]["session_attack_off_rate"] > 0.0
    assert len(core_runner.reset_calls) == 2
    assert len(cross_runner.reset_calls) == 2


def test_evaluate_pack_split_context_required_slice():
    rows: List[SessionTurnRow] = []
    rows.extend(
        _mk_session(
            "core-ti",
            actor="a1",
            bucket="core",
            family="distributed_wo_explicit",
            label_session="attack",
            turns=["attack", "attack"],
            eval_slice="text_intrinsic",
        ).turns
    )
    rows.extend(
        _mk_session(
            "core-cr",
            actor="a2",
            bucket="core",
            family="distributed_wo_explicit",
            label_session="attack",
            turns=["attack", "attack"],
            eval_slice="context_required",
        ).turns
    )

    core_runner = _Runner(off_map={"core-ti": 1})
    cross_runner = _Runner(off_map={})
    out = evaluate_pack_with_runner(rows=rows, core_runner=core_runner, cross_runner=cross_runner)

    assert out["core"]["summary_all"]["attack_sessions"] == 2
    assert out["core"]["summary_text_intrinsic"]["attack_sessions"] == 1
    assert out["core"]["summary_context_required"]["attack_sessions"] == 1
    assert out["core"]["summary_text_intrinsic"]["session_attack_off_rate"] == 1.0
    assert out["core"]["summary_context_required"]["session_attack_off_rate"] == 0.0


def test_baseline_compare_uses_text_intrinsic_key():
    cur = {
        "summary_core_text_intrinsic": {"session_attack_off_rate": 0.6, "session_benign_off_rate": 0.01},
        "cross_session": {"session_attack_off_rate": 0.2, "session_benign_off_rate": 0.0},
    }
    base = {
        "summary_core_text_intrinsic": {"session_attack_off_rate": 0.4, "session_benign_off_rate": 0.01},
        "cross_session": {"session_attack_off_rate": 0.1, "session_benign_off_rate": 0.0},
    }
    delta = _baseline_compare(cur, base)
    assert "summary_core_text_intrinsic_delta" in delta
    assert abs(delta["summary_core_text_intrinsic_delta"]["session_attack_off_rate"] - 0.2) < 1e-9
