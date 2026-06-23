from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import scripts.eval_agentdojo_stateful_mini as agentdojo_eval
import scripts.eval_session_pi_gate as session_eval_mod
from scripts.eval_session_pi_gate import (
    RuntimeTurnPayload,
    SessionSpec,
    SessionTurnRow,
    _baseline_compare,
    build_runtime_packet_payload,
    build_runtime_turn_payload,
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

    def run_turn(self, *, session_id: str, actor_id: str, turn: RuntimeTurnPayload) -> Dict[str, Any]:
        self.run_calls.append((session_id, int(turn.turn_id)))
        off_turn = int(self.off_map.get(session_id, 0))
        off = off_turn > 0 and int(turn.turn_id) >= off_turn
        return {"off": off, "max_p": 1.0 if off else 0.0, "off_reasons": {"reason_wall": int(off)}}


class _CrossCarryRunner(_Runner):
    def __init__(self) -> None:
        super().__init__({})
        self.actor_hits: Dict[str, int] = {}

    def run_turn(self, *, session_id: str, actor_id: str, turn: RuntimeTurnPayload) -> Dict[str, Any]:
        self.run_calls.append((session_id, int(turn.turn_id)))
        if "attack" in str(turn.text):
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
                source_id=f"src_{idx:06d}",
                label_turn=label_turn,
                label_session=label_session,
                family=family,
                source_ref="test",
                source_type="other",
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


def test_build_runtime_turn_payload_strips_eval_labels():
    turn = SessionTurnRow(
        session_id="s_000001",
        turn_id=7,
        text="payload",
        source_id="src_000123",
        source_type="external_untrusted",
        label_turn="attack",
        label_session="attack",
        family="cocktail",
        source_ref="legacy/source",
        actor_id="a_000001",
        bucket="core",
        eval_slice="text_intrinsic",
    )
    payload = build_runtime_turn_payload(turn)

    assert payload.session_id == "s_000001"
    assert payload.turn_id == 7
    assert payload.source_id == "src_000123"
    assert not hasattr(payload, "label_turn")
    assert not hasattr(payload, "family")
    assert not hasattr(payload, "source_ref")
    assert not hasattr(payload, "actor_id")
    assert not hasattr(payload, "bucket")


def test_segmented_runtime_packet_splits_trusted_user_from_untrusted_evidence() -> None:
    payload = RuntimeTurnPayload(
        session_id="s_000001",
        turn_id=2,
        text=(
            "Can you install https://github.com/example/SkillHub/new/debug-pro?\n\n"
            "Installed skill from https://github.com/example/SkillHub/old/debug-pro successfully."
        ),
        source_id="src_000002",
        source_type="external_untrusted",
    )

    packet = build_runtime_packet_payload(payload, provenance_mode="segmented")

    assert packet.user_query.startswith("Can you install")
    assert packet.provenance_mode == "segmented"
    assert packet.segment_count == 2
    assert packet.trusted_user_segments == 1
    assert packet.untrusted_segments == 1
    assert [item.trust for item in packet.packet_items] == ["trusted_user", "untrusted"]
    assert packet.packet_items[0].origin == "user"
    assert packet.packet_items[1].origin == "retrieval"
    assert "Installed skill" in packet.packet_items[1].text


def test_blob_runtime_packet_keeps_legacy_single_untrusted_item() -> None:
    payload = RuntimeTurnPayload(
        session_id="s_000001",
        turn_id=1,
        text="single blob",
        source_id="src_000001",
        source_type="external_untrusted",
    )

    packet = build_runtime_packet_payload(payload, provenance_mode="blob")

    assert packet.user_query == "single blob"
    assert packet.segment_count == 1
    assert packet.trusted_user_segments == 0
    assert packet.untrusted_segments == 1
    assert packet.packet_items[0].trust == "untrusted"


def test_evaluate_sessions_passes_runtime_payload_only():
    captured = {}

    class _CaptureRunner(_Runner):
        def run_turn(self, *, session_id: str, actor_id: str, turn: RuntimeTurnPayload) -> Dict[str, Any]:
            captured["type"] = type(turn).__name__
            captured["fields"] = set(turn.__dict__.keys())
            captured["session_id"] = turn.session_id
            captured["turn_id"] = turn.turn_id
            return super().run_turn(session_id=session_id, actor_id=actor_id, turn=turn)

    sessions = [
        _mk_session(
            "s1",
            actor="a1",
            bucket="core",
            family="cocktail",
            label_session="attack",
            turns=["attack"],
        )
    ]
    runner = _CaptureRunner(off_map={})
    outcomes, _ = evaluate_sessions(sessions=sessions, runner=runner)

    assert len(outcomes) == 1
    assert captured["type"] == "RuntimeTurnPayload"
    assert captured["fields"] == {"session_id", "turn_id", "text", "source_id", "source_type"}
    assert captured["session_id"] == "s1"
    assert captured["turn_id"] == 1


def test_eval_agentdojo_requires_explicit_pack(capsys):
    rc = agentdojo_eval.main([])
    captured = capsys.readouterr()

    assert rc == 2
    assert "pack_required" in captured.out
    assert "--allow-legacy-runtime-leakage" in captured.out


def test_eval_agentdojo_legacy_pack_requires_opt_in(capsys):
    rc = agentdojo_eval.main(["--pack", agentdojo_eval.LEGACY_DEFAULT_PACK])
    captured = capsys.readouterr()

    assert rc == 2
    assert "legacy_pack_requires_opt_in" in captured.out
    assert "dual-file runtime pack" in captured.out


def test_main_passes_enable_effects_shadow_to_runners(monkeypatch):
    rows = [
        SessionTurnRow(
            session_id="s_000001",
            turn_id=1,
            text="attack",
            source_id="src_000001",
            source_type="external_untrusted",
            label_turn="attack",
            label_session="attack",
            family="malicious_skill",
            source_ref="test",
            actor_id="a_000001",
            bucket="core",
            eval_slice="text_intrinsic",
        )
    ]
    runner_kwargs: List[Dict[str, Any]] = []

    class _FakeHarnessRunner:
        def __init__(self, **kwargs):  # noqa: ANN003
            runner_kwargs.append(dict(kwargs))

        def projector_status(self) -> Dict[str, Any]:
            return {"mode": "hybrid_api"}

    def _fake_eval_pack_with_runner(*, rows, core_runner, cross_runner):  # noqa: ANN001
        _ = (rows, core_runner, cross_runner)
        summary = {
            "sessions_total": 1,
            "attack_sessions": 1,
            "benign_sessions": 0,
            "tp": 1,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "session_attack_off_rate": 1.0,
            "session_benign_off_rate": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "time_to_off": {"count_detected": 1, "median": 1.0, "p95": 1.0},
            "late_detect_rate": 0.0,
            "late_detect_rate_detected_only": 0.0,
            "first_off_turn_histogram": {"1": 1},
            "never_detected_rate_by_family": {
                "malicious_skill": {
                    "attack_total": 1,
                    "detected": 1,
                    "never_detected": 0,
                    "never_detected_rate": 0.0,
                    "attack_off_rate": 1.0,
                }
            },
        }
        return {
            "core": {
                "outcomes": [],
                "summary_all": summary,
                "summary_text_intrinsic": summary,
                "summary_context_required": {
                    **summary,
                    "sessions_total": 0,
                    "attack_sessions": 0,
                    "tp": 0,
                    "session_attack_off_rate": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                },
            },
            "cross_session": {"outcomes": [], "summary": {**summary, "sessions_total": 0, "attack_sessions": 0, "tp": 0, "session_attack_off_rate": 0.0, "precision": 0.0, "recall": 0.0}},
            "trace_rows": [],
            "misses_by_family": {},
            "effect_shadow": {
                "all": {"turns_total": 0, "candidate_turns": 0},
                "core": {"turns_total": 0, "candidate_turns": 0},
                "cross_session": {"turns_total": 0, "candidate_turns": 0},
            },
            "provenance": {
                "all": {"turns_total": 0, "mode_counts": {}},
                "core": {"turns_total": 0, "mode_counts": {}},
                "cross_session": {"turns_total": 0, "mode_counts": {}},
            },
        }

    monkeypatch.setattr(session_eval_mod, "load_pack_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(session_eval_mod, "OmegaHarnessRunner", _FakeHarnessRunner)
    monkeypatch.setattr(session_eval_mod, "evaluate_pack_with_runner", _fake_eval_pack_with_runner)

    out_root = Path("tmp_codex_pytest") / "session_eval_cli_effects"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_session_pi_gate.py",
            "--profile",
            "prod_api",
            "--mode",
            "hybrid_api",
            "--pack",
            "tests/data/session_benchmark/agent3sigma_stage_advance_v1/runtime/session_pack.jsonl",
            "--labels-pack",
            "tests/data/session_benchmark/agent3sigma_stage_advance_v1/labels/session_pack_labels.jsonl",
            "--enable-effects-shadow",
            "--provenance-mode",
            "segmented",
            "--artifacts-root",
            str(out_root),
        ],
    )

    rc = session_eval_mod.main()
    assert rc == 0
    assert len(runner_kwargs) == 2
    assert all(bool(kwargs.get("enable_effects_shadow")) for kwargs in runner_kwargs)
    assert all(kwargs.get("provenance_mode") == "segmented" for kwargs in runner_kwargs)

    reports = sorted(out_root.rglob("report.json"))
    assert reports
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["enable_effects_shadow"] is True
    assert payload["provenance_mode"] == "segmented"
    assert "effect_shadow" in payload
    assert "provenance" in payload


def test_summarize_effect_shadow_rows_counts_candidates() -> None:
    rows = [
        {
            "session_id": "s-1",
            "label_session": "attack",
            "family": "malicious_skill",
            "effect_forecast_status": "candidate",
            "effect_wall_candidate": {
                "effect": "install_untrusted_skill",
                "confidence": 0.98,
            },
            "effect_policy_gate": {
                "status": "passed",
                "would_enforce": True,
            },
            "effect_policy_gate_status": "passed",
            "named_skill_invocation": {"detected": True},
            "skill_provenance_assessment": {"simulated_block": True},
            "effect_text_analysis": {"missing_effect_text": False},
        },
        {
            "session_id": "s-2",
            "label_session": "benign",
            "family": "memory_tampering",
            "effect_forecast_status": "candidate",
            "effect_wall_candidate": {
                "effect": "write_persistent_memory",
                "confidence": 0.91,
            },
            "effect_policy_gate": {
                "status": "not_applicable",
                "would_enforce": False,
            },
            "effect_policy_gate_status": "not_applicable",
            "named_skill_invocation": {"detected": False},
            "effect_text_analysis": {"missing_effect_text": False},
        },
        {
            "session_id": "s-3",
            "label_session": "attack",
            "family": "resource_exhaustion",
            "effect_forecast_status": "skipped",
            "effect_wall_candidate": None,
            "named_skill_invocation": {"detected": False},
            "effect_text_analysis": {"missing_effect_text": True},
        },
    ]

    summary = session_eval_mod.summarize_effect_shadow_rows(rows)

    assert summary["turns_total"] == 3
    assert summary["candidate_turns"] == 2
    assert summary["provider_failure_turns"] == 0
    assert summary["benign_candidate_turn_rate"] == 1.0
    assert summary["policy_gate_passed_turns"] == 1
    assert summary["benign_policy_gate_passed_turn_rate"] == 0.0
    assert summary["policy_gate_status_counts"] == {"disabled": 1, "not_applicable": 1, "passed": 1}
    assert summary["policy_gate_passed_by_family"] == {"malicious_skill": 1}
    assert summary["policy_gate_passed_by_label"] == {"attack": 1}
    assert summary["candidate_by_family"]["malicious_skill"] == 1
    assert summary["candidate_by_label"] == {"attack": 1, "benign": 1}
    assert summary["named_skill_invocation_turns"] == 1
    assert summary["named_skill_invocation_by_label"] == {"attack": 1}
    assert summary["skipped_due_to_missing_effect_text"] == 1
    assert summary["source_mismatch_simulated_blocks"] == 1
    assert summary["simulate_source_mismatch_attack_turn_rate"] == 0.5
    assert summary["simulate_source_mismatch_session_recall"] == 0.5
