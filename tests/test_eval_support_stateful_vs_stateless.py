from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import pytest

from scripts.eval_session_pi_gate import SessionSpec, SessionTurnRow
import scripts.eval_support_stateful_vs_stateless as support_eval


def _workspace_tmp(name: str) -> Path:
    root = Path("tmp_codex_pytest") / "support_eval_tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _mk_session(
    session_id: str,
    *,
    actor_id: str,
    label_session: str,
    turns: list[str],
    label_turns: list[str] | None = None,
    family: str = "xsrc_family",
    bucket: str = "core",
) -> SessionSpec:
    labels = list(label_turns) if label_turns is not None else [label_session for _ in turns]
    rows: list[SessionTurnRow] = []
    for idx, (text, lbl_turn) in enumerate(zip(turns, labels), start=1):
        rows.append(
            SessionTurnRow(
                session_id=session_id,
                turn_id=idx,
                text=text,
                label_turn=lbl_turn,
                label_session=label_session,
                family=family,
                source_ref=f"src/{session_id}/{idx}",
                source_type="external_untrusted",
                actor_id=actor_id,
                bucket=bucket,
                eval_slice="text_intrinsic",
            )
        )
    return SessionSpec(
        session_id=session_id,
        actor_id=actor_id,
        bucket=bucket,
        family=family,
        label_session=label_session,
        eval_slice="text_intrinsic",
        turns=rows,
    )


class _MemoryRunner:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, str]] = []
        self._counter: dict[tuple[str, str], int] = {}

    def reset(self, *, session_id: str, actor_id: str) -> None:
        self.reset_calls.append((session_id, actor_id))
        self._counter[(session_id, actor_id)] = 0

    def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> support_eval.TurnSignal:
        key = (session_id, actor_id)
        self._counter[key] = self._counter.get(key, 0) + 1
        seen = int(self._counter[key])
        return support_eval.TurnSignal(
            turn_id=int(turn.turn_id),
            off=bool(seen >= 2),
            max_p=float(seen) / 2.0,
            max_m_next=0.0,
            sum_m_next=0.0,
            count_m_ge_theta=0,
            walls=[],
            action_types=[],
            control_outcome="ALLOW",
        )


def test_baseline_a_resets_each_turn_and_has_no_memory():
    runner = _MemoryRunner()
    session = _mk_session("sess-a", actor_id="actor-a", label_session="attack", turns=["t1", "t2", "t3"])
    outcomes, _, _ = support_eval.evaluate_baseline_a_sessions(pack_id="pack", sessions=[session], runner=runner)  # type: ignore[arg-type]

    assert len(runner.reset_calls) == 3
    assert outcomes[0].detected_off is False
    assert outcomes[0].first_off_turn is None


def test_baseline_b_uses_tau_and_running_max():
    session = _mk_session("sess-b", actor_id="actor-b", label_session="attack", turns=["a", "b", "c"])
    signals = [
        support_eval.TurnSignal(turn_id=1, off=False, max_p=0.2, max_m_next=0.0, sum_m_next=0.0, count_m_ge_theta=0, walls=[], action_types=[], control_outcome="ALLOW"),
        support_eval.TurnSignal(turn_id=2, off=False, max_p=0.55, max_m_next=0.0, sum_m_next=0.0, count_m_ge_theta=0, walls=[], action_types=[], control_outcome="ALLOW"),
        support_eval.TurnSignal(turn_id=3, off=False, max_p=0.7, max_m_next=0.0, sum_m_next=0.0, count_m_ge_theta=0, walls=[], action_types=[], control_outcome="ALLOW"),
    ]
    outcomes, rows = support_eval.derive_baseline_b_outcomes(
        pack_id="pack",
        sessions=[session],
        signals_by_session={session.session_id: signals},
        tau_b=0.6,
    )
    assert outcomes[0].detected_off is True
    assert outcomes[0].first_off_turn == 3
    assert abs(outcomes[0].session_risk - 0.7) < 1e-9
    assert rows[-1]["off_by_tau"] is True


def test_calibrate_tau_b_uses_benign_q95():
    outcomes = [
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_BASELINE_A,
            session_id="ben1",
            actor_id="a1",
            family="f1",
            bucket="core",
            label_session="benign",
            turn_count=2,
            detected_off=False,
            first_off_turn=None,
            session_risk=0.1,
            max_turn_p=0.1,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome="ALLOW",
        ),
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_BASELINE_A,
            session_id="ben2",
            actor_id="a2",
            family="f1",
            bucket="core",
            label_session="benign",
            turn_count=2,
            detected_off=False,
            first_off_turn=None,
            session_risk=0.5,
            max_turn_p=0.5,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome="ALLOW",
        ),
    ]
    calibration = support_eval.calibrate_tau_b(outcomes, method="benign_q95")
    assert calibration["method"] == "benign_q95"
    assert calibration["benign_count"] == 2
    assert abs(float(calibration["tau_B"]) - 0.48) < 1e-9


def test_calibrate_tau_d_uses_benign_q95():
    outcomes = [
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_BASELINE_D,
            session_id="ben1",
            actor_id="a1",
            family="f1",
            bucket="core",
            label_session="benign",
            turn_count=2,
            detected_off=False,
            first_off_turn=None,
            session_risk=0.1,
            max_turn_p=0.1,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome="ALLOW",
        ),
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_BASELINE_D,
            session_id="ben2",
            actor_id="a2",
            family="f1",
            bucket="core",
            label_session="benign",
            turn_count=2,
            detected_off=False,
            first_off_turn=None,
            session_risk=0.5,
            max_turn_p=0.5,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome="ALLOW",
        ),
    ]
    calibration = support_eval.calibrate_tau_d(outcomes, method="benign_q95")
    assert calibration["method"] == "benign_q95"
    assert calibration["benign_count"] == 2
    assert abs(float(calibration["tau_D"]) - 0.48) < 1e-9


def test_bare_detector_payload_builder_is_session_agnostic():
    one = support_eval._build_bare_detector_turn_payload(turn_text="same turn", source_type="external_untrusted")  # noqa: SLF001
    two = support_eval._build_bare_detector_turn_payload(turn_text="same turn", source_type="external_untrusted")  # noqa: SLF001
    assert one == two
    assert "session_id" not in one
    assert "actor_id" not in one


def test_validate_bare_llm_detector_payload():
    wall_scores = {str(w): 0.0 for w in support_eval.WALLS_V1}
    wall_scores[support_eval.WALL_EXFIL] = 0.9
    payload = {
        "risk_score": 0.75,
        "wall_scores": wall_scores,
        "wall_flags": ["secret_exfiltration"],
    }
    validated = support_eval._validate_bare_llm_detector_payload(payload)  # noqa: SLF001
    assert abs(validated["risk_score"] - 0.75) < 1e-9
    assert validated["wall_flags"] == ["secret_exfiltration"]


def test_market_ready_mapping_detector_only_variants():
    table = support_eval.build_market_ready_table(
        {
            support_eval.VARIANT_STATEFUL: {"session_attack_off_rate": 0.9, "session_benign_off_rate": 0.1, "steps_to_off": {}, "late_detect_rate": 0.1, "enforcement_correctness": {"ok": True}},
            support_eval.VARIANT_BASELINE_A: {"session_attack_off_rate": 0.8, "session_benign_off_rate": 0.2, "steps_to_off": {}, "late_detect_rate": 0.2, "enforcement_correctness": {"ok": True}},
            support_eval.VARIANT_BASELINE_B: {"session_attack_off_rate": 0.7, "session_benign_off_rate": 0.1, "steps_to_off": {}, "late_detect_rate": 0.3},
            support_eval.VARIANT_BASELINE_C: {"session_attack_off_rate": 0.7, "session_benign_off_rate": 0.1, "steps_to_off": {}, "late_detect_rate": 0.3},
            support_eval.VARIANT_BASELINE_D: {"session_attack_off_rate": 0.6, "session_benign_off_rate": 0.1, "steps_to_off": {}, "late_detect_rate": 0.3},
        }
    )
    assert table["variants"][support_eval.VARIANT_STATEFUL]["enforcement_scope"] == "native_policy_actions"
    assert table["variants"][support_eval.VARIANT_BASELINE_A]["enforcement_scope"] == "native_policy_actions"
    assert table["variants"][support_eval.VARIANT_BASELINE_B]["enforcement_scope"] == "detector_only_not_comparable"
    assert table["variants"][support_eval.VARIANT_BASELINE_C]["enforcement_scope"] == "detector_only_not_comparable"
    assert table["variants"][support_eval.VARIANT_BASELINE_D]["enforcement_scope"] == "detector_only_not_comparable"


def test_baseline_c_prefix_scan_detects_earliest_prefix():
    class _PrefixRunner:
        def reset(self, *, session_id: str, actor_id: str) -> None:
            _ = (session_id, actor_id)

        def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> support_eval.TurnSignal:
            text = str(turn.text)
            off = ("alpha" in text) and ("beta" in text)
            return support_eval.TurnSignal(
                turn_id=int(turn.turn_id),
                off=off,
                max_p=1.0 if off else 0.1,
                max_m_next=0.0,
                sum_m_next=0.0,
                count_m_ge_theta=0,
                walls=[],
                action_types=[],
                control_outcome="SOFT_BLOCK" if off else "ALLOW",
            )

    session = _mk_session(
        "sess-c",
        actor_id="actor-c",
        label_session="attack",
        turns=["alpha", "beta", "gamma"],
        label_turns=["benign", "attack", "attack"],
    )
    outcomes, _ = support_eval.evaluate_baseline_c_prefix_scan(
        pack_id="pack",
        sessions=[session],
        runner=_PrefixRunner(),  # type: ignore[arg-type]
    )
    assert outcomes[0].detected_off is True
    assert outcomes[0].first_off_turn == 2


def test_prefix_text_does_not_include_gold_label():
    session = _mk_session(
        "sess-prefix",
        actor_id="actor-prefix",
        label_session="attack",
        turns=["alpha", "beta"],
        label_turns=["benign", "attack"],
    )
    text = support_eval._build_prefix_text(session.turns)  # noqa: SLF001
    assert "label=" not in text
    assert "attack" not in text
    assert "benign" not in text
    assert text == "alpha\nbeta"


def test_blind_trust_for_source_type_mapping():
    assert support_eval._blind_trust_for_source_type("internal_trusted") == "trusted"  # noqa: SLF001
    assert support_eval._blind_trust_for_source_type("external_untrusted") == "untrusted"  # noqa: SLF001
    assert support_eval._blind_trust_for_source_type("semi_trusted") == "semi"  # noqa: SLF001


def test_summarize_variant_outcomes_with_enforcement_correctness():
    outcomes = [
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_STATEFUL,
            session_id="atk1",
            actor_id="a1",
            family="fam",
            bucket="core",
            label_session="attack",
            turn_count=4,
            detected_off=True,
            first_off_turn=2,
            session_risk=0.8,
            max_turn_p=0.8,
            late_detect=False,
            off_turn_walls=[support_eval.WALL_EXFIL],
            off_turn_action_types=["HUMAN_ESCALATE"],
            off_turn_control_outcome="HUMAN_ESCALATE",
        ),
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_STATEFUL,
            session_id="atk2",
            actor_id="a2",
            family="fam",
            bucket="core",
            label_session="attack",
            turn_count=4,
            detected_off=True,
            first_off_turn=4,
            session_risk=0.9,
            max_turn_p=0.9,
            late_detect=True,
            off_turn_walls=[support_eval.WALL_TOOL],
            off_turn_action_types=["TOOL_FREEZE"],
            off_turn_control_outcome="TOOL_FREEZE",
        ),
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_STATEFUL,
            session_id="atk3",
            actor_id="a3",
            family="fam",
            bucket="core",
            label_session="attack",
            turn_count=4,
            detected_off=False,
            first_off_turn=None,
            session_risk=0.2,
            max_turn_p=0.2,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome="ALLOW",
        ),
        support_eval.SessionOutcome(
            pack_id="p1",
            variant=support_eval.VARIANT_STATEFUL,
            session_id="ben1",
            actor_id="b1",
            family="fam",
            bucket="core",
            label_session="benign",
            turn_count=3,
            detected_off=True,
            first_off_turn=1,
            session_risk=0.7,
            max_turn_p=0.7,
            late_detect=False,
            off_turn_walls=[support_eval.WALL_EXFIL, support_eval.WALL_TOOL],
            off_turn_action_types=["HUMAN_ESCALATE"],
            off_turn_control_outcome="SOFT_BLOCK",
        ),
    ]
    summary = support_eval.summarize_variant_outcomes(outcomes)
    assert abs(summary["session_attack_off_rate"] - (2.0 / 3.0)) < 1e-9
    assert summary["session_benign_off_rate"] == 1.0
    assert summary["steps_to_off"]["median"] == 3.0
    assert summary["steps_to_off"]["never_detected"] == 1
    assert summary["enforcement_correctness"]["exfil_escalate_rate"] == 1.0
    assert summary["enforcement_correctness"]["tool_freeze_rate"] == 0.5
    assert summary["enforcement_correctness"]["exfil_plus_tool_freeze_rate"] == 0.0


def test_matched_benign_rate_comparison_uses_stateful_target():
    def _outcome(variant: str, sid: str, label: str, risk: float, detected: bool) -> support_eval.SessionOutcome:
        return support_eval.SessionOutcome(
            pack_id="p1",
            variant=variant,
            session_id=sid,
            actor_id=f"a-{sid}",
            family="fam",
            bucket="core",
            label_session=label,
            turn_count=3,
            detected_off=detected,
            first_off_turn=(2 if detected else None),
            session_risk=risk,
            max_turn_p=risk,
            late_detect=False,
            off_turn_walls=[],
            off_turn_action_types=[],
            off_turn_control_outcome=("SOFT_BLOCK" if detected else "ALLOW"),
        )

    outcomes = {
        support_eval.VARIANT_STATEFUL: [
            _outcome(support_eval.VARIANT_STATEFUL, "s_atk_1", "attack", 0.9, True),
            _outcome(support_eval.VARIANT_STATEFUL, "s_atk_2", "attack", 0.2, False),
            _outcome(support_eval.VARIANT_STATEFUL, "s_ben_1", "benign", 0.4, False),
            _outcome(support_eval.VARIANT_STATEFUL, "s_ben_2", "benign", 0.8, True),
        ],
        support_eval.VARIANT_BASELINE_B: [
            _outcome(support_eval.VARIANT_BASELINE_B, "b_atk_1", "attack", 0.7, True),
            _outcome(support_eval.VARIANT_BASELINE_B, "b_atk_2", "attack", 0.3, False),
            _outcome(support_eval.VARIANT_BASELINE_B, "b_ben_1", "benign", 0.6, True),
            _outcome(support_eval.VARIANT_BASELINE_B, "b_ben_2", "benign", 0.1, False),
        ],
    }
    overall = {
        support_eval.VARIANT_STATEFUL: support_eval.summarize_variant_outcomes(outcomes[support_eval.VARIANT_STATEFUL]),
        support_eval.VARIANT_BASELINE_B: support_eval.summarize_variant_outcomes(outcomes[support_eval.VARIANT_BASELINE_B]),
    }
    matched = support_eval.build_matched_benign_rate_comparison(
        outcomes_by_variant=outcomes,
        overall_metrics=overall,
        reference_variant=support_eval.VARIANT_STATEFUL,
        compare_variants=[support_eval.VARIANT_BASELINE_B],
    )
    row = matched["variants"][support_eval.VARIANT_BASELINE_B]
    assert abs(float(matched["target_session_benign_off_rate"]) - 0.5) < 1e-9
    assert abs(float(row["session_benign_off_rate_matched"]) - 0.5) < 1e-9
    assert 0.0 <= float(row["tau_matched"]) <= 1.0


def _write_pack_runtime(pack_dir: Path, *, attack_trigger: str) -> None:
    (pack_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (pack_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (pack_dir / "README.md").write_text("# pack\n", encoding="utf-8")
    rows = [
        {
            "session_id": "attack_sess",
            "turn_id": 1,
            "text": "benign context",
            "label_turn": "benign",
            "label_session": "attack",
            "family": "xsrc",
            "source_ref": "a/1",
            "source_type": "external_untrusted",
            "actor_id": "actor_attack",
            "bucket": "core",
            "eval_slice": "text_intrinsic",
        },
        {
            "session_id": "attack_sess",
            "turn_id": 2,
            "text": attack_trigger,
            "label_turn": "attack",
            "label_session": "attack",
            "family": "xsrc",
            "source_ref": "a/2",
            "source_type": "external_untrusted",
            "actor_id": "actor_attack",
            "bucket": "core",
            "eval_slice": "text_intrinsic",
        },
        {
            "session_id": "benign_sess",
            "turn_id": 1,
            "text": "normal request",
            "label_turn": "benign",
            "label_session": "benign",
            "family": "xsrc",
            "source_ref": "b/1",
            "source_type": "internal_trusted",
            "actor_id": "actor_benign",
            "bucket": "core",
            "eval_slice": "text_intrinsic",
        },
    ]
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    (pack_dir / "runtime" / "session_pack.jsonl").write_text(body, encoding="utf-8")


def test_run_eval_smoke_four_packs(monkeypatch):
    tmp_path = _workspace_tmp("run_eval_smoke")
    try:
        packs_root = tmp_path / "unpacked"
        artifacts_root = tmp_path / "artifacts"
        for idx in range(1, 5):
            _write_pack_runtime(packs_root / f"pack_{idx}", attack_trigger=f"TRIGGER_{idx}")

        class _FakeOmegaDecisionRunner:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

            def reset(self, *, session_id: str, actor_id: str) -> None:
                _ = (session_id, actor_id)

            def projector_status(self) -> dict:
                return {"mode": "fake"}

            def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> support_eval.TurnSignal:
                text = str(turn.text).lower()
                off = "trigger" in text
                walls = [support_eval.WALL_EXFIL, support_eval.WALL_TOOL] if off else []
                actions = ["HUMAN_ESCALATE", "TOOL_FREEZE"] if off else []
                return support_eval.TurnSignal(
                    turn_id=int(turn.turn_id),
                    off=off,
                    max_p=0.9 if off else 0.1,
                    max_m_next=0.0,
                    sum_m_next=0.0,
                    count_m_ge_theta=0,
                    walls=walls,
                    action_types=actions,
                    control_outcome="SOFT_BLOCK" if off else "ALLOW",
                )

        monkeypatch.setattr(support_eval, "OmegaDecisionRunner", _FakeOmegaDecisionRunner)
        report = support_eval.run_eval(
            packs_root=packs_root,
            profile="dev",
            stateful_mode="hybrid_api",
            strict_projector=True,
            allow_api_fallback=False,
            enable_stateful_support_tuning=False,
            api_model="gpt-5.4-mini",
            api_base_url=None,
            api_timeout_sec=None,
            api_retries=None,
            api_cache_path=None,
            api_error_log_path=None,
            baseline_b_calibration="benign_q95",
            baseline_c_mode="prefix_scan",
            artifacts_root=artifacts_root,
            seed=41,
        )

        assert set(report["metrics"]["overall"].keys()) == {
            support_eval.VARIANT_STATEFUL,
            support_eval.VARIANT_BASELINE_A,
            support_eval.VARIANT_BASELINE_B,
            support_eval.VARIANT_BASELINE_C,
        }
        report_path = Path(report["artifacts"]["report_json"])
        assert report_path.exists()
        assert Path(report["artifacts"]["rows_jsonl"]).exists()
        assert Path(report["artifacts"]["calibration_json"]).exists()
        assert Path(report["artifacts"]["packs_summary_json"]).exists()
        assert Path(report["artifacts"]["baseline_d_calibration_json"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_eval_fail_fast_on_semantic_inactive(monkeypatch):
    tmp_path = _workspace_tmp("run_eval_semantic_guard")
    try:
        packs_root = tmp_path / "unpacked"
        artifacts_root = tmp_path / "artifacts"
        _write_pack_runtime(packs_root / "pack_1", attack_trigger="TRIGGER_X")

        class _FakeOmegaDecisionRunner:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

            def reset(self, *, session_id: str, actor_id: str) -> None:
                _ = (session_id, actor_id)

            def projector_status(self) -> dict:
                return {"mode": "fake", "semantic": {"active": False, "error": "semantic init failed"}}

            def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> support_eval.TurnSignal:
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

        monkeypatch.setattr(support_eval, "OmegaDecisionRunner", _FakeOmegaDecisionRunner)
        with pytest.raises(RuntimeError, match="semantic runtime inactive"):
            support_eval.run_eval(
                packs_root=packs_root,
                profile="dev",
                stateful_mode="hybrid_api",
                strict_projector=True,
                allow_api_fallback=False,
                require_semantic_active=True,
                enable_stateful_support_tuning=False,
                api_model="gpt-5.4-mini",
                api_base_url=None,
                api_timeout_sec=None,
                api_retries=None,
                api_cache_path=None,
                api_error_log_path=None,
                baseline_b_calibration="benign_q95",
                baseline_c_mode="prefix_scan",
                artifacts_root=artifacts_root,
                seed=41,
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_eval_smoke_four_packs_with_baseline_d(monkeypatch):
    tmp_path = _workspace_tmp("run_eval_smoke_d")
    try:
        packs_root = tmp_path / "unpacked"
        artifacts_root = tmp_path / "artifacts"
        for idx in range(1, 5):
            _write_pack_runtime(packs_root / f"pack_{idx}", attack_trigger=f"TRIGGER_{idx}")

        class _FakeOmegaDecisionRunner:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

            def reset(self, *, session_id: str, actor_id: str) -> None:
                _ = (session_id, actor_id)

            def projector_status(self) -> dict:
                return {"mode": "fake"}

            def run_turn(self, *, session_id: str, actor_id: str, turn: SessionTurnRow) -> support_eval.TurnSignal:
                text = str(turn.text).lower()
                off = "trigger" in text
                walls = [support_eval.WALL_EXFIL, support_eval.WALL_TOOL] if off else []
                actions = ["HUMAN_ESCALATE", "TOOL_FREEZE"] if off else []
                return support_eval.TurnSignal(
                    turn_id=int(turn.turn_id),
                    off=off,
                    max_p=0.9 if off else 0.1,
                    max_m_next=0.0,
                    sum_m_next=0.0,
                    count_m_ge_theta=0,
                    walls=walls,
                    action_types=actions,
                    control_outcome="SOFT_BLOCK" if off else "ALLOW",
                )

        class _FakeBareDetectorRunner:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

            def status(self) -> dict:
                return {"mode": "fake_bare_detector"}

            def run_turn(self, *, turn: SessionTurnRow) -> support_eval.DetectorTurnPrediction:
                text = str(turn.text).lower()
                risk = 0.9 if "trigger" in text else 0.1
                wall_scores = {
                    "secret_exfiltration": risk,
                    "tool_or_action_abuse": 0.0,
                    "policy_override_or_jailbreak": 0.0,
                    "integrity_subversion": 0.0,
                }
                return support_eval.DetectorTurnPrediction(
                    signal=support_eval.TurnSignal(
                        turn_id=int(turn.turn_id),
                        off=False,
                        max_p=risk,
                        max_m_next=0.0,
                        sum_m_next=0.0,
                        count_m_ge_theta=0,
                        walls=[support_eval.WALL_EXFIL] if risk >= 0.5 else [],
                        action_types=[],
                        control_outcome="ALLOW",
                    ),
                    wall_scores=wall_scores,
                )

        monkeypatch.setattr(support_eval, "OmegaDecisionRunner", _FakeOmegaDecisionRunner)
        monkeypatch.setattr(support_eval, "BareLLMDetectorRunner", _FakeBareDetectorRunner)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        report = support_eval.run_eval(
            packs_root=packs_root,
            profile="dev",
            stateful_mode="hybrid_api",
            strict_projector=True,
            allow_api_fallback=False,
            enable_stateful_support_tuning=False,
            api_model="gpt-5.4-mini",
            api_base_url=None,
            api_timeout_sec=None,
            api_retries=None,
            api_cache_path=None,
            api_error_log_path=None,
            baseline_b_calibration="benign_q95",
            baseline_c_mode="prefix_scan",
            baseline_d_enable=True,
            baseline_d_model="gpt-5.4-mini",
            baseline_d_base_url=None,
            baseline_d_timeout_sec=5.0,
            baseline_d_retries=0,
            baseline_d_calibration="benign_q95",
            baseline_d_mode="per_turn_only",
            artifacts_root=artifacts_root,
            seed=41,
        )

        assert set(report["metrics"]["overall"].keys()) == {
            support_eval.VARIANT_STATEFUL,
            support_eval.VARIANT_BASELINE_A,
            support_eval.VARIANT_BASELINE_B,
            support_eval.VARIANT_BASELINE_C,
            support_eval.VARIANT_BASELINE_D,
        }
        assert report["calibration"]["baseline_d"] is not None
        assert Path(report["artifacts"]["baseline_d_calibration_json"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
