from __future__ import annotations

import json
from pathlib import Path

from scripts.build_attack_layer_pack_v1 import build_attack_layer_pack_v1
from scripts.run_attack_layer_cycle import (
    ManifestTurn,
    _extract_casepacks,
    _gate_checks,
    _hash_signature,
    _load_manifest_rows,
    _source_id_for_turn,
    _summary,
    evaluate_attack_rows,
    map_control_outcome_to_policy,
    validate_manifest_rows,
)


class _FakeRunner:
    def __init__(self) -> None:
        self._session_id = ""
        self._actor_id = ""

    def reset(self, *, session_id: str, actor_id: str) -> None:
        self._session_id = str(session_id)
        self._actor_id = str(actor_id)

    def run_turn(self, *, turn, actor_id: str):
        _ = actor_id
        text = f"{turn.input_text} {turn.tool_output_text}".lower()
        pred_policy = "PASS"
        off = False

        if turn.family == "benign_docs":
            pred_policy = "BLOCK"
            off = True
        elif turn.family == "distributed_wo_explicit":
            pred_policy = "PASS"
            off = False
        elif any(x in text for x in ("ignore", "reveal", "secret", "call_tool", "override")):
            pred_policy = "BLOCK"
            off = True
        elif "approval" in text or "training example" in text:
            pred_policy = "WARN"
            off = False

        control_outcome = {
            "PASS": "ALLOW",
            "WARN": "REQUIRE_APPROVAL",
            "BLOCK": "SOFT_BLOCK",
        }[pred_policy]
        return {
            "off": off,
            "max_p": 1.0 if off else (0.5 if pred_policy == "WARN" else 0.0),
            "control_outcome": control_outcome,
            "predicted_policy": pred_policy,
            "trace_id": f"tr::{self._session_id}::{turn.turn_id}",
            "decision_id": f"dc::{self._actor_id}::{turn.turn_id}",
            "api_schema_version": None,
            "api_confidence": None,
            "api_defensive_context": False,
            "api_directive_intent_any": False,
            "api_deescalation_applied": False,
            "api_max_positive_pressure": 0.0,
            "api_negative_polarity_any": False,
            "pi0_hard_any": False,
            "pi0_soft_any": True,
            "hybrid_soft_suppressed_any": False,
            "hybrid_soft_confirmation_any": False,
            "hybrid_soft_suppressed_walls": [],
            "hybrid_benign_stabilizer_applied": False,
            "hybrid_benign_stabilizer_walls": [],
            "projector_top_signals": [],
        }


class _FakeRunnerWithApi(_FakeRunner):
    def run_turn(self, *, turn, actor_id: str):
        payload = super().run_turn(turn=turn, actor_id=actor_id)
        payload.update(
            {
                "api_schema_version": "api_hybrid_v2",
                "api_confidence": 0.91,
                "api_defensive_context": True,
                "api_directive_intent_any": False,
                "api_deescalation_applied": True,
                "api_max_positive_pressure": 0.12,
                "api_negative_polarity_any": True,
                "pi0_hard_any": False,
                "pi0_soft_any": True,
                "hybrid_soft_suppressed_any": True,
                "hybrid_soft_confirmation_any": False,
                "hybrid_soft_suppressed_walls": ["override_instructions"],
                "hybrid_benign_stabilizer_applied": True,
                "hybrid_benign_stabilizer_walls": ["override_instructions"],
                "projector_top_signals": [{"signal": "api_perception", "hits": 2}],
            }
        )
        return payload


def test_policy_mapping_partner_contract():
    assert map_control_outcome_to_policy("ALLOW") == "PASS"
    assert map_control_outcome_to_policy("WARN") == "WARN"
    assert map_control_outcome_to_policy("REQUIRE_APPROVAL") == "WARN"
    assert map_control_outcome_to_policy("SOFT_BLOCK") == "BLOCK"
    assert map_control_outcome_to_policy("TOOL_FREEZE") == "BLOCK"


def test_attack_cycle_determinism_and_casepacks(tmp_path: Path):
    out_root = tmp_path / "attack_layers" / "v1"
    build_attack_layer_pack_v1(out_root=out_root, seed=41)
    rows = _load_manifest_rows(out_root / "manifest_all.jsonl")
    validate_manifest_rows(rows)

    out1 = evaluate_attack_rows(rows=rows, core_runner=_FakeRunner(), cross_runner=_FakeRunner())
    out2 = evaluate_attack_rows(rows=rows, core_runner=_FakeRunner(), cross_runner=_FakeRunner())

    case_rows_sig_1 = [
        {
            "case_id": x.case_id,
            "predicted_policy": x.predicted_policy,
            "predicted_off": x.predicted_off,
            "expected_policy": x.expected_policy,
            "expected_off": x.expected_off,
            "match": x.match,
        }
        for x in out1["case_results"]
    ]
    case_rows_sig_2 = [
        {
            "case_id": x.case_id,
            "predicted_policy": x.predicted_policy,
            "predicted_off": x.predicted_off,
            "expected_policy": x.expected_policy,
            "expected_off": x.expected_off,
            "match": x.match,
        }
        for x in out2["case_results"]
    ]
    assert _hash_signature(case_rows_sig_1) == _hash_signature(case_rows_sig_2)

    casepacks = _extract_casepacks(out1["case_results"])
    assert len(casepacks["fn_casepack"]) >= 1
    assert len(casepacks["fp_casepack"]) >= 1

    summary = _summary(out1["case_results"])
    assert 0.0 <= float(summary["match_rate"]) <= 1.0
    gates = _gate_checks(out1["case_results"], micro_pass=1)
    assert isinstance(gates["passed"], bool)
    assert len(gates["checks"]) >= 5


def test_attack_cycle_layer_smoke(tmp_path: Path):
    out_root = tmp_path / "attack_layers" / "v1"
    build_attack_layer_pack_v1(out_root=out_root, seed=41)
    rows = _load_manifest_rows(out_root / "manifest_all.jsonl")
    validate_manifest_rows(rows)

    layers = sorted({r.layer for r in rows})
    for layer in layers:
        subset = [r for r in rows if r.layer == layer]
        out = evaluate_attack_rows(rows=subset, core_runner=_FakeRunner(), cross_runner=_FakeRunner())
        assert len(out["case_results"]) >= 1


def test_turn_label_and_generation_trace_id_passthrough(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    row = {
        "case_id": "AL-PS-001",
        "layer": "tool_chain",
        "family": "benign_tool_output",
        "mode": "stateless",
        "phase": "active",
        "session_id": "sess_al_ps_001",
        "turn_id": 1,
        "input_text": "Parser output for benign pipeline continuation.",
        "tool_output_text": "Parsed receipt amount=18.20.",
        "expected_policy": "PASS",
        "expected_off": False,
        "expected_block_turn": None,
        "tags": ["tool_chain", "api_gen", "benign"],
        "source": "attack_layers_v1_api_generative",
        "notes": "",
        "turn_label": "benign_seed",
        "generation_trace_id": "rsp_test_001",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = _load_manifest_rows(path)
    validate_manifest_rows(rows)

    out = evaluate_attack_rows(rows=rows, core_runner=_FakeRunner(), cross_runner=_FakeRunner())
    assert len(out["turn_rows"]) == 1
    r = out["turn_rows"][0]
    assert r["turn_label"] == "benign_seed"
    assert r["generation_trace_id"] == "rsp_test_001"
    assert "api_schema_version" in r
    assert "api_confidence" in r
    assert "api_defensive_context" in r
    assert "api_directive_intent_any" in r
    assert "api_deescalation_applied" in r
    assert "api_max_positive_pressure" in r
    assert "api_negative_polarity_any" in r
    assert "pi0_hard_any" in r
    assert "pi0_soft_any" in r
    assert "hybrid_soft_suppressed_any" in r
    assert "hybrid_soft_confirmation_any" in r
    assert "hybrid_soft_suppressed_walls" in r
    assert "hybrid_benign_stabilizer_applied" in r
    assert "hybrid_benign_stabilizer_walls" in r
    assert "projector_top_signals" in r


def test_turn_rows_include_api_diagnostics(tmp_path: Path):
    path = tmp_path / "manifest_api_diag.jsonl"
    row = {
        "case_id": "AL-API-DIAG-001",
        "layer": "context_accumulation",
        "family": "benign_long_context",
        "mode": "stateless",
        "phase": "active",
        "session_id": "sess_api_diag_001",
        "turn_id": 1,
        "input_text": "safe request",
        "tool_output_text": "",
        "expected_policy": "PASS",
        "expected_off": False,
        "expected_block_turn": None,
        "tags": ["api_gen", "benign"],
        "source": "attack_layers_v1_api_generative",
        "notes": "",
        "turn_label": "benign_seed",
        "generation_trace_id": "rsp_diag_001",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = _load_manifest_rows(path)
    validate_manifest_rows(rows)

    out = evaluate_attack_rows(rows=rows, core_runner=_FakeRunnerWithApi(), cross_runner=_FakeRunnerWithApi())
    r = out["turn_rows"][0]
    assert r["api_schema_version"] == "api_hybrid_v2"
    assert float(r["api_confidence"]) == 0.91
    assert bool(r["api_defensive_context"]) is True
    assert bool(r["api_directive_intent_any"]) is False
    assert bool(r["api_deescalation_applied"]) is True
    assert float(r["api_max_positive_pressure"]) == 0.12
    assert bool(r["api_negative_polarity_any"]) is True
    assert bool(r["pi0_hard_any"]) is False
    assert bool(r["pi0_soft_any"]) is True
    assert bool(r["hybrid_soft_suppressed_any"]) is True
    assert bool(r["hybrid_soft_confirmation_any"]) is False
    assert r["hybrid_soft_suppressed_walls"] == ["override_instructions"]
    assert bool(r["hybrid_benign_stabilizer_applied"]) is True
    assert r["hybrid_benign_stabilizer_walls"] == ["override_instructions"]
    assert r["projector_top_signals"] == [{"signal": "api_perception", "hits": 2}]


def test_source_id_for_turn_is_turn_scoped():
    turn1 = ManifestTurn(
        case_id="AL-TST-001",
        layer="context_accumulation",
        family="benign_long_context",
        mode="session",
        phase="active",
        session_id="sess_al_tst_001",
        turn_id=1,
        input_text="safe",
        tool_output_text="",
        turn_label="benign_seed",
        generation_trace_id="gen_001",
        expected_policy="PASS",
        expected_off=False,
        expected_block_turn=None,
        tags=["benign"],
        source="attack_layers_v1_api_generative",
        notes="",
    )
    turn2 = ManifestTurn(
        case_id="AL-TST-001",
        layer="context_accumulation",
        family="benign_long_context",
        mode="session",
        phase="active",
        session_id="sess_al_tst_001",
        turn_id=2,
        input_text="safe 2",
        tool_output_text="",
        turn_label="bridge",
        generation_trace_id="gen_001",
        expected_policy="PASS",
        expected_off=False,
        expected_block_turn=None,
        tags=["benign"],
        source="attack_layers_v1_api_generative",
        notes="",
    )

    s1 = _source_id_for_turn(turn1, "input")
    s2 = _source_id_for_turn(turn2, "input")
    st = _source_id_for_turn(turn1, "tool")
    assert s1.endswith(":turn:001:input")
    assert s2.endswith(":turn:002:input")
    assert st.endswith(":turn:001:tool")
    assert s1 != s2
