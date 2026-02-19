from __future__ import annotations

from omega.oss_runtime import run_demo, run_demo_all, run_quick_eval


def test_demo_attack_off_and_freeze():
    out = run_demo("attack", llm_backend="mock")
    action_types = {action["type"] for action in out["actions"]}
    assert out["off"] is True
    assert "SOFT_BLOCK" in action_types
    assert "TOOL_FREEZE" in action_types
    assert out["tool_executions_count"] == 0


def test_demo_benign_is_ok():
    out = run_demo("benign", llm_backend="mock")
    assert out["off"] is False
    assert out["actions"] == []


def test_demo_all_runs():
    out = run_demo_all(llm_backend="mock")
    assert "attack" in out
    assert "benign" in out


def test_quick_eval_outputs_required_fields():
    out = run_quick_eval(llm_backend="mock")
    assert "attack_pass_rate" in out
    assert "benign_pass_rate" in out
    assert "fp" in out
    assert "fn" in out
    assert "steps_to_off" in out
    assert out["passed"] is True
