from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import OffAction
from omega.policy.cross_session_state import CrossSessionStateManager


def _cfg(tmp_path):
    cfg = deepcopy(load_resolved_config(profile="dev").resolved)
    cfg["off_policy"]["cross_session"]["sqlite_path"] = str(tmp_path / "cross_session_state.db")
    cfg["off_policy"]["cross_session"]["enabled"] = True
    cfg["off_policy"]["cross_session"]["actor_ttl_steps"] = 10
    cfg["off_policy"]["cross_session"]["decay"]["half_life_steps"] = 120
    cfg["off_policy"]["source_quarantine"]["duration_steps"] = 12
    return cfg


def _mk_tmp_dir(name: str) -> Path:
    path = Path("artifacts/test_tmp") / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_hydrate_decay_and_snapshot():
    tmp_path = _mk_tmp_dir("cross-session-state")
    cfg = _cfg(tmp_path)
    manager = CrossSessionStateManager.from_config(cfg)

    h1 = manager.hydrate_actor_state(actor_id="actor-1", session_id="sess-a")
    assert h1.carryover_applied is False
    assert np.allclose(h1.carried_scars_after_decay, np.zeros(4))

    manager.record_step(
        actor_id="actor-1",
        session_id="sess-a",
        step_result=SimpleNamespace(m_next=np.array([1.0, 0.0, 0.0, 0.0])),
        policy_actions=[],
        packet_items=[],
    )

    h2 = manager.hydrate_actor_state(actor_id="actor-1", session_id="sess-b")
    assert h2.carryover_applied is True
    assert float(h2.carried_scars_before_decay[0]) == 1.0
    assert 0.99 < float(h2.carried_scars_after_decay[0]) < 1.0

    snap = manager.snapshot(actor_id="actor-1", session_id="sess-b", step=1)
    assert "cross_session" in snap
    assert snap["cross_session"]["carryover_applied"] is True
    assert len(snap["cross_session"]["carried_scars_after_decay"]) == 4


def test_freeze_actor_scoped_and_quarantine_global():
    tmp_path = _mk_tmp_dir("cross-session-scope")
    cfg = _cfg(tmp_path)
    manager = CrossSessionStateManager.from_config(cfg)

    manager.hydrate_actor_state(actor_id="attacker-1", session_id="sess-a")
    manager.record_step(
        actor_id="attacker-1",
        session_id="sess-a",
        step_result=SimpleNamespace(m_next=np.array([0.0, 0.0, 0.0, 0.0])),
        policy_actions=[
            OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=3),
            OffAction(
                type="SOURCE_QUARANTINE",
                target="SOURCE",
                source_ids=["web:repeat.example"],
                horizon_steps=8,
            ),
        ],
        packet_items=[],
    )

    manager.hydrate_actor_state(actor_id="attacker-1", session_id="sess-b")
    manager.record_step(
        actor_id="attacker-1",
        session_id="sess-b",
        step_result=SimpleNamespace(m_next=np.array([0.0, 0.0, 0.0, 0.0])),
        policy_actions=[
            OffAction(
                type="SOURCE_QUARANTINE",
                target="SOURCE",
                source_ids=["web:repeat.example"],
                horizon_steps=8,
            )
        ],
        packet_items=[],
    )

    same_actor_actions = manager.active_actions(actor_id="attacker-1", session_id="sess-b", step=1)
    assert any(a.type == "TOOL_FREEZE" for a in same_actor_actions)
    assert any(a.type == "SOURCE_QUARANTINE" for a in same_actor_actions)

    manager.hydrate_actor_state(actor_id="other-actor", session_id="sess-c")
    other_actor_actions = manager.active_actions(actor_id="other-actor", session_id="sess-c", step=1)
    assert not any(a.type == "TOOL_FREEZE" for a in other_actor_actions)
    assert any(a.type == "SOURCE_QUARANTINE" for a in other_actor_actions)


def test_ttl_expiry_and_reset_actor():
    tmp_path = _mk_tmp_dir("cross-session-ttl")
    cfg = _cfg(tmp_path)
    cfg["off_policy"]["cross_session"]["actor_ttl_steps"] = 2
    manager = CrossSessionStateManager.from_config(cfg)

    manager.hydrate_actor_state(actor_id="actor-ttl", session_id="sess-1")
    manager.record_step(
        actor_id="actor-ttl",
        session_id="sess-1",
        step_result=SimpleNamespace(m_next=np.array([0.8, 0.2, 0.0, 0.0])),
        policy_actions=[],
        packet_items=[],
    )

    manager.hydrate_actor_state(actor_id="other", session_id="s2")
    manager.hydrate_actor_state(actor_id="other", session_id="s3")
    manager.hydrate_actor_state(actor_id="other", session_id="s4")

    expired = manager.hydrate_actor_state(actor_id="actor-ttl", session_id="sess-2")
    assert np.allclose(expired.carried_scars_after_decay, np.zeros(4))

    manager.reset_actor("actor-ttl")
    reset_h = manager.hydrate_actor_state(actor_id="actor-ttl", session_id="sess-3")
    assert reset_h.carryover_applied is False
