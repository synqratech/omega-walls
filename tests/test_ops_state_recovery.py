from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np
import pytest

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import OffAction
from omega.policy.cross_session_state import CrossSessionStateManager
from scripts.ops_state_recovery import (
    clear_actor_freeze,
    clear_quarantine_all,
    clear_quarantine_source,
    collect_state_snapshot,
)


@dataclass
class _StepResultStub:
    m_next: np.ndarray


def _case_dir() -> Path:
    path = Path("artifacts/test_tmp") / f"ops_recovery_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mk_manager(case_dir: Path) -> tuple[CrossSessionStateManager, Path]:
    cfg = load_resolved_config(profile="dev").resolved
    cfg = dict(cfg)
    cfg["off_policy"] = dict(cfg["off_policy"])
    cfg["off_policy"]["cross_session"] = dict(cfg["off_policy"]["cross_session"])
    sqlite_path = case_dir / "cross_session_state.db"
    cfg["off_policy"]["cross_session"]["sqlite_path"] = str(sqlite_path)
    manager = CrossSessionStateManager.from_config(cfg)
    return manager, sqlite_path


def _seed_actor_state(manager: CrossSessionStateManager, actor_id: str, source_id: str = "web:test-source") -> None:
    manager.hydrate_actor_state(actor_id=actor_id, session_id=f"{actor_id}-s1")
    manager.record_step(
        actor_id=actor_id,
        session_id=f"{actor_id}-s1",
        step_result=_StepResultStub(m_next=np.array([0.6, 0.0, 0.0, 0.0], dtype=float)),
        policy_actions=[
            OffAction(
                type="TOOL_FREEZE",
                target="TOOLS",
                tool_mode="TOOLS_DISABLED",
                allowlist=["retrieval_readonly"],
                horizon_steps=12,
            ),
            OffAction(
                type="SOURCE_QUARANTINE",
                target="SOURCE",
                source_ids=[source_id],
                horizon_steps=24,
            ),
        ],
        packet_items=[],
    )
    manager.hydrate_actor_state(actor_id=actor_id, session_id=f"{actor_id}-s2")
    manager.record_step(
        actor_id=actor_id,
        session_id=f"{actor_id}-s2",
        step_result=_StepResultStub(m_next=np.array([0.7, 0.0, 0.0, 0.0], dtype=float)),
        policy_actions=[
            OffAction(
                type="SOURCE_QUARANTINE",
                target="SOURCE",
                source_ids=[source_id],
                horizon_steps=24,
            )
        ],
        packet_items=[],
    )


def test_ops_state_recovery_clear_freeze_does_not_clear_quarantine():
    manager, sqlite_path = _mk_manager(_case_dir())
    _seed_actor_state(manager, actor_id="actor-freeze", source_id="web:freeze-source")

    before = collect_state_snapshot(sqlite_path=sqlite_path)
    assert before["counts"]["actor_freeze"] >= 1
    assert before["counts"]["source_quarantine"] >= 1

    removed = clear_actor_freeze(sqlite_path=sqlite_path, actor_id="actor-freeze")
    assert removed >= 1

    after = collect_state_snapshot(sqlite_path=sqlite_path)
    assert after["counts"]["actor_freeze"] == 0
    assert after["counts"]["source_quarantine"] >= 1


def test_ops_state_recovery_quarantine_source_and_all():
    manager, sqlite_path = _mk_manager(_case_dir())
    _seed_actor_state(manager, actor_id="actor-quarantine", source_id="web:q-source")

    removed_one = clear_quarantine_source(sqlite_path=sqlite_path, source_id="web:q-source")
    assert removed_one >= 1
    after_one = collect_state_snapshot(sqlite_path=sqlite_path)
    assert after_one["counts"]["source_quarantine"] == 0

    _seed_actor_state(manager, actor_id="actor-quarantine-2", source_id="web:q-source-2")
    with pytest.raises(ValueError):
        clear_quarantine_all(sqlite_path=sqlite_path, confirm=False)
    removed_all = clear_quarantine_all(sqlite_path=sqlite_path, confirm=True)
    assert removed_all >= 1
    after_all = collect_state_snapshot(sqlite_path=sqlite_path)
    assert after_all["counts"]["source_quarantine"] == 0


def test_ops_state_recovery_reset_actor_clears_actor_scoped_state():
    manager, sqlite_path = _mk_manager(_case_dir())
    _seed_actor_state(manager, actor_id="actor-reset", source_id="web:shared-source")

    before_actor = collect_state_snapshot(sqlite_path=sqlite_path, actor_id="actor-reset")
    assert len(before_actor["actor_state"]) >= 1
    assert len(before_actor["actor_freeze"]) >= 1
    assert len(before_actor["session_actor"]) >= 1

    manager.reset_actor("actor-reset")
    after_actor = collect_state_snapshot(sqlite_path=sqlite_path, actor_id="actor-reset")
    assert len(after_actor["actor_state"]) == 0
    assert len(after_actor["actor_freeze"]) == 0
    assert len(after_actor["session_actor"]) == 0
