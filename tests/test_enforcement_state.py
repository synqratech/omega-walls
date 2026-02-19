from __future__ import annotations

from omega.interfaces.contracts_v1 import OffAction
from omega.policy.enforcement_state import EnforcementStateManager


def test_tool_freeze_persists_until_horizon():
    manager = EnforcementStateManager(strikes_to_quarantine=2, default_source_horizon_steps=24)
    manager.record_policy_actions(
        [
            OffAction(
                type="TOOL_FREEZE",
                target="TOOLS",
                tool_mode="TOOLS_DISABLED",
                horizon_steps=3,
            )
        ],
        step=5,
    )

    assert any(a.type == "TOOL_FREEZE" for a in manager.active_actions(step=5))
    assert any(a.type == "TOOL_FREEZE" for a in manager.active_actions(step=8))
    assert not any(a.type == "TOOL_FREEZE" for a in manager.active_actions(step=9))


def test_source_quarantine_activates_after_strike_threshold():
    manager = EnforcementStateManager(strikes_to_quarantine=2, default_source_horizon_steps=24)
    signal = OffAction(
        type="SOURCE_QUARANTINE",
        target="SOURCE",
        source_ids=["web:repeat.example"],
        horizon_steps=4,
    )

    manager.record_policy_actions([signal], step=1)
    assert manager.is_source_quarantined("web:repeat.example", step=1) is False

    manager.record_policy_actions([signal], step=2)
    assert manager.is_source_quarantined("web:repeat.example", step=2) is True
    assert manager.is_source_quarantined("web:repeat.example", step=6) is True
    assert manager.is_source_quarantined("web:repeat.example", step=7) is False
