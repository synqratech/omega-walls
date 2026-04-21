from __future__ import annotations

import numpy as np

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import OmegaOffReasons, OmegaStepResult
from omega.policy.off_policy_v1 import OffPolicyV1
from tests.helpers import mk_item


def _non_off_warn_like_step() -> OmegaStepResult:
    return OmegaStepResult(
        session_id="sess-isolation",
        step=1,
        v_total=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
        p=np.asarray([0.36, 0.0, 0.0, 0.0], dtype=float),
        m_prev=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float),
        m_next=np.asarray([0.10, 0.02, 0.02, 0.01], dtype=float),
        off=False,
        reasons=OmegaOffReasons(False, False, False, False),
        top_docs=["doc-1"],
        contribs=[],
    )


def test_control_outcome_profile_switch_is_isolated() -> None:
    dev = load_resolved_config(profile="dev").resolved
    local = load_resolved_config(profile="local_dev").resolved
    pilot = load_resolved_config(profile="pilot_canonical").resolved

    assert bool(dev["off_policy"]["control_outcome"]["warn"]["enabled"]) is False
    assert bool(local["off_policy"]["control_outcome"]["warn"]["enabled"]) is False
    assert bool(pilot["off_policy"]["control_outcome"]["warn"]["enabled"]) is True

    assert bool(dev["off_policy"]["control_outcome"]["require_approval"]["enabled"]) is False
    assert bool(local["off_policy"]["control_outcome"]["require_approval"]["enabled"]) is False
    assert bool(pilot["off_policy"]["control_outcome"]["require_approval"]["enabled"]) is True


def test_dev_and_local_non_off_behavior_stays_allow() -> None:
    step = _non_off_warn_like_step()
    items = [mk_item("doc-1", "warn-like but benign", source_id="web:benign")]
    for profile in ("dev", "local_dev"):
        cfg = load_resolved_config(profile=profile).resolved
        policy = OffPolicyV1(cfg)
        decision = policy.select_actions(step, items)
        assert decision.actions == []
        assert decision.control_outcome == "ALLOW"
