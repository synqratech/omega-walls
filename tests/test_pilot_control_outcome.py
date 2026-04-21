from __future__ import annotations

import numpy as np

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import OmegaOffReasons, OmegaStepResult
from omega.policy.off_policy_v1 import OffPolicyV1
from tests.helpers import mk_item


def _step(*, off: bool, p: list[float], m_next: list[float]) -> OmegaStepResult:
    return OmegaStepResult(
        session_id="sess-pilot",
        step=2,
        v_total=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
        p=np.asarray(p, dtype=float),
        m_prev=np.asarray([0.05, 0.05, 0.05, 0.05], dtype=float),
        m_next=np.asarray(m_next, dtype=float),
        off=off,
        reasons=OmegaOffReasons(False, False, False, False),
        top_docs=["doc-1"],
        contribs=[],
    )


def test_pilot_warn_outcome_is_emitted_without_off() -> None:
    cfg = load_resolved_config(profile="pilot_canonical").resolved
    policy = OffPolicyV1(cfg)
    decision = policy.select_actions(
        _step(off=False, p=[0.35, 0.0, 0.0, 0.0], m_next=[0.10, 0.04, 0.03, 0.02]),
        [mk_item("doc-1", "benign but suspicious phrasing", source_id="web:warn")],
    )
    action_types = {a.type for a in decision.actions}
    assert decision.off is False
    assert "WARN" in action_types
    assert "REQUIRE_APPROVAL" not in action_types
    assert decision.control_outcome == "WARN"


def test_pilot_require_approval_outcome_is_emitted_for_off() -> None:
    cfg = load_resolved_config(profile="pilot_canonical").resolved
    policy = OffPolicyV1(cfg)
    decision = policy.select_actions(
        _step(off=True, p=[0.1, 0.95, 0.0, 0.0], m_next=[0.2, 0.7, 0.1, 0.1]),
        [mk_item("doc-1", "exfil attempt", source_id="web:off")],
    )
    action_types = {a.type for a in decision.actions}
    assert "SOFT_BLOCK" in action_types
    assert "SOURCE_QUARANTINE" in action_types
    assert "REQUIRE_APPROVAL" in action_types
    assert decision.control_outcome in {"SOURCE_QUARANTINE", "HUMAN_ESCALATE", "SOFT_BLOCK", "TOOL_FREEZE"}
