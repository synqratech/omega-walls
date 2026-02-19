from __future__ import annotations

import numpy as np

from omega.interfaces.contracts_v1 import OmegaOffReasons, OmegaStepResult
from tests.helpers import mk_item


def _mk_step_result(off=True):
    return OmegaStepResult(
        session_id="sess-policy",
        step=3,
        v_total=np.array([0.3, 0.8, 0.9, 0.1]),
        p=np.array([0.2, 0.95, 0.92, 0.1]),
        m_prev=np.array([0.1, 0.1, 0.1, 0.1]),
        m_next=np.array([0.2, 0.5, 0.6, 0.2]),
        off=off,
        reasons=OmegaOffReasons(True, False, True, True),
        top_docs=["doc-1"],
        contribs=[],
    )


def test_off_policy_actions(off_policy):
    items = [
        mk_item("doc-1", "malicious", source_id="web:evil.example"),
        mk_item("doc-2", "benign", source_id="web:ok.example"),
    ]
    decision = off_policy.select_actions(_mk_step_result(off=True), items)
    types = [a.type for a in decision.actions]

    assert decision.off is True
    assert "SOFT_BLOCK" in types
    assert "TOOL_FREEZE" in types
    assert "HUMAN_ESCALATE" in types


def test_off_policy_source_quarantine_signal(off_policy):
    items = [mk_item("doc-1", "m", source_id="web:repeat.example")]
    first = off_policy.select_actions(_mk_step_result(off=True), items)
    second = off_policy.select_actions(_mk_step_result(off=True), items)

    assert "SOURCE_QUARANTINE" in [a.type for a in first.actions]
    assert "SOURCE_QUARANTINE" in [a.type for a in second.actions]
