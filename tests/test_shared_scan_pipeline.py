from __future__ import annotations

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult
from omega.runtime.scan_pipeline import compose_control_outcome_state, dedupe_pressure_items_step_local, evaluate_projection_phase


def _proj(*, semantic_failed: bool) -> ProjectionResult:
    return ProjectionResult(
        doc_id="doc-1",
        v=np.zeros(4, dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, 0, 0, 0],
            debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
            matches={
                "api_perception": {
                    "semantic_failed": bool(semantic_failed),
                    "semantic_status": ("semantic_failed" if semantic_failed else "ok"),
                }
            },
        ),
    )


def _step_result() -> OmegaStepResult:
    return OmegaStepResult(
        session_id="sess-1",
        step=1,
        v_total=np.zeros(4, dtype=float),
        p=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
        m_prev=np.zeros(4, dtype=float),
        m_next=np.array([0.4, 0.0, 0.0, 0.0], dtype=float),
        off=True,
        reasons=OmegaOffReasons(True, False, False, False),
        top_docs=["doc-1"],
        contribs=[],
    )


def test_projection_phase_detects_semantic_failure() -> None:
    cfg = {"projector": {"api_perception": {"semantic_failure_policy": "escalate"}}}
    phase = evaluate_projection_phase(cfg=cfg, projections=[_proj(semantic_failed=True)])
    assert phase.semantic_failure_detected is True
    assert phase.semantic_failure_policy == "escalate"
    assert phase.semantic_failure_status == "semantic_failed"
    assert phase.semantic_policy_branch == "escalate"


def test_control_outcome_state_matches_expected_fields() -> None:
    cfg = {"projector": {"api_perception": {"semantic_failure_policy": "degrade"}}}
    semantic_phase = evaluate_projection_phase(cfg=cfg, projections=[_proj(semantic_failed=False)])
    out = compose_control_outcome_state(
        walls=["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        step_result=_step_result(),
        policy_action_types=["WARN"],
        cross_action_types=["SOURCE_QUARANTINE"],
        semantic_phase=semantic_phase,
        extra_reason_flags=["ingestion_error"],
    )
    assert out.walls_triggered == ["override_instructions"]
    assert out.action_types == ["SOURCE_QUARANTINE", "WARN"]
    assert out.intended_action_types == ["SOURCE_QUARANTINE", "WARN"]
    assert "reason_spike" in out.reason_flags
    assert "ingestion_error" in out.reason_flags
    assert out.semantic_failure_status == "ok"
    assert out.semantic_policy_branch == "none"


def test_control_outcome_state_escalates_on_semantic_failure() -> None:
    cfg = {"projector": {"api_perception": {"semantic_failure_policy": "escalate"}}}
    semantic_phase = evaluate_projection_phase(cfg=cfg, projections=[_proj(semantic_failed=True)])
    out = compose_control_outcome_state(
        walls=["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        step_result=_step_result(),
        policy_action_types=["WARN"],
        semantic_phase=semantic_phase,
    )
    assert "HUMAN_ESCALATE" in out.action_types
    assert "semantic_failed" in out.reason_flags
    assert "semantic_failure_policy_escalate" in out.reason_flags


def test_dedupe_pressure_items_step_local_preserves_source_level_signal() -> None:
    items = [
        ContentItem(
            doc_id="d1",
            source_id="src:1",
            source_type="other",
            trust="untrusted",
            text="same",
            content_hash="h1",
            boundary_step=1,
        ),
        ContentItem(
            doc_id="d2",
            source_id="src:1",
            source_type="other",
            trust="untrusted",
            text="same",
            content_hash="h1",
            boundary_step=1,
        ),
        ContentItem(
            doc_id="d3",
            source_id="src:2",
            source_type="other",
            trust="untrusted",
            text="same",
            content_hash="h1",
            boundary_step=1,
        ),
    ]
    kept, stats = dedupe_pressure_items_step_local(items=items, current_step=1)
    assert len(kept) == 2
    assert [x.doc_id for x in kept] == ["d1", "d3"]
    assert int(stats["deduped_count"]) == 1
    assert int(stats["deduped_by_content_hash_source"]) == 1
