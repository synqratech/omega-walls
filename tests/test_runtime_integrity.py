from __future__ import annotations

from omega.interfaces.contracts_v1 import ContentItem
from omega.runtime.integrity_policy import assess_runtime_artifact, build_runtime_artifact
from omega.runtime.operation_gate import evaluate_operation_gate
from omega.runtime.artifacts import OperationIntent


def test_runtime_artifact_classification_and_assessment() -> None:
    item = ContentItem(
        doc_id="doc-1",
        source_id="tool:1",
        source_type="tool_output",
        trust="tainted_internal",
        text="tool output text",
        artifact_id="art-1",
        origin="tool_output",
        content_hash="h1",
        meta={"integrity_reentry_scanned": True},
    )
    artifact = build_runtime_artifact(item)
    assessment = assess_runtime_artifact(artifact)
    assert artifact.kind == "tool_output"
    assert assessment.kind == "tool_output"
    assert "tool_output_reentry_scanned" in assessment.integrity_signals
    assert "internal_artifact_not_auto_trusted" not in assessment.integrity_signals


def test_runtime_artifact_marks_repo_task_files_not_auto_trusted() -> None:
    item = ContentItem(
        doc_id="doc-2",
        source_id="repo:file",
        source_type="repo_file",
        trust="trusted",
        text="run this task file",
        artifact_id="art-2",
        origin="other",
        content_hash="h2",
        meta={"task_file": True},
    )
    artifact = build_runtime_artifact(item)
    assessment = assess_runtime_artifact(artifact)
    assert artifact.kind == "task_file"
    assert "workspace_artifact_not_auto_trusted" in assessment.integrity_signals


def test_operation_gate_denies_quarantined_memory_write() -> None:
    decision = evaluate_operation_gate(
        config={"runtime_integrity": {"enabled": True, "mode": "shadow"}},
        intent=OperationIntent(
            operation_type="memory_write",
            target="mem:1",
            source_artifact_ids=["art-1"],
            source_trust_states=["tainted_internal"],
        ),
        artifact_assessments=[
            {
                "artifact_id": "art-1",
                "shadow_verdict": "quarantine",
                "hard_invariant_hits": ["quarantined_source_artifact"],
            }
        ],
    )
    assert decision.status == "deny"
    assert decision.reason_code == "integrity_quarantined_source_memory_write"
    assert decision.shadow_only is True
    assert decision.would_enforce is False


def test_operation_gate_requires_budget_for_resource_heavy() -> None:
    decision = evaluate_operation_gate(
        config={"runtime_integrity": {"enabled": True, "mode": "shadow"}},
        intent=OperationIntent(
            operation_type="resource_heavy_action",
            target="heavy-job",
            source_artifact_ids=["art-2"],
            source_trust_states=["untrusted"],
            metadata={"resource_heavy": True, "budget_available": False},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "deny"
    assert decision.reason_code == "integrity_resource_budget_required"
    assert decision.would_enforce is False


def test_operation_gate_reviews_untrusted_tool_call_in_shadow() -> None:
    decision = evaluate_operation_gate(
        config={"runtime_integrity": {"enabled": True, "mode": "shadow"}},
        intent=OperationIntent(
            operation_type="tool_call",
            target="summarize",
            source_artifact_ids=["art-3"],
            source_trust_states=["untrusted"],
        ),
        artifact_assessments=[],
    )
    assert decision.status == "review"
    assert decision.shadow_only is True


def test_operation_gate_skillbox_source_mismatch_is_default_off_shadow() -> None:
    decision = evaluate_operation_gate(
        config={
            "runtime_integrity": {"enabled": True, "mode": "enforce"},
            "skillbox": {"enabled": True, "mode": "shadow"},
        },
        intent=OperationIntent(
            operation_type="tool_call",
            target="skill-runner",
            metadata={"skillbox_verification_status": "source_mismatch"},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "deny"
    assert decision.reason_code == "integrity_skillbox_source_mismatch"
    assert decision.shadow_only is True
    assert decision.would_enforce is False


def test_operation_gate_skillbox_source_mismatch_enforces_with_explicit_flag() -> None:
    decision = evaluate_operation_gate(
        config={
            "runtime_integrity": {"enabled": True, "mode": "enforce"},
            "skillbox": {
                "enabled": True,
                "mode": "shadow",
                "enforcement": {"source_mismatch": True},
            },
        },
        intent=OperationIntent(
            operation_type="tool_call",
            target="skill-runner",
            metadata={"skillbox_verification_status": "source_mismatch"},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "deny"
    assert decision.reason_code == "integrity_skillbox_source_mismatch"
    assert decision.shadow_only is False
    assert decision.would_enforce is True


def test_operation_gate_skillbox_source_mismatch_respects_runtime_shadow_mode() -> None:
    decision = evaluate_operation_gate(
        config={
            "runtime_integrity": {"enabled": True, "mode": "shadow"},
            "skillbox": {
                "enabled": True,
                "mode": "shadow",
                "enforcement": {"source_mismatch": True},
            },
        },
        intent=OperationIntent(
            operation_type="tool_call",
            target="skill-runner",
            metadata={"skillbox_verification_status": "source_mismatch"},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "deny"
    assert decision.shadow_only is True
    assert decision.would_enforce is False


def test_operation_gate_disabled_short_circuits() -> None:
    decision = evaluate_operation_gate(
        config={"runtime_integrity": {"enabled": False, "mode": "enforce"}},
        intent=OperationIntent(
            operation_type="resource_heavy_action",
            target="heavy-job",
            metadata={"resource_heavy": True, "budget_available": False},
        ),
        artifact_assessments=[],
    )
    assert decision.status == "disabled"
    assert decision.would_enforce is False


def test_runtime_artifact_trace_metadata_is_allowlisted() -> None:
    item = ContentItem(
        doc_id="doc-3",
        source_id="api:r1",
        source_type="other",
        trust="untrusted",
        text="hello",
        artifact_id="art-3",
        origin="api_request",
        content_hash="h3",
        meta={
            "request_id": "r1",
            "boundary_step": 1,
            "secret_blob": "must-not-leak",
        },
    )
    artifact = build_runtime_artifact(item)
    payload = artifact.to_dict()
    assert payload["metadata"]["request_id"] == "r1"
    assert "secret_blob" not in payload["metadata"]


def test_runtime_artifact_assessment_includes_named_skill_invocation_signal() -> None:
    item = ContentItem(
        doc_id="doc-4",
        source_id="user:skill",
        source_type="user",
        trust="trusted_user",
        text="I have debug-pro skill installed. Use it now.",
        artifact_id="art-4",
        origin="user",
        content_hash="h4",
    )
    artifact = build_runtime_artifact(item)
    assessment = assess_runtime_artifact(
        artifact,
        effect_shadow={
            "named_skill_invocation": {"detected": True},
            "skill_provenance_assessment": {"status": "requires_verification"},
        },
    )
    assert "named_skill_invocation_detected" in assessment.integrity_signals
    assert "skill_provenance_requires_verification" in assessment.integrity_signals
