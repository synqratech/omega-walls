from __future__ import annotations

from omega.interfaces.contracts_v1 import ProjectionEvidence, ProjectionResult
from scripts.diagnose_run_eval_misses import SampleOutcome, build_diagnostic_report, classify_sample_outcome


def _proj(v, raw, polarity, tier_map):
    return ProjectionResult(
        doc_id="s1",
        v=v,
        evidence=ProjectionEvidence(
            polarity=polarity,
            debug_scores_raw=raw,
            matches={"pi0_rule_tier": tier_map},
        ),
    )


def test_classify_sample_outcome_marks_wrong_wall_and_polarity_block():
    sample = {
        "id": "s1",
        "family": "xsrc_policy_tool",
        "targets": ["policy_evasion", "tool_or_action_abuse"],
        "text": "policy exception path with vendor transfer",
    }
    proj = _proj(
        v=[1.2, 0.0, 0.0, 0.0],
        raw=[1.2, 0.0, 0.45, 0.61],
        polarity=[1, 0, 0, 0],
        tier_map={
            "tool_or_action_abuse": {"tier": "soft", "soft_signals": ["semantic_boost"]},
            "policy_evasion": {"tier": "soft", "soft_signals": ["semantic_boost"]},
        },
    )
    outcome = classify_sample_outcome(sample, proj)
    assert outcome.primary_class == "wrong_wall_zero_target"
    assert "misrouting_non_target_hit" in outcome.flags
    assert "raw_positive_final_zero" in outcome.flags
    assert "polarity_block" in outcome.flags
    assert "semantic_boost_only" in outcome.flags


def test_classify_sample_outcome_marks_partial_hit():
    sample = {
        "id": "s2",
        "family": "xsrc_secret_policy",
        "targets": ["secret_exfiltration", "policy_evasion"],
        "text": "share token under exception route",
    }
    proj = _proj(
        v=[0.0, 2.1, 0.0, 0.0],
        raw=[0.0, 2.1, 0.0, 0.51],
        polarity=[0, 1, 0, 0],
        tier_map={
            "secret_exfiltration": {"tier": "hard_live", "soft_signals": []},
            "policy_evasion": {"tier": "soft", "soft_signals": ["semantic_boost"]},
        },
    )
    outcome = classify_sample_outcome(sample, proj)
    assert outcome.primary_class == "partial_target_hit"
    assert outcome.detected_targets == ["secret_exfiltration"]
    assert outcome.missed_targets == ["policy_evasion"]


def test_build_diagnostic_report_groups_by_family_and_wall():
    outcome_a = SampleOutcome(
        sample_id="a",
        family="xsrc_policy_tool",
        targets=["policy_evasion", "tool_or_action_abuse"],
        detected_targets=[],
        non_target_hits=["override_instructions"],
        missed_targets=["policy_evasion", "tool_or_action_abuse"],
        primary_class="wrong_wall_zero_target",
        flags=["misrouting_non_target_hit", "polarity_block"],
        raw_scores={"override_instructions": 1.1, "secret_exfiltration": 0.0, "tool_or_action_abuse": 0.4, "policy_evasion": 0.5},
        polarity={"override_instructions": 1, "secret_exfiltration": 0, "tool_or_action_abuse": 0, "policy_evasion": 0},
        tiers={"override_instructions": "hard_live", "secret_exfiltration": "none", "tool_or_action_abuse": "soft", "policy_evasion": "soft"},
        soft_signals={"override_instructions": [], "secret_exfiltration": [], "tool_or_action_abuse": ["semantic_boost"], "policy_evasion": ["semantic_boost"]},
        text="Legacy routing artifact references 'activate exception route'.",
    )
    outcome_b = SampleOutcome(
        sample_id="b",
        family="xsrc_secret_policy",
        targets=["secret_exfiltration", "policy_evasion"],
        detected_targets=["secret_exfiltration"],
        non_target_hits=[],
        missed_targets=["policy_evasion"],
        primary_class="partial_target_hit",
        flags=["partial_polarity_block"],
        raw_scores={"override_instructions": 0.0, "secret_exfiltration": 1.4, "tool_or_action_abuse": 0.0, "policy_evasion": 0.3},
        polarity={"override_instructions": 0, "secret_exfiltration": 1, "tool_or_action_abuse": 0, "policy_evasion": 0},
        tiers={"override_instructions": "none", "secret_exfiltration": "hard_live", "tool_or_action_abuse": "none", "policy_evasion": "soft"},
        soft_signals={"override_instructions": [], "secret_exfiltration": [], "tool_or_action_abuse": [], "policy_evasion": ["semantic_boost"]},
        text="Support note says 'policy exception path' after token review.",
    )

    report = build_diagnostic_report(
        [outcome_a, outcome_b],
        families=["xsrc_policy_tool", "xsrc_secret_policy"],
        top_n=4,
        data_source="unit",
    )
    assert report["overall"]["miss_count"] == 2
    assert report["families_report"]["xsrc_policy_tool"]["primary_class_counts"]["wrong_wall_zero_target"] == 1
    assert report["families_report"]["xsrc_secret_policy"]["primary_class_counts"]["partial_target_hit"] == 1
    phrases = report["families_report"]["xsrc_policy_tool"]["top_phrase_hints"]
    assert any(item["phrase"] == "activate exception route" for item in phrases)
    wall_rows = report["wall_miss_table"]
    policy_tool_row = next(
        row for row in wall_rows
        if row["family"] == "xsrc_policy_tool" and row["missed_wall"] == "policy_evasion"
    )
    assert policy_tool_row["count"] == 1
    assert policy_tool_row["raw_positive_count"] == 1
    assert policy_tool_row["polarity_zero_count"] == 1
    assert policy_tool_row["misrouted_count"] == 1
    assert any(item["phrase"] == "activate exception route" for item in policy_tool_row["top_phrase_clusters"])
    phrase_rows = report["phrase_wall_priority_table"]
    phrase_row = next(
        row for row in phrase_rows
        if row["family"] == "xsrc_policy_tool"
        and row["missed_wall"] == "policy_evasion"
        and row["phrase_cluster"] == "activate exception route"
    )
    assert phrase_row["count"] == 1
    assert phrase_row["raw_positive_count"] == 1
    assert phrase_row["polarity_zero_count"] == 1
    assert phrase_row["raw_positive_polarity_zero_count"] == 1
    assert phrase_row["misrouted_count"] == 1
