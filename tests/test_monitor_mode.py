from __future__ import annotations

from pathlib import Path

from omega.monitoring.collector import build_monitor_collector_from_config
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.monitoring.models import MonitorEvent
from omega.monitoring.report import build_monitor_report


def test_guard_mode_runtime_flag_takes_precedence() -> None:
    cfg = {
        "runtime": {"guard_mode": "monitor"},
        "off_policy": {"enforcement_mode": "ENFORCE"},
        "tools": {"execution_mode": "ENFORCE"},
    }
    assert resolve_guard_mode(cfg) == GuardMode.MONITOR


def test_guard_mode_legacy_log_only_and_dry_run_maps_to_monitor() -> None:
    cfg = {
        "off_policy": {"enforcement_mode": "LOG_ONLY"},
        "tools": {"execution_mode": "DRY_RUN"},
    }
    assert resolve_guard_mode(cfg) == GuardMode.MONITOR


def test_monitor_collector_and_report(tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    cfg = {
        "monitoring": {
            "enabled": True,
            "export": {
                "path": str(events_path),
                "rotation": "none",
                "rotation_size_mb": 10,
            },
        }
    }
    collector = build_monitor_collector_from_config(config=cfg)
    collector.emit(
        MonitorEvent(
            ts="2026-04-16T10:00:00Z",
            surface="sdk",
            session_id="s-1",
            actor_id="a-1",
            mode="monitor",
            risk_score=0.81,
            intended_action="SOFT_BLOCK",
            actual_action="ALLOW",
            triggered_rules=["override_instructions"],
            attribution=[],
            reason_codes=["reason_spike"],
            trace_id="trc_1",
            decision_id="dec_1",
        )
    )
    collector.emit(
        MonitorEvent(
            ts="2026-04-16T10:01:00Z",
            surface="sdk",
            session_id="s-1",
            actor_id="a-1",
            mode="monitor",
            risk_score=0.25,
            intended_action="ALLOW",
            actual_action="ALLOW",
            triggered_rules=[],
            attribution=[],
            reason_codes=[],
            trace_id="trc_2",
            decision_id="dec_2",
            false_positive_hint="Possible FP: transient context spike without sustained multi-wall pressure. Check benign context for policy-like keywords.",
        )
    )
    report = build_monitor_report(events_path=events_path)
    assert report["total_checks"] == 2
    assert report["would_block"] == 1
    assert report["risk_distribution"]["0.7-1.0"] == 1
    assert report["risk_distribution"]["0.0-0.3"] == 1


def test_false_positive_hints_are_deterministic() -> None:
    cfg = {
        "monitoring": {
            "false_positive_hints": {
                "low_confidence_near_threshold": {
                    "min_risk": 0.65,
                    "max_risk": 0.82,
                    "max_triggered_rules": 2,
                    "allowed_reason_codes": ["reason_spike"],
                },
                "trusted_source_mismatch": {"trusted_levels": ["trusted"]},
                "transient_spike": {"spike_only_reason_codes": ["reason_spike"]},
            }
        }
    }

    low_conf = infer_false_positive_hint(
        risk_score=0.7,
        intended_action="SOFT_BLOCK",
        reason_codes=["reason_spike"],
        triggered_rules=["override_instructions"],
        attribution=[],
        config=cfg,
    )
    trusted = infer_false_positive_hint(
        risk_score=0.9,
        intended_action="HUMAN_ESCALATE",
        reason_codes=["reason_wall"],
        triggered_rules=["secret_exfiltration"],
        attribution=[{"trust": "trusted"}],
        config=cfg,
    )
    transient = infer_false_positive_hint(
        risk_score=0.91,
        intended_action="REQUIRE_APPROVAL",
        reason_codes=["reason_spike"],
        triggered_rules=["policy_evasion", "tool_or_action_abuse", "secret_exfiltration"],
        attribution=[],
        config=cfg,
    )
    assert isinstance(low_conf, str) and "near-threshold" in low_conf
    assert isinstance(trusted, str) and "trusted source" in trusted
    assert isinstance(transient, str) and "transient context spike" in transient
