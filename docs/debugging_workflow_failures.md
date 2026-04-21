# Debugging Workflow Failures

## Problem

Your agent appears to "go silent", returns partial output, or stops tool execution.
You need to determine if Omega intentionally suppressed a risky path and where it happened.

## Config

Run in monitor mode while debugging so detection stays identical to enforce, but actions remain non-blocking:

```yaml
runtime:
  guard_mode: monitor

monitoring:
  enabled: true
  export:
    path: artifacts/monitor/monitor_events.jsonl
```

## CLI Output

Run the executable docs demo first (creates monitor events and prints timeline summary):

```bash
python examples/reliability_quickstart/explain_timeline_demo.py
```

Generate a compact report:

```bash
omega-walls report --session <session_id> --events-path artifacts/monitor/monitor_events.jsonl --format json
```

Generate per-step timeline:

```bash
omega-walls explain --session <session_id> --events-path artifacts/monitor/monitor_events.jsonl --format json
```

Optional:

```bash
omega-walls explain --session <session_id> --events-path artifacts/monitor/monitor_events.jsonl --window 24h --limit 200 --format csv
```

## Verification

Use these fields to isolate root cause:

- `summary.intended_outcomes_count`: which non-allow outcomes would have fired.
- `timeline[].rules.triggered_rules` and `timeline[].rules.reason_codes`: why it triggered.
- `timeline[].primary_fragment`: highest-contribution redacted fragment.
- `timeline[].downstream`: intended continuity impact (`context_prevented`, `tool_execution_prevented`, affected docs/sources/tools).
- `mttd`: first non-allow index/timestamp and time from session start.

Monitor invariant check:
- `intended_action != ALLOW` + `actual_action == ALLOW` is expected in monitor mode.

## Troubleshooting

Symptom -> likely cause -> next action:

- No events for session -> session id mismatch or monitor disabled -> rerun with stable session id and verify `monitoring.enabled`.
- Too many non-allow events on benign traffic -> threshold/policy too tight -> follow [Policy Tuning](policy_tuning.md).
- Frequent tool suppression -> tool policy too broad -> refine allowlist/freeze mapping and re-run monitor pass.
- Workflow stalls after escalation -> missing continuity route -> implement route from [Workflow Continuity](workflow_continuity.md).
