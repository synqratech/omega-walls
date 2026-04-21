# Workflow Continuity

## Problem

Security decisions should not look like random outages.
Users must get predictable behavior when Omega detects risk.

## Config

Use explicit guard mode:

```yaml
runtime:
  guard_mode: enforce
```

Keep monitor available for validation:

```yaml
monitoring:
  enabled: true
  export:
    path: artifacts/monitor/monitor_events.jsonl
```

## Continuity Routing Pattern

Map outcomes to runtime behavior:

- `ALLOW` -> continue normally
- `SOFT_BLOCK|SOURCE_QUARANTINE|TOOL_FREEZE|WARN` -> redact/degrade and continue
- `HUMAN_ESCALATE|REQUIRE_APPROVAL` -> pause risky branch and request human decision

Example policy-router sketch:

```python
def continuity_route(control_outcome: str) -> str:
    outcome = str(control_outcome).strip().upper()
    if outcome in {"HUMAN_ESCALATE", "REQUIRE_APPROVAL"}:
        return "ESCALATE"
    if outcome in {"SOFT_BLOCK", "SOURCE_QUARANTINE", "TOOL_FREEZE", "WARN"}:
        return "REDACT_AND_CONTINUE"
    return "ALLOW"
```

## CLI Output

Run the executable docs demo first (prints continuity routes):

```bash
python examples/reliability_quickstart/workflow_continuity_demo.py
```

Use monitor `downstream` fields to validate expected continuity effect:

```bash
omega-walls explain --session <session_id> --events-path artifacts/monitor/monitor_events.jsonl --format json
```

Inspect:
- `timeline[].downstream.context_prevented`
- `timeline[].downstream.tool_execution_prevented`
- `timeline[].downstream.blocked_doc_ids`
- `timeline[].downstream.quarantined_source_ids`

## Verification

Continuity behavior is healthy when:

- risky branch is degraded/escalated, not silently dropped
- user receives deterministic response path (`ALLOW`, `REDACT_AND_CONTINUE`, or `ESCALATE`)
- no orphan tool executions in blocked/degraded paths
- fallback behavior is documented and testable

## Troubleshooting

- Agent appears to freeze without explanation -> attach continuity outcome to user-facing response or ops notification.
- Tool calls still execute after non-allow outcome -> verify tool gateway is the single execution path.
- Excessive escalations -> tune escalation thresholds via [Policy Tuning](policy_tuning.md).
