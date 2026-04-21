# Quickstart: Reliability-First in <10 Minutes

This quickstart is intentionally split into two phases.

- Phase 1: monitor-first validation (no disruptive blocking side effects)
- Phase 2: production hardening (alerts + approvals required)

## 1) Install

```bash
pip install omega-walls
```

If you use an agent framework:
- [Framework Integrations Quickstart](framework_integrations_quickstart.md)
- [OpenClaw Integration (P0)](openclaw_integration.md)
- [Custom Integration From Scratch](custom_integration_from_scratch.md)

## 2) Phase 1: monitor-first validation

Run local monitor smoke (no API key required):

```bash
python scripts/smoke_monitor_mode.py --profile dev --projector-mode pi0
```

Then inspect timeline and aggregated report:

```bash
omega-walls report --session monitor-smoke --events-path <events_path> --format json
omega-walls explain --session monitor-smoke --events-path <events_path> --format json
```

Expected:
- `status: ok`
- attack sample has `intended_action != ALLOW`
- attack sample has `actual_action == ALLOW` in monitor mode

Framework route map:
`install -> adapter wiring -> strict smoke -> alerts setup -> API run`

## 3) Phase 2: required production hardening

Before production usage, configure alerts and approvals:
- Slack or Telegram channel integration
- approval callbacks and lifecycle (`approval_id`, resolve endpoints)
- startup preflight + outreach toggles under `notifications.startup.*`

Runbook:
- [Monitoring & Alerts](monitoring_alerts.md)

This step is required to avoid silent workflow pauses and to make escalations observable.

## 4) Enforce transition

After Phase 1 + Phase 2 are complete, switch to enforce:

```yaml
runtime:
  guard_mode: enforce
```

Use continuity-aware routing:
- `ALLOW` -> continue
- `SOFT_BLOCK|SOURCE_QUARANTINE|TOOL_FREEZE|WARN` -> continue with degraded context
- `HUMAN_ESCALATE|REQUIRE_APPROVAL` -> pause high-risk action and resolve approval

## Verification Checklist

- benign sessions are mostly `ALLOW` in monitor mode
- expected risky samples produce non-allow intended outcomes
- `explain` timeline includes actionable reasons/fragments/downstream impact
- alerts/approvals are visible and resolvable by operators

## Troubleshooting

- `intended_action=BLOCK` with `actual_action=ALLOW` is normal in monitor mode.
- If monitor events are missing, verify:
  - `runtime.guard_mode: monitor`
  - `monitoring.enabled: true`
  - `monitoring.export.path` is writable.
- Deep triage:
  - [Debugging Workflow Failures](debugging_workflow_failures.md)
  - [Policy Tuning](policy_tuning.md)
  - [Workflow Continuity](workflow_continuity.md)
