# Pilot Operations Runbook

This runbook is the day-2 operations baseline for pilot environments.

## 1. Monitoring Baseline

- Start in `monitor` mode and verify event flow before any enforcement rollout.
- Confirm report/explain outputs are available and actionable.
- Ensure alert channels are configured before moving to enforce.

## 2. False Positive Handling

- Triage with `report` and `explain` first.
- Tune policy thresholds and source/tool controls in config profiles.
- Re-run targeted smokes after each tuning change.

## 3. Quarantine and Escalation

- Use quarantine/approval outcomes as controlled pause points, not silent failures.
- Keep approval paths observable in operator channels and audit logs.
- Resolve or reject approvals with clear incident rationale.

## 4. Enforce Transition

- Switch to `enforce` only after monitor-phase verification and alert readiness.
- Roll out gradually (dev/stage first, then prod).
- Keep a quick rollback path through profile/config versioning.

## 5. Safe Bypass in Staging

- Bypass is staging-only and time-scoped.
- Keep audit evidence for bypass start/end and actor.
- Re-enable normal policy path immediately after diagnostic window.

## 6. Fallback and Continuity

- Configure provider fallback (`primary -> backup -> rule_only`) for quota/outage resilience.
- Verify degraded decisions are explicitly marked in telemetry/status outputs.
- Treat prolonged degraded mode as an incident and escalate.

## 7. Minimum Pilot Checklist

- Alerts configured and tested.
- Approval flow validated.
- At least one framework strict smoke green.
- Monitor-to-enforce change reviewed.
- Rollback command path tested once in staging.

## Related References

- [Quickstart](quickstart.md)
- [Configuration & Policy Tuning](config.md)
- [Monitoring & Alerts](monitoring_alerts.md)
- [Debugging Workflow Failures](debugging_workflow_failures.md)
- [Policy Tuning](policy_tuning.md)
- [Workflow Continuity](workflow_continuity.md)
