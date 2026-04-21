# Policy Tuning (Monitor -> Enforce)

## Problem

You need enforcement confidence without introducing avoidable workflow disruption.
Tuning should be data-driven, using monitor events from your real traffic.

## Config

Start in monitor:

```yaml
runtime:
  guard_mode: monitor

monitoring:
  enabled: true
  export:
    path: artifacts/monitor/monitor_events.jsonl
```

Core knobs to tune:

```yaml
monitoring:
  false_positive_hints:
    low_confidence_near_threshold:
      min_risk: 0.65
      max_risk: 0.82
```

Policy controls (example):

```yaml
off_policy:
  tool_freeze:
    enabled: true
    horizon_steps: 8
  source_quarantine:
    enabled: true
```

Soft hallucination guard (API policy mapper, opt-in):

```yaml
api:
  policy_mapper:
    hallucination_guard_lite:
      enabled: true
      apply_when_source_trust: [untrusted, mixed]
      low_confidence_lte: 0.35
      only_if_intended_allow: true
```

When active, low-confidence untrusted/mixed inputs produce soft `WARN` with `response_constraints`
(disclaimer + citations required) instead of hard blocking.

## CLI Output

Run the executable docs demo first (monitor samples + aggregate report):

```bash
python examples/reliability_quickstart/monitor_quickstart_demo.py
```

Aggregate view:

```bash
omega-walls report --window 24h --events-path artifacts/monitor/monitor_events.jsonl --format json
```

Session-level triage:

```bash
omega-walls explain --session <session_id> --events-path artifacts/monitor/monitor_events.jsonl --format json
```

## Verification

Use this checklist before switching to enforce:

- benign sessions are predominantly `intended_action=ALLOW`
- non-allow outcomes align with known risky cases
- explain timeline identifies consistent high-signal rules/fragments (not random spikes)
- false-positive hints are actionable and shrinking over iterations

## Symptom -> Action Table

| Symptom | Likely Cause | Tuning Action |
|---|---|---|
| Benign traffic frequently gets `SOFT_BLOCK` | Threshold too low or over-broad rule trigger | Raise threshold / narrow rule scope / add trusted-source allowances |
| `TOOL_FREEZE` triggers too often on normal automation | Tool policy too strict | Reduce freeze horizon or narrow tool-abuse triggers |
| `HUMAN_ESCALATE` appears late in session | Stateful threshold too conservative | Lower escalation thresholds for high-confidence patterns |
| Spiky non-repeatable triggers | Transient context keywords | Use false-positive hints and tune low-confidence guardrails |

## Controlled Enforce Rollout

Switch in stages:
1. keep monitor in staging
2. enforce in a limited slice (selected workflows or tenants)
3. compare outcome drift with monitor baseline
4. expand when continuity and false-positive levels are acceptable

```yaml
runtime:
  guard_mode: enforce
```

### Interpreting `response_constraints`

If API response includes:
- `response_constraints.enabled=true`
- `response_constraints.disclaimer_required=true`
- `response_constraints.citation_required=true`

then the integrating app should:
1. show uncertainty disclaimer,
2. include citations from `citation_candidates[]`,
3. avoid definitive claims without source references.

DevOps-focused baseline profile (deny destructive/exfil shell patterns + approval for force push/deploy):
- [DevOps Minimal Policy Pack](policy_pack_devops_minimal.md)

## Troubleshooting

- If enforce starts causing unexplained pauses, keep enforce enabled but add continuity routes (see [Workflow Continuity](workflow_continuity.md)).
- If analysis is insufficient, return to monitor for one cycle and inspect `explain` timelines.
