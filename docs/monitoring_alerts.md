# Monitoring & Alerts (Slack / Telegram)

This runbook describes the v1 notification and human-escalation flow.

## Scope

Supported surfaces in this step:
- API (`omega-walls-api`)
- `OmegaRAGHarness`

Supported channels:
- Slack (bot API + Block Kit actions)
- Telegram (Bot API + inline callbacks)

## Config

Notifications are configured under `notifications.*` (see `config/notifications.yml`).

Key defaults:
- `notifications.enabled=false` (opt-in)
- triggers: `BLOCK`, `SOFT_BLOCK`, `SOURCE_QUARANTINE`, `TOOL_FREEZE`, `HUMAN_ESCALATE`, `REQUIRE_APPROVAL`, `FALLBACK`
- throttling: `WARN=300s`, `BLOCK=60s`
- approval timeout: `900s` (auto-expire)

## Required ENV

Slack:
- `SLACK_BOT_TOKEN`
- `SLACK_ALERT_CHANNEL`
- `SLACK_SIGNING_SECRET`

Telegram:
- `TG_BOT_TOKEN`
- `TG_ADMIN_CHAT_ID`
- `TG_BOT_SECRET_TOKEN`

Internal approval resolve endpoint:
- `OMEGA_NOTIFICATION_HMAC_SECRET`

## API Endpoints

Callbacks:
- `POST /v1/notifications/callback/slack`
- `POST /v1/notifications/callback/telegram`

Approval lifecycle:
- `GET /v1/approvals/{approval_id}`
- `POST /v1/approvals/{approval_id}/resolve`

Manual resolve requires:
- `X-API-Key`
- internal HMAC headers:
  - `X-Internal-Signature`
  - `X-Internal-Timestamp`
  - `X-Internal-Nonce`

## Security Verification

Implemented checks:
- Slack signature verification (`X-Slack-Signature`, `X-Slack-Request-Timestamp`)
- Telegram secret token verification (`X-Telegram-Bot-Api-Secret-Token`)
- Internal callback resolve HMAC + nonce/timestamp anti-replay

## Payload Policy

Alerts use redacted summary payloads by default:
- `control_outcome`
- `reasons`
- `action_types`
- `trace_id`, `decision_id`
- `incident_artifact_id`
- tenant/session/actor references

Raw attachment text is not included.

## Runtime Behavior

- Notification delivery failures are fail-open with metrics/audit (`notifications_failed`, etc.).
- Tool path remains fail-closed by policy/tool gateway.
- When `HUMAN_ESCALATE` or `REQUIRE_APPROVAL` is active, an approval record is created with `approval_id`.
- Approval timeout expires pending requests (auto-deny by policy lifecycle).

## Startup Preflight & Adoption Messages

At startup, Omega can emit two optional message types:

1. `startup_preflight`: operator checklist with status (`OK|WARN|MISSING|DISABLED`) for guard mode, projector semantic readiness/fallback, tool mode, approval config, and channel readiness.
2. `startup_outreach`: short one-time onboarding message (GitHub/docs/LinkedIn links, optional commercial CTA).

Config keys:

```yaml
notifications:
  startup:
    preflight:
      enabled: true
      terminal: true
      channels: true
      once_per_process: true
    outreach:
      enabled: true
      terminal: true
      channels: true
      once_per_process: true
```

Notes:
- Startup flow is `warn + continue`: runtime does not stop on notification/preflight issues.
- `notifications.enabled=false` disables channel delivery, but terminal preflight can still be shown when enabled.
- Startup dedup is process-local (`once_per_process=true`).

## Quick Smoke (Local)

1. Enable notifications in your profile override:

```yaml
notifications:
  enabled: true
  approvals:
    backend: memory
```

2. Run API and hit a blocking case:

```bash
omega-walls-api --profile quickstart --host 127.0.0.1 --port 8080
```

3. Confirm response includes:
- `approval_required`
- `approval_id`
- `approval_status`

4. Resolve manually (ops fallback) via:
- `POST /v1/approvals/{approval_id}/resolve`

## Troubleshooting

Agent stopped responding or tool execution paused:
1. Check latest `approval_id` from runtime/API response.
2. Query `GET /v1/approvals/{approval_id}`.
3. Resolve via Slack/Telegram action or internal resolve endpoint.
