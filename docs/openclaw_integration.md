# OpenClaw Integration (P0): Plugin SDK + WebFetch + Omega Guard

This guide shows how to wire Omega Walls into OpenClaw via a plugin package:

- hook-based guard on message/tool lifecycle
- tool preflight fail-closed behavior
- guarded web fetch provider
- strict local bridge to `omega-walls-api`

## 1) What gets installed

Plugin package in this repository:

- `plugins/openclaw-omega-guard/`

It includes:

- `openclaw.plugin.json`
- `src/index.ts` (`definePluginEntry`)
- hook wiring (`before_agent_reply`, `before_tool_call`, `message_sending`)
- `registerWebFetchProvider(...)`

## 2) Prerequisites

- Node.js 20+ for plugin build/runtime.
- Python 3.13 recommended for local Omega API runs.
- Install API runtime extras once:

```bash
pip install "omega-walls[api]"
```

## 3) Start omega-walls-api (strict auth)

Run API in a strict profile (HMAC enabled by default in `dev` profile):

```bash
export OMEGA_API_HMAC_SECRET="change-me"
omega-walls-api --profile dev --host 127.0.0.1 --port 8080
```

PowerShell:

```powershell
$env:OMEGA_API_HMAC_SECRET="change-me"
omega-walls-api --profile dev --host 127.0.0.1 --port 8080
```

The plugin bridge requires:

- API key auth enabled
- HMAC verification enabled (`X-Signature`, `X-Timestamp`, `X-Nonce`)

The plugin assumes strict mode by default.

## 4) Configure the plugin

Use plugin config like:

```json
{
  "tenantId": "acme-prod",
  "omegaApi": {
    "baseUrl": "http://127.0.0.1:8080",
    "apiKey": "YOUR_API_KEY",
    "hmacSecret": "YOUR_HMAC_SECRET",
    "timeoutMs": 8000
  },
  "omega": {
    "guard": { "runtimeMode": "stateful" },
    "failMode": "fail_closed"
  },
  "alerts": {
    "enabled": true
  }
}
```

## 5) Hook decision contract

The plugin maps Omega decisions into OpenClaw hook decisions:

- block-like outcome -> `{ block: true, reason?: string }`
- escalation/approval outcome -> `{ requireApproval: true, reason?: string }`
- allow -> no decision object

Human escalation uses OpenClaw-native approval flow (`requireApproval` / `/approve`).

## 6) WebFetch behavior

`registerWebFetchProvider(...)` applies guard checks to fetched text before returning it:

- clean -> returns content
- quarantine/block -> returns redacted/blocked payload
- escalation -> returns `requireApproval` signal

If upstream fetch fails, provider returns controlled error (no runtime crash).

## 7) Smoke / validation

From plugin directory:

```bash
npm install
npm run typecheck
npm run test
npm run build
npm run smoke
```

Expected smoke result: JSON with `status: "ok"` and at least one blocked sample decision.

Real local e2e (requires running `omega-walls-api`):

```bash
export OMEGA_OPENCLAW_API_BASE_URL=http://127.0.0.1:8080
export OMEGA_OPENCLAW_API_KEY=...
export OMEGA_OPENCLAW_HMAC_SECRET=...
npm run smoke:local-api
```

Quick API sanity check before running plugin smoke:

```bash
curl -fsS http://127.0.0.1:8080/healthz
```

## 8) Troubleshooting

- `invalid_signature` / `stale_timestamp` / `replay_detected`
  - verify API key, HMAC secret, server time sync, and nonce uniqueness.
- frequent `omega_guard_unavailable`
  - check `omegaApi.baseUrl`, API health, network path, timeout value.
- repeated approval stalls
  - inspect OpenClaw approval queue and Omega `/v1/approvals/*` backend status.
- unexpected allow on guard failure
  - ensure `omega.failMode` is `fail_closed` (default).
