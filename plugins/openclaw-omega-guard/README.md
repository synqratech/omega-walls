# Omega OpenClaw Guard Plugin (P0)

OpenClaw Plugin SDK integration for Omega Walls:

- lifecycle hooks guard (`before_agent_reply`, `before_tool_call`, `message_sending`)
- tool preflight fail-closed behavior
- guarded `registerWebFetchProvider(...)` path
- strict bridge to local `omega-walls-api` (`X-API-Key` + HMAC + nonce/timestamp)

## Install

```bash
cd plugins/openclaw-omega-guard
npm install
```

## Build / Test

```bash
npm run typecheck
npm run test
npm run build
npm run smoke
```

Local e2e against running `omega-walls-api`:

```bash
export OMEGA_OPENCLAW_API_BASE_URL=http://127.0.0.1:8080
export OMEGA_OPENCLAW_API_KEY=...
export OMEGA_OPENCLAW_HMAC_SECRET=...
npm run smoke:local-api
```

## Required Plugin Config

```json
{
  "tenantId": "your-tenant",
  "omegaApi": {
    "baseUrl": "http://127.0.0.1:8080",
    "apiKey": "your-omega-api-key",
    "hmacSecret": "your-omega-hmac-secret",
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

## Behavior Mapping

- Omega BLOCK-like outcome -> `{ block: true }`
- Omega escalation / approval-required -> `{ requireApproval: true }`
- Omega allow -> no hook decision object

## Notes

- P0 scope includes `WebFetch` provider guard only (no `WebSearch` provider yet).
- Human escalation path is OpenClaw-native (`requireApproval` / `/approve`).
- Slack/Telegram transport remains on backend side via Omega notification subsystem.
