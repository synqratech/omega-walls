# OpenClaw Integration (Repository Runtime Package)

Status: **beta**

This package describes the executable repository contract for OpenClaw via Plugin SDK + Omega Guard bridge.

## Supported Versions

- Omega Walls API runtime: `0.1.4`
- Node.js: `20.x`
- Plugin implementation path: `plugins/openclaw-omega-guard`

## Quickstart

```bash
pip install -e ".[api,integrations]"
cd plugins/openclaw-omega-guard
npm ci --no-audit --no-fund
npm run smoke
```

Local API smoke (strict path):

```bash
python scripts/run_framework_matrix_stand.py --layer workflow --profile dev --strict
```

Expected result: plugin smoke exits `0` and emits blocked/approval mapping samples.

## Insertion Points

- `before_agent_reply` hook before model output release.
- `before_tool_call` hook before tool execution side effects.
- `message_sending` hook for outbound response metadata.
- Guarded WebFetch provider boundary before fetched content returns to agent loop.
- API bridge boundary for decision and approval routing.

## Block and Fallback Behavior

- Block path maps to OpenClaw decision payload with `action/controlOutcome`, `incidentArtifactId`, `policyId`, and `fallbackHint` when available.
- Approval path maps to `requireApproval` and keeps OpenClaw-native approval flow semantics.

## Validation

- Local executable test: [`test_integration.py`](test_integration.py)
- Operational guide: [OpenClaw Integration Guide](../../docs/openclaw_integration.md)
- CI workflows: [openclaw-plugin-ci](../../.github/workflows/openclaw-plugin-ci.yml), [framework-contract-gate](../../.github/workflows/framework-contract-gate.yml)
