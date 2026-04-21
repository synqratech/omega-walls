# Real-Agent Validation Stand Runbook

This stand validates real workflow wiring for:

1. LangChain guard path (in-process, fail-closed).
2. OpenClaw plugin path against local `omega-walls-api` with strict auth primitives (API key + HMAC).

## One-Command Run

```bash
python scripts/run_real_agent_stand.py --profile dev --strict
```

Artifacts are written to:

```text
artifacts/real_agent_stand/<run_id>/
```

Key files:

- `report.json` (unified gates + overall status)
- `phase1_langchain.json` (10-case LangChain suite)
- `phase2_openclaw.json` (OpenClaw phase command/results payload)
- `phase2_npm_*.stdout.txt` / `phase2_npm_*.stderr.txt` (plugin command logs)
- `api.log` (local API process log)

## What Is Checked

Mandatory gates in `report.json`:

- `blocked_input_seen`
- `blocked_tool_seen`
- `require_approval_seen`
- `webfetch_guard_seen`
- `outage_fail_closed_seen`
- `orphan_executions_zero`
- `gateway_coverage_ok`
- `session_reset_seen`

`overall_status`:

- `ok`: both phases passed and all gates true.
- `partial`: stand ran, but one or more gates failed (only when `--strict` is not set).
- `fail`: hard infra failure or strict gate failure.

## Strict Auth Notes

Phase 2 starts local API with:

- API key auth enabled (`dev-api-key`)
- HMAC signing enabled (`OMEGA_API_HMAC_SECRET` per run)

For local loopback reproducibility, HTTPS transport enforcement is disabled only for this stand process via env override. HMAC remains required.

## Troubleshooting

### API does not boot

- Inspect `api.log`.
- Check optional API deps:

```bash
pip install "omega-walls[api]"
```

### Plugin command fails (`npm run ...`)

- Check `phase2_*.stderr.txt`.
- Ensure Node 20+ and install plugin deps:

```bash
cd plugins/openclaw-omega-guard
npm ci --no-audit --no-fund
```

### Signature mismatch or 401

- Ensure phase 2 uses same generated HMAC secret for API and plugin env.
- Confirm `OMEGA_OPENCLAW_API_KEY=dev-api-key` in phase logs.

### Stand reports `partial`

- Open `report.json` and inspect `gates`.
- Use `phase1_langchain.json` case rows to see which category failed (`expected_category` vs `observed_category`).
