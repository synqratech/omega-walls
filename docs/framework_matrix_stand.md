# Framework Matrix Stand

This runbook describes the canonical three-layer validation stand for adapters and OpenClaw integration.

## One-command run

```bash
python scripts/run_framework_matrix_stand.py --layer all --profile dev --strict --artifacts-root artifacts
```

Layer-only runs:

```bash
python scripts/run_framework_matrix_stand.py --layer contract --strict
python scripts/run_framework_matrix_stand.py --layer workflow --strict
python scripts/run_framework_matrix_stand.py --layer stress --strict
```

## Artifacts

- `artifacts/framework_contract/<run_id>/report.json`
- `artifacts/real_workflow_matrix/<run_id>/<framework>.json`
- `artifacts/stress/<run_id>/chaos_report.json`

Aggregate summary is written to:

- `artifacts/framework_matrix_summary/<run_id>.json`

## Release gates (PASS/FAIL)

- `gateway_coverage >= 1.0`
- `orphan_executions == 0`
- `blocked_input_seen == true`
- `blocked_tool_seen == true`
- `require_approval_resume_success_rate == 1.0`
- `replay_block_rate == 1.0`
- `webfetch_edge_handling_rate == 1.0`

In `--strict` mode, any gate failure returns non-zero exit code.

## Layer semantics

### Contract layer (PR-blocking)

- Runs 6 framework guard smokes.
- Runs OpenClaw mapper contract subset (`npm run typecheck`, `npm run test`).
- Writes canonical report with target-level statuses and aggregate gates.

### Workflow layer (nightly)

- Runs matrix reports for each framework target.
- Runs OpenClaw local plugin smoke against local `omega-walls-api` with strict auth headers.
- Produces per-framework JSON plus OpenClaw JSON.

### Stress layer (nightly heavy)

- Runs race/lifecycle/webfetch edge suites.
- Produces normalized rates for resume, replay, and edge handling.

## Troubleshooting

- **API boot failure**: check `api.log` in workflow artifacts; verify profile and port availability.
- **HMAC mismatch**: verify `OMEGA_API_HMAC_SECRET` and plugin env values match.
- **Replay rejections**: ensure nonce/timestamp are unique and clock skew is in policy range.
- **Approval stuck pending**: inspect approval records and timeout handling in stress report.
- **WebFetch edge failure**: verify `tests/data/framework_matrix/webfetch_edge_corpus/*` fixtures are present and readable.
