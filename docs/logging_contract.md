# Structured Logging Contract (FW-005 P0)

This document defines the canonical structured log schema used by Omega Walls for:
- monitor events
- enforce decisions
- API audit/error events
- SDK/harness runtime events

Scope in FW-005 P0:
- core surfaces only (`OmegaRAGHarness`, API scan runtime, SDK `analyze_text`, monitor collector mirror)
- legacy telemetry dict builders are preserved (no breaking migration)

Structured logs are emitted to `stdout` as JSON through `structlog`.
Monitor JSONL artifacts continue to exist in parallel (`monitoring.export.path`).

## 1) Canonical Event Model

Python model: `omega.log_contract.OmegaLogEvent`.

Required fields (always present):
- `ts` (ISO-8601 UTC timestamp string)
- `level` (`DEBUG|INFO|WARN|ERROR|CRITICAL`)
- `event` (event type id)
- `session_id` (non-empty string)
- `mode` (`monitor|enforce`)
- `engine_version` (package/app version string)
- `risk_score` (float in `[0.0, 1.0]`)
- `intended_action` (`ALLOW|BLOCK|QUARANTINE|ESCALATE`)
- `actual_action` (`ALLOW|BLOCK|QUARANTINE|ESCALATE`)
- `triggered_rules` (string array)
- `attribution` (array of attribution objects)

Optional fields:
- `fp_hint` (deterministic false-positive hint string)
- `error` (`code`, `message`, optional `details`)
- `intended_action_native`
- `actual_action_native`
- `risk_score_native`
- `trace_id`
- `decision_id`
- `surface`
- `input_type`
- `input_length`
- `source_type`
- `chunk_hash`

Attribution item model:
- `source` (non-empty string)
- `chunk_hash` (non-empty deterministic hash/id)
- `score_contrib` (float `>= 0`)

## 2) Hard Invariants

1. `risk_score` is always normalized into `[0,1]`.
2. `session_id` must be a non-empty string.
3. In `monitor` mode, `actual_action` is always `ALLOW`.
4. Action mapping is canonicalized even when native policy names differ.

## 3) Native -> Canonical Mapping

Action normalization (`omega.log_contract.canonical_action`):
- `HUMAN_ESCALATE|REQUIRE_APPROVAL -> ESCALATE`
- `SOFT_BLOCK|TOOL_FREEZE -> BLOCK`
- `SOURCE_QUARANTINE|WARN -> QUARANTINE`
- otherwise `ALLOW`

Precedence when multiple action types are present:
1. `ESCALATE`
2. `BLOCK`
3. `QUARANTINE`
4. `ALLOW`

Risk normalization (`normalize_api_risk_score`):
- API-native `0..100` is converted to `0..1`
- native value is preserved in `risk_score_native`

## 4) Privacy Guardrails

Structured logging sanitizer redacts sensitive payload keys by default, including:
- `raw_prompt`
- `full_context`
- `tool_args`
- `api_key`/`token`/`secret`/`password` and similar key names

PII-like string patterns (email/phone) are also redacted in logged values.

Recommended safe metadata in logs:
- `input_type`
- `input_length`
- `source_type`
- `chunk_hash`

## 5) Configuration

Enable structured logs in config:

```yaml
logging:
  structured:
    enabled: true
    level: INFO
    json_output: true
    validate: true
```

Runtime behavior:
- fail-open on logger errors (runtime is not interrupted)
- emitter health counters are tracked (`events_total`, `emit_failures`, `last_error`)

## 6) Example Event

```json
{
  "ts": "2026-04-17T10:12:34.567Z",
  "level": "WARN",
  "event": "risk_assessed",
  "session_id": "sess-42",
  "mode": "monitor",
  "engine_version": "0.1.2",
  "risk_score": 0.82,
  "intended_action": "BLOCK",
  "actual_action": "ALLOW",
  "triggered_rules": ["override_instructions", "tool_or_action_abuse"],
  "attribution": [
    {
      "source": "ticket_482.pdf",
      "chunk_hash": "a1b2c3...",
      "score_contrib": 0.65
    }
  ],
  "fp_hint": "Possible FP: transient context spike. Check benign policy-keywords."
}
```

## 7) Parsing Examples

With `jq`:

```bash
jq 'select(.event=="risk_assessed") | {ts, session_id, risk_score, intended_action, actual_action}' omega.log
```

With Python:

```python
import json

with open("omega.log", "r", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("event") == "risk_assessed":
            print(row["session_id"], row["risk_score"], row["intended_action"], row["actual_action"])
```

