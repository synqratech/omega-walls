# Memory Hygiene Contract (Adapters)

This contract defines what can and cannot be persisted into agent memory when using Omega guard adapters.

## Why this exists

Memory is an attack surface. Untrusted or quarantined signals must not become long-lived agent state.

The adapter contract enforces:

- source-trust tagging on every memory write check,
- deny/quarantine decisions before persistence,
- session/actor-aware stateful evaluation.

## Public API

All framework guards expose `check_memory_write(...)` with the same policy semantics:

- `OmegaLangChainGuard`
- `OmegaLangGraphGuard`
- `OmegaLlamaIndexGuard`
- `OmegaHaystackGuard`
- `OmegaAutoGenGuard`
- `OmegaCrewAIGuard`

Each call delegates to `OmegaAdapterRuntime.check_memory_write(...)` and returns `MemoryWriteDecision`.

## Decision semantics

`MemoryWriteDecision.mode`:

- `allow`: safe to persist.
- `quarantine`: do not persist; source should be treated as isolated.
- `deny`: do not persist; blocking policy signal is active.

`MemoryWriteDecision.allowed` is `True` only for `allow`.

## Rules (v1)

1. **Hard deny**
   - Any blocking/off signal (`OFF`, `BLOCK`, `TOOL_FREEZE`, `ESCALATE`, `SOFT_BLOCK`, `HUMAN_ESCALATE`, `REQUIRE_APPROVAL`) results in `mode=deny`.
2. **Source quarantine**
   - `SOURCE_QUARANTINE` action or untrusted source with high-risk spike/multi-signal results in `mode=quarantine`.
3. **Allow**
   - Trusted source without blocking/quarantine signals results in `mode=allow`.

## Required tags

Every decision carries source metadata:

- `source_id`
- `source_type`
- `source_trust` (`trusted|mixed|untrusted`)

Use these tags when storing any memory record, audit row, or derived summary.

## Example

```python
from omega.integrations import OmegaLangChainGuard

guard = OmegaLangChainGuard(profile="quickstart", projector_mode="pi0")
decision = guard.check_memory_write(
    memory_text="User preference: weekly digest on Monday.",
    source_id="ticket-482",
    source_type="support_ticket",
    source_trust="trusted",
    state={"thread_id": "sess-123", "user_id": "actor-42"},
)
if decision.allowed:
    persist_memory(...)
else:
    skip_write(reason=decision.reason, mode=decision.mode)
```

## Validation

Contract tests:

- `tests/test_memory_hygiene_contract.py`

These tests verify runtime rules and cross-adapter session/actor + source-trust propagation.
