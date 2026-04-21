# Custom Integration From Scratch (`OmegaAdapterRuntime`)

Use this path when your agent framework is not covered by the official adapters.

The contract stays the same:
- resolve `session_id` / `actor_id`
- run model/input pre-check
- run tool preflight check before execution (fail-closed)
- run memory-write pre-check with source/trust tagging

## 1) Install

```bash
pip install omega-walls
```

## 2) Minimal runtime wiring

```python
from omega.adapters import (
    AdapterSessionContext,
    OmegaAdapterRuntime,
    OmegaBlockedError,
    OmegaToolBlockedError,
)

runtime = OmegaAdapterRuntime(
    profile="quickstart",
    projector_mode="pi0",  # local no-key path
    max_chars=8000,
)


def resolve_ctx(payload: dict) -> AdapterSessionContext:
    session_id = (
        payload.get("thread_id")
        or payload.get("conversation_id")
        or payload.get("session_id")
        or "omega-custom-default"
    )
    actor_id = (
        payload.get("actor_id")
        or payload.get("user_id")
        or session_id
    )
    return AdapterSessionContext(
        session_id=session_id,
        actor_id=actor_id,
        metadata={"framework": "custom"},
    )


def guard_model_input(payload: dict) -> None:
    decision = runtime.check_model_input(
        messages_text=str(payload.get("text", "")),
        ctx=resolve_ctx(payload),
    )
    block_outcomes = {"OFF", "BLOCK", "TOOL_FREEZE", "ESCALATE"}
    if decision.off or str(decision.control_outcome).upper() in block_outcomes:
        raise OmegaBlockedError("Omega blocked model/input step", decision=decision)


def guard_tool_call(payload: dict, tool_name: str, tool_args: dict) -> None:
    gate = runtime.check_tool_call(
        tool_name=tool_name,
        tool_args=tool_args,
        ctx=resolve_ctx(payload),
    )
    if not gate.allowed:
        raise OmegaToolBlockedError("Omega blocked tool call", gate_decision=gate)


def guard_memory_write(payload: dict, memory_text: str, source_id: str, source_trust: str) -> bool:
    decision = runtime.check_memory_write(
        memory_text=memory_text,
        source_id=source_id,
        source_type="external_text",
        source_trust=source_trust,
        ctx=resolve_ctx(payload),
    )
    return bool(decision.allowed)
```

## 3) Integration pattern

1. Call `guard_model_input(...)` before your model/agent step.
2. Call `guard_tool_call(...)` before every tool execution.
3. Call `guard_memory_write(...)` before persisting summaries/facts/notes.
4. If exceptions are raised, stop the operation (fail-closed) and surface the reason from:
   - `exc.decision.control_outcome`, `exc.decision.reason_codes` for model/input block
   - `exc.gate_decision.reason`, `exc.gate_decision.tool_name` for tool block
   - on tool denies, handle argument-validation reasons explicitly:
     - `INVALID_TOOL_ARGS_SCHEMA`
     - `INVALID_TOOL_ARGS_SECURITY`
     - `INVALID_TOOL_ARGS_SHELLLIKE`
5. For memory writes, only persist when `check_memory_write(...).allowed is True`.

## 4) Monitor-first rollout

For dry-run tuning, set:

```yaml
runtime:
  guard_mode: monitor
```

In monitor mode, decisions are still computed, but effective action remains `ALLOW`.
Use:

```bash
omega-walls report --window 24h --format json
omega-walls explain --session <session_id> --format json
```

## 5) Validation checklist

- `session_id` is stable across turns of one conversation.
- `actor_id` is set (fallback to `session_id` if absent).
- Every tool call goes through `check_tool_call(...)` before execution.
- Every persisted memory fact goes through `check_memory_write(...)`.
- Memory record includes `source_id`, `source_type`, `source_trust`.
- On blocked tool decision, downstream tool execution does not happen.
- In strict smoke, `orphan_executions == 0` and `gateway_coverage >= 1.0`.
