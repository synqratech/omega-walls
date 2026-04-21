# Framework Integrations Quickstart (5-Minute Path)

This guide is the practical onboarding path for all official Omega adapters:

- `OmegaLangChainGuard`
- `OmegaLangGraphGuard`
- `OmegaLlamaIndexGuard`
- `OmegaHaystackGuard`
- `OmegaAutoGenGuard`
- `OmegaCrewAIGuard`

Goal: install, wire the guard in a few lines, and verify blocking behavior quickly.

For OpenClaw Plugin SDK integration (npm plugin package + guarded WebFetch), see:
- [OpenClaw Integration (P0)](openclaw_integration.md)
- [Real-Agent Validation Stand](real_agent_stand_runbook.md)
- [Framework Matrix Stand](framework_matrix_stand.md)

Prefer starting from a generated scaffold:

```bash
python scripts/init_secure_agent_template.py --framework langchain --out ./_tmp/omega-secure-agent-lc
```

Template runbook:
- [Secure Agent Template (FW-007)](secure_agent_template.md)

## 1. Prerequisites

- Python `3.13` recommended for local reproducibility.
- Install package and framework dependencies:

```bash
pip install "omega-walls[integrations]"
```

If you prefer selective installs, install `omega-walls` and only the framework package you use.

### 1.1 Adapter Matrix (Install + Smoke)

| Adapter | Class | Install | Smoke |
|---|---|---|---|
| LangChain | `OmegaLangChainGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_langchain_guard.py --strict` |
| LangGraph | `OmegaLangGraphGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_langgraph_guard.py --strict` |
| LlamaIndex | `OmegaLlamaIndexGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_llamaindex_guard.py --strict` |
| Haystack | `OmegaHaystackGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_haystack_guard.py --strict` |
| AutoGen | `OmegaAutoGenGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_autogen_guard.py --strict` |
| CrewAI | `OmegaCrewAIGuard` | `pip install "omega-walls[integrations]"` | `python scripts/smoke_crewai_guard.py --strict` |

### 1.2 Connector Coverage

- OpenClaw runtime connector (Plugin SDK + WebFetch): [OpenClaw Integration (P0)](openclaw_integration.md)
- Slack/Telegram alerting connectors (notifications + approvals): [Monitoring & Alerts Runbook](monitoring_alerts.md)

## 2. Common Contract (All Integrations)

All official adapters follow one runtime contract:

1. Session/actor resolution:
   - prefers configured getter callbacks
   - then tries common context keys (`thread_id`, `conversation_id`, `session_id`, etc.)
   - then falls back to framework default session id
2. Model/input check:
   - user/agent input is normalized to text
   - evaluated with stateful Omega runtime
3. Tool preflight check:
   - tool call is checked before execution
   - fail-closed behavior via typed exception
   - per-tool argument validation is enforced at the same chokepoint (`network_post`, `write_file`, and shell-like names)
4. Memory write check:
   - persistence candidate is checked with source/trust tags
   - decision mode is `allow|quarantine|deny`

Blocking semantics:

- model/input blocked -> `OmegaBlockedError`
- tool call blocked -> `OmegaToolBlockedError`

Memory persistence semantics:

- all guards expose `check_memory_write(...)`
- return type: `MemoryWriteDecision`
- always tag records with `source_id`, `source_type`, `source_trust`

If a tool call is denied due to malformed/unsafe arguments, gateway reason codes are:
- `INVALID_TOOL_ARGS_SCHEMA`
- `INVALID_TOOL_ARGS_SECURITY`
- `INVALID_TOOL_ARGS_SHELLLIKE`

Contract doc:
- [Memory Hygiene Contract](memory_hygiene_contract.md)

## 3. Exception Handling Pattern (Recommended)

```python
from omega.adapters import OmegaBlockedError, OmegaToolBlockedError

try:
    # invoke your framework call here
    ...
except OmegaBlockedError as exc:
    print("Blocked model/input step")
    print(exc.decision.control_outcome, exc.decision.reason_codes)
except OmegaToolBlockedError as exc:
    print("Blocked tool call")
    print(exc.gate_decision.tool_name, exc.gate_decision.reason)
```

## 4. LangChain (Middleware)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from langchain.agents import create_agent
from omega.integrations import OmegaLangChainGuard

guard = OmegaLangChainGuard(profile="quickstart")
agent = create_agent(model="openai:gpt-4.1-mini", tools=[...], middleware=guard.middleware())
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize this note"}]})
```

Verify quickly:

```bash
python scripts/smoke_langchain_guard.py --strict
```

## 5. LangGraph (Wrapper + Guard Node + Tools)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from omega.integrations import OmegaLangGraphGuard

guard = OmegaLangGraphGuard(profile="quickstart")
safe_graph = guard.wrap_graph(compiled_graph)
safe_tool = guard.wrap_tool("network_post", network_post_fn)
guard_node = guard.build_guard_node()  # optional StateGraph node helper
```

Verify quickly:

```bash
python scripts/smoke_langgraph_guard.py --strict
```

## 6. LlamaIndex (Query Engine + Tools)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from omega.integrations import OmegaLlamaIndexGuard

guard = OmegaLlamaIndexGuard(profile="quickstart")
query_engine = guard.wrap_query_engine(index.as_query_engine())
safe_tool = guard.wrap_tool("network_post", network_post_fn)

response = query_engine.query("Summarize this support note", thread_id="sess-123")
```

Verify quickly:

```bash
python scripts/smoke_llamaindex_guard.py --strict
```

## 7. Haystack (Pipeline Component + Tools)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from omega.integrations import OmegaHaystackGuard

guard = OmegaHaystackGuard(profile="quickstart")
pipeline = guard.wrap_pipeline(pipeline, component_name="omega_guard_component")
safe_tool = guard.wrap_tool("network_post", network_post_fn)
```

Verify quickly:

```bash
python scripts/smoke_haystack_guard.py --strict
```

## 8. AutoGen (Agent Wrapper + Tools)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from omega.integrations import OmegaAutoGenGuard

guard = OmegaAutoGenGuard(profile="quickstart")
wrapped_agent = guard.wrap_agent(agent)
safe_tool = guard.wrap_tool("network_post", network_post_fn)
```

Verify quickly:

```bash
python scripts/smoke_autogen_guard.py --strict
```

## 9. CrewAI (Execution Hooks + Tools)

Install:

```bash
pip install "omega-walls[integrations]"
```

Integration:

```python
from omega.integrations import OmegaCrewAIGuard

guard = OmegaCrewAIGuard(profile="quickstart")
safe_tool = guard.wrap_tool("network_post", network_post_fn)

with guard.install_global_hooks():
    result = crew.kickoff(inputs={"topic": "Summarize this support ticket"})
```

Verify quickly:

```bash
python scripts/smoke_crewai_guard.py --strict
```

## 10. Unified Release-Gate Smoke

Run all six integration smokes with one command:

```bash
python scripts/run_framework_smokes.py --strict
```

Expected `summary` invariants:

- `status = ok`
- `framework_count = 6`
- `total_failures = 0`
- `min_gateway_coverage >= 1.0`
- `total_orphans = 0`

For the full three-layer stand (contract + workflow + stress) use:

```bash
python scripts/run_framework_matrix_stand.py --layer all --profile dev --strict
```

## 11. What Goes In / What Comes Out

Input to adapters:

- framework-native messages/query text
- optional context ids in kwargs/state/metadata (`thread_id`, `conversation_id`, `session_id`, `actor_id`, ...)
- tool name + tool args for wrapped tool calls

Output on allow path:

- original framework response (adapter is transparent)

Output on block path:

- typed exception (`OmegaBlockedError` or `OmegaToolBlockedError`)
- structured decision payload for logging/audit:
  - `control_outcome`
  - `reason_codes`
  - `tool_name` / `reason`
- `gateway_coverage`, `orphan_executions`

## 12. Custom Integration (No Official Adapter)

If your framework is not listed above, implement the same contract directly with:
- `OmegaAdapterRuntime`
- `AdapterSessionContext`
- `check_model_input(...)`
- `check_tool_call(...)`

Runbook:
- [Custom Integration From Scratch](custom_integration_from_scratch.md)

## 13. Troubleshooting

If an integration import fails:

1. Check Python version:
   - `python --version`
2. Ensure dependencies are installed:
   - `pip install "omega-walls[integrations]"`
3. Run the integration-specific smoke script to isolate setup issues.
4. If needed, run unified smoke and inspect per-framework stdout/stderr files in the run directory.

## 14. Real Workflow Stand (LangChain + OpenClaw Local E2E)

Run the two-phase stand with one command:

```bash
python scripts/run_real_agent_stand.py --profile dev --strict
```

This validates:
- LangChain mini-agent 10-case contract suite
- OpenClaw plugin hooks + guarded WebFetch + local API bridge

Runbook:
- [Real-Agent Validation Stand](real_agent_stand_runbook.md)
