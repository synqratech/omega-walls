# Framework Integrations Quickstart (Repository Source of Truth)

This is the canonical repository entry point for framework integrations.
The repo owns the executable integration surface; deeper operational guidance can be layered externally.

## 1. Integration Matrix (Status + Executable Packages)

| Integration | Runtime Guard | Status | Repository Package |
|---|---|---|---|
| LangChain | `OmegaLangChainGuard` | `stable` | [`/integrations/langchain`](../integrations/langchain/README.md) |
| LangGraph | `OmegaLangGraphGuard` | `stable` | [`/integrations/langgraph`](../integrations/langgraph/README.md) |
| LlamaIndex | `OmegaLlamaIndexGuard` | `stable` | [`/integrations/llamaindex`](../integrations/llamaindex/README.md) |
| Haystack | `OmegaHaystackGuard` | `stable` | [`/integrations/haystack`](../integrations/haystack/README.md) |
| AutoGen | `OmegaAutoGenGuard` | `stable` | [`/integrations/autogen`](../integrations/autogen/README.md) |
| CrewAI | `OmegaCrewAIGuard` | `stable` | [`/integrations/crewai`](../integrations/crewai/README.md) |
| OpenClaw (Plugin SDK + WebFetch) | plugin bridge | `beta` | [`/integrations/openclaw`](../integrations/openclaw/README.md) |

Related runbooks:

- [OpenClaw Integration](openclaw_integration.md)
- [Framework Matrix Stand](framework_matrix_stand.md)
- [Real-Agent Validation Stand](real_agent_stand_runbook.md)
- [Custom Integration From Scratch](custom_integration_from_scratch.md)

## 2. One-Time Install and Fast Verification

```bash
pip install "omega-walls[integrations]"
python scripts/run_framework_smokes.py --strict
```

Expected summary invariants:

- `status = ok`
- `framework_count = 6`
- `min_gateway_coverage >= 1.0`
- `total_orphans = 0`

For full contract/workflow/stress validation:

```bash
python scripts/run_framework_matrix_stand.py --layer all --profile dev --strict
```

## 3. Canonical Contract Terms

This section is the shared term glossary for all integration packages.

- `block contract`: structured deny payload fields (`action`, `reason`, `trace_id`, `decision_id`, optional `policy_id`, `incident_id`, `fallback_hint`).
- `security_metadata`: side-channel output metadata with `mode`, `risk`, `action`, `trace_id`, `decision_id`, and fallback markers.
- `session_id mapping`: deterministic binding from framework-native run/thread/conversation IDs into Omega runtime context.
- `llm_fallback_active`: explicit marker that semantic checks are degraded and fallback routing is active.

Exception path remains backward-compatible:

- `OmegaBlockedError` for model/input blocking.
- `OmegaToolBlockedError` for tool-call blocking.

## 4. Minimal Adapter Wiring Pattern

```python
from omega.adapters import OmegaBlockedError, OmegaToolBlockedError
from omega.integrations import OmegaLangChainGuard

guard = OmegaLangChainGuard(profile="quickstart")

try:
    # invoke framework runtime here
    ...
except OmegaBlockedError as exc:
    print(exc.to_contract_payload())
except OmegaToolBlockedError as exc:
    print(exc.to_contract_payload())
```

Use the same pattern in all official adapters and in custom wrappers.

## 5. What Is Executable in the Repo

For each integration package under `/integrations/<framework>/`:

- `README.md`: concise install/wire/verify contract.
- `example.py`: minimal wiring snippet.
- `requirements.txt`: pinned dependency set.
- `test_integration.py`: executable smoke launcher for CI.

Template for new frameworks:

- [`/integrations/_template`](../integrations/_template/README.md)

## 6. Troubleshooting (Fast Path)

1. Verify Python and package install:
   - `python --version`
   - `pip install "omega-walls[integrations]"`
2. Run framework-specific smoke from the integration package README.
3. Run matrix stand contract layer:
   - `python scripts/run_framework_matrix_stand.py --layer contract --profile dev --strict`
4. Inspect generated artifacts under `artifacts/` for exact failure causes.
