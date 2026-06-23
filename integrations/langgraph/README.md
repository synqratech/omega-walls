# LangGraph Integration (Repository Runtime Package)

Status: **stable**

This package is the executable repository surface for LangGraph integration.
It is intentionally minimal: install, wire, and verify.

## Supported Versions

- Omega Walls: `0.1.4`
- Framework pins: see [`requirements.txt`](requirements.txt)

## Quickstart

```bash
pip install -e ".[integrations]"
python scripts/smoke_langgraph_guard.py --strict
```

Minimal wiring:

```python
from omega.integrations import OmegaLangGraphGuard

guard = OmegaLangGraphGuard(profile="quickstart")
# framework-specific wiring in example.py
```

Expected result: smoke exits with code `0` and reports `status: ok`.

## Insertion Points

- Input ingestion boundary before context assembly.
- Context/memory carry-over boundary before model call.
- Model input boundary before generation.
- Tool execution boundary via ToolGateway before side effects.
- Output boundary with `security_metadata` on deny/degrade paths.

## Block and Fallback Behavior

- Block path: `OmegaBlockedError` / `OmegaToolBlockedError` with structured block contract.
- Fallback path: marked degradation metadata (`llm_fallback_active`, `fallback_level`) when configured.

## Validation

- Local executable test: [`test_integration.py`](test_integration.py)
- Shared contract runbook: [Framework Matrix Stand](../../docs/framework_matrix_stand.md)
- CI workflow: [framework-contract-gate](../../.github/workflows/framework-contract-gate.yml)
