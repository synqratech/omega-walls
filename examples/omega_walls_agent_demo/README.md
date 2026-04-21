# Omega Walls Agent Demo

This folder is a minimal "external project" style demo that shows where to place
`OmegaWalls` in an agent workflow.

## What it demonstrates

- guard on user input before planning;
- guard on external memory/RAG chunks before prompt assembly;
- guard on tool intent before tool execution.

## Run

From repository root (with package already installed in your chosen venv):

```powershell
cd .\examples\omega_walls_agent_demo
python .\agent_demo.py
```

Expected outcome:

- safe scenario is `allowed`;
- injected scenario is `blocked`;
- script exits with code `0`.

## HTTP API Demo

This scenario validates the API server end-to-end:

```powershell
cd .\examples\omega_walls_agent_demo
python .\api_http_demo.py
```

What it does:

- starts `omega-walls-api` (`quickstart` profile) on localhost;
- sends a safe request and expects `verdict=allow`;
- sends an injected request and expects non-allow verdict;
- stops server automatically.
