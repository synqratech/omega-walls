# Integration Template

Use this template when adding a new framework integration package under `/integrations/<framework>/`.

## Required Files

- `README.md`
- `example.py`
- `requirements.txt`
- `test_integration.py`

## README Contract

Every integration README must include:

1. Supported versions (framework + Omega Walls).
2. Quickstart (10-15 lines).
3. Insertion points (input/context/model/tool/output).
4. Block and fallback behavior.
5. Validation commands and CI workflow references.

## CI Contract

- `test_integration.py` must be executable in CI.
- The integration package must be linked from `docs/framework_integrations_quickstart.md`.
- Local links in README must resolve.
