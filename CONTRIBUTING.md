# Contributing

Thanks for contributing to Omega Walls.

## Local Setup

1. Use Python 3.10+.
2. Install project dependencies:
```bash
pip install -e .[dev]
```

## Run Checks

```bash
python -m pytest
```

Optional local gates:
```bash
make check
```

## Pull Requests

- Keep changes focused and small.
- Update docs in the same PR if behavior/config/CLI changes.
- Add or update tests for user-visible behavior.
- Ensure CI is green before requesting review.

## Security

For security issues, follow [SECURITY.md](SECURITY.md) and do not use public issues.
