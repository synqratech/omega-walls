# Release Surfaces: PyPI vs GitHub OSS

Use two different publication surfaces with different packaging rules:

1. **PyPI package surface** (`pip install omega-walls`)
2. **GitHub OSS source surface** (curated public repository tree)

This avoids manual copy/cleanup on every update.

## 1) PyPI surface (Python package only)

PyPI content is controlled by:

- `pyproject.toml` (`project.readme = README_PYPI.md`, package discovery under `omega*`)
- `MANIFEST.in` (explicit include/prune rules)

Validate package locally:

```bash
python -m build
python -m twine check dist/*
python scripts/smoke_package_install.py
```

## 2) GitHub OSS surface (curated source tree)

GitHub export is controlled by:

- `config/oss_export_github.json` (allowlist + exclude globs)
- `scripts/sync_public_github_repo.py` (export + sync + optional commit/push)

### Recommended flow

1. Clone/open your public GitHub repo locally (target path).
2. Run sync from this repo:

```bash
python scripts/sync_public_github_repo.py \
  --target-repo-dir "<PATH_TO_PUBLIC_REPO>" \
  --clean-staging \
  --delete-extra
```

Validate docs link contract against curated export before push:

```bash
python scripts/validate_oss_docs_contract.py
```

This does:

1. Exports curated files into staging (`artifacts/oss_export/github_public`)
2. Copies curated files into the target public repo
3. Optionally deletes files in target repo that are not part of curated export (`--delete-extra`)

### Optional commit/push from one command

```bash
python scripts/sync_public_github_repo.py \
  --target-repo-dir "<PATH_TO_PUBLIC_REPO>" \
  --clean-staging \
  --delete-extra \
  --git-commit \
  --commit-message "chore: sync OSS public tree" \
  --git-push
```

Optional flags:

- `--git-remote origin`
- `--git-branch main`

## What is intentionally excluded from GitHub OSS export

By default, curated export excludes internal/non-OSS layers:

- `internal_data/`, `enterprise/`, `redteam/`, `data/`, `notebooks/`, `cloud/`
- local env/cache/build artifacts (`.venv*`, `.vendor`, `artifacts/`, `build/`, `dist/`, etc.)
- local credentials/files such as `API_OpenAI.txt`

The exact source of truth is `config/oss_export_github.json`.

## Safety checks

After sync, verify in target repo before release:

```bash
git status --short
rg -n "OPENAI_API_KEY|sk-|API_OpenAI|internal_data|redteam" -S
```

The sync script also writes:

- `artifacts/oss_export/github_public/export_report.json`
- `artifacts/oss_export/github_public/sync_report.json`
