"""Minimal repo-local .env loader without external dependency."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Optional

_LOADED_ENV_FILES: set[str] = set()
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_env_path() -> Path:
    return _repo_root() / ".env"


def load_repo_env_file(*, env_path: Optional[str | Path] = None, override: bool = False) -> Optional[Path]:
    path = Path(env_path) if env_path is not None else default_env_path()
    path = path.resolve()
    if not path.exists() or not path.is_file():
        return None
    cache_key = f"{path}|override={bool(override)}"
    if cache_key in _LOADED_ENV_FILES:
        return path

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = str(raw_line).strip()
        line = line.lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = str(key).strip()
        if not _ENV_KEY_RE.match(key):
            continue
        parsed = str(value).strip()
        if len(parsed) >= 2 and parsed[:1] == parsed[-1:] and parsed[:1] in {"'", '"'}:
            parsed = parsed[1:-1]
        if override or not str(os.environ.get(key, "")).strip():
            os.environ[key] = parsed

    _LOADED_ENV_FILES.add(cache_key)
    return path
