"""Environment expansion and file-backed secret helpers.

The deployment layer intentionally supports the conventional ``NAME_FILE`` form
used by Docker/Compose/Kubernetes secrets.  Callers must never log returned
values.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Optional

import yaml

_ENV_REF_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def parse_env_override(value: str) -> Any:
    """Parse an environment override without surprising plain-string coercion."""
    raw = str(value)
    stripped = raw.strip()
    lower = stripped.lower()
    if lower in {"true", "false", "null", "none"}:
        return yaml.safe_load(lower if lower != "none" else "null")
    if stripped.startswith(("[", "{", '"', "'")):
        parsed = yaml.safe_load(stripped)
        return parsed
    if re.fullmatch(r"[-+]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    return raw


def _expand_string(value: str, env: Mapping[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if not name.startswith("OMEGA_"):
            raise ValueError(f"only OMEGA_* environment references are allowed in config: {name}")
        resolved = env.get(name)
        if resolved is not None and str(resolved) != "":
            return str(resolved)
        if default is not None:
            return str(default)
        raise ValueError(f"required environment variable is not set: {name}")

    return _ENV_REF_RE.sub(repl, value)


def expand_omega_environment(value: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Recursively expand ``${OMEGA_NAME}`` and ``${OMEGA_NAME:-default}``."""
    source = env or os.environ
    if isinstance(value, str):
        return _expand_string(value, source)
    if isinstance(value, list):
        return [expand_omega_environment(item, source) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_omega_environment(item, source) for item in value)
    if isinstance(value, dict):
        return {key: expand_omega_environment(item, source) for key, item in value.items()}
    return value


def read_secret_value(
    env_name: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    required: bool = False,
    max_bytes: int = 65_536,
) -> str:
    """Read a secret from ``ENV`` or ``ENV_FILE`` with ambiguity rejection."""
    source = env or os.environ
    name = str(env_name).strip()
    if not name:
        raise ValueError("secret environment name must be non-empty")
    direct = str(source.get(name, ""))
    file_name = f"{name}_FILE"
    file_value = str(source.get(file_name, "")).strip()
    if direct and file_value:
        raise ValueError(f"set only one of {name} or {file_name}")
    if file_value:
        path = Path(file_value).expanduser()
        if not path.exists():
            raise ValueError(f"secret file does not exist: {file_name}")
        if not path.is_file():
            raise ValueError(f"secret path is not a regular file: {file_name}")
        if path.stat().st_size > int(max_bytes):
            raise ValueError(f"secret file exceeds {max_bytes} bytes: {file_name}")
        try:
            direct = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"secret file must be UTF-8 text: {file_name}") from exc
    value = direct.strip()
    if required and not value:
        raise ValueError(f"required secret is missing: {name} or {file_name}")
    return value


def read_secret_list(env_name: str, *, env: Optional[Mapping[str, str]] = None, required: bool = False) -> list[str]:
    raw = read_secret_value(env_name, env=env, required=required)
    values: list[str] = []
    for line in raw.splitlines():
        for item in line.split(","):
            normalized = item.strip()
            if normalized:
                values.append(normalized)
    return list(dict.fromkeys(values))
