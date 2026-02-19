"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfigSnapshot:
    resolved: Dict[str, Any]
    file_hashes: Dict[str, str]
    resolved_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_bytes()
    parsed = yaml.safe_load(content) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return parsed


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict[str, Any], env: Dict[str, str], prefix: str = "OMEGA__") -> Dict[str, Any]:
    """Apply env vars like OMEGA__OMEGA__EPSILON=0.2 to nested keys."""
    updated = dict(config)
    for key, value in env.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        cur: Dict[str, Any] = updated
        for part in path[:-1]:
            next_val = cur.get(part)
            if not isinstance(next_val, dict):
                next_val = {}
                cur[part] = next_val
            cur = next_val
        leaf = path[-1]
        parsed: Any = value
        for cast in (int, float):
            try:
                parsed = cast(value)
                break
            except ValueError:
                continue
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        cur[leaf] = parsed
    return updated


def validate_resolved_config(config: Dict[str, Any]) -> None:
    walls = config["omega"]["walls"]
    if walls != [
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    ]:
        raise ValueError("Wall ordering mismatch with v1 contract")

    gamma_omega = config["omega"]["attribution"]["gamma"]
    gamma_policy = config["off_policy"]["block"]["gamma"]
    if abs(float(gamma_omega) - float(gamma_policy)) > 1e-9:
        raise ValueError("gamma mismatch between omega.attribution and off_policy.block")

    enforcement_mode = str(config["off_policy"].get("enforcement_mode", "ENFORCE")).upper()
    if enforcement_mode not in {"ENFORCE", "LOG_ONLY"}:
        raise ValueError("off_policy.enforcement_mode must be ENFORCE or LOG_ONLY")

    source_policy = config.get("source_policy", {})
    default_trust = source_policy.get("default_trust", "untrusted")
    valid_trust = {"trusted", "semi", "untrusted", "semi_trusted"}
    if default_trust not in valid_trust:
        raise ValueError("source_policy.default_trust must be trusted|semi|semi_trusted|untrusted")

    tools_cfg = config.get("tools", {})
    execution_mode = str(tools_cfg.get("execution_mode", "ENFORCE")).upper()
    if execution_mode not in {"ENFORCE", "DRY_RUN"}:
        raise ValueError("tools.execution_mode must be ENFORCE or DRY_RUN")



def load_resolved_config(
    config_dir: str = "config",
    profile: str = "dev",
    cli_overrides: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> ConfigSnapshot:
    root = Path(config_dir)
    files = {
        "pi0": root / "pi0_defaults.yml",
        "omega": root / "omega_defaults.yml",
        "off_policy": root / "off_policy.yml",
        "source_policy": root / "source_policy.yml",
        "tools": root / "tools.yml",
        "profile": root / "profiles" / f"{profile}.yml",
    }

    resolved: Dict[str, Any] = {}
    file_hashes: Dict[str, str] = {}

    for name in ("pi0", "omega", "off_policy", "source_policy", "tools", "profile"):
        path = files[name]
        if path.exists():
            file_hashes[str(path.as_posix())] = _sha256_bytes(path.read_bytes())
        layer = _load_yaml(path)
        resolved = _deep_merge(resolved, layer)

    resolved = _apply_env_overrides(resolved, env or os.environ)
    if cli_overrides:
        resolved = _deep_merge(resolved, cli_overrides)

    validate_resolved_config(resolved)

    resolved_json = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    resolved_sha = _sha256_bytes(resolved_json)

    LOGGER.info(
        "config_snapshot",
        extra={
            "file_hashes": file_hashes,
            "resolved_sha256": resolved_sha,
        },
    )

    return ConfigSnapshot(resolved=resolved, file_hashes=file_hashes, resolved_sha256=resolved_sha)


def config_refs_from_snapshot(snapshot: ConfigSnapshot, code_commit: str = "unknown") -> Dict[str, str]:
    refs = {
        "code_commit": code_commit,
        "resolved_config_sha256": snapshot.resolved_sha256,
    }
    for path, digest in snapshot.file_hashes.items():
        base = Path(path).name.replace(".yml", "")
        refs[f"{base}_sha256"] = digest
    return refs

