"""Build and validate immutable enterprise release manifests."""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from jsonschema import Draft202012Validator, FormatChecker

from .metadata import ReleaseInfo, get_release_info


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_schema() -> Mapping[str, Any]:
    payload = json.loads(
        resources.files("omega.release").joinpath("schemas", "release_manifest_v1.schema.json").read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping):
        raise ValueError("release manifest schema must be an object")
    return payload


def validate_release_manifest(payload: Mapping[str, Any], *, check_runtime_metadata: bool = True) -> None:
    errors = sorted(
        Draft202012Validator(load_manifest_schema(), format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "$"
        raise ValueError(f"release manifest schema validation failed at {location}: {first.message}")
    channel = str(payload.get("channel", ""))
    if channel in {"stable", "lts"}:
        if str(payload.get("git_commit", "")) == "unknown":
            raise ValueError("stable/lts release manifests require a concrete git commit")
        if str(payload.get("build_id", "")) in {"", "local", "unknown"}:
            raise ValueError("stable/lts release manifests require a non-local build_id")
    for name, artifact in dict(payload.get("artifacts", {})).items():
        if not isinstance(artifact, Mapping):
            continue
        if str(artifact.get("sha256", "")) == "0" * 64 and channel in {"stable", "lts"}:
            raise ValueError(f"stable/lts release artifact uses a placeholder hash: {name}")
        if ":latest" in str(artifact.get("path", "")).lower():
            raise ValueError(f"mutable latest artifact reference is forbidden: {name}")
    if check_runtime_metadata:
        info = get_release_info()
        contracts = payload["contracts"]
        packs = payload["packs"]
        expected = {
            "release": info.engine_version,
            "api": info.api_version,
            "config_schema": info.config_schema_version,
            "runtime_state_schema": info.runtime_state_schema_version,
            "control_plane_schema": info.control_plane_schema_version,
            "license_schema": info.license_schema_version,
            "policy": info.policy_pack_version,
            "vision": info.vision_pack_version,
        }
        observed = {
            "release": payload["release"],
            "api": contracts["api"],
            "config_schema": contracts["config_schema"],
            "runtime_state_schema": contracts["runtime_state_schema"],
            "control_plane_schema": contracts["control_plane_schema"],
            "license_schema": contracts["license_schema"],
            "policy": packs["policy"],
            "vision": packs["vision"],
        }
        mismatches = [key for key, value in expected.items() if observed.get(key) != value]
        if mismatches:
            raise ValueError("release manifest conflicts with runtime metadata: " + ", ".join(mismatches))


def artifact_record(path: str | Path, *, kind: str = "file", oci_digest: Optional[str] = None) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    record: Dict[str, Any] = {
        "kind": str(kind),
        "path": file_path.as_posix(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }
    if oci_digest is not None:
        record["oci_digest"] = str(oci_digest)
    return record


def build_release_manifest(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    channel: Optional[str] = None,
    git_commit: Optional[str] = None,
    build_id: Optional[str] = None,
    created_utc: Optional[str] = None,
    release_info: Optional[ReleaseInfo] = None,
    compatibility: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    info = release_info or get_release_info()
    compat = dict(compatibility or {})
    manifest: MutableMapping[str, Any] = {
        "schema_version": 1,
        "release": info.engine_version,
        "channel": str(channel or info.release_channel),
        "git_commit": str(git_commit or info.git_commit),
        "build_id": str(build_id or info.build_id),
        "created_utc": str(created_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        "artifacts": {str(name): dict(record) for name, record in artifacts.items()},
        "contracts": {
            "api": info.api_version,
            "config_schema": info.config_schema_version,
            "runtime_state_schema": info.runtime_state_schema_version,
            "control_plane_schema": info.control_plane_schema_version,
            "license_schema": info.license_schema_version,
        },
        "packs": {"policy": info.policy_pack_version, "vision": info.vision_pack_version},
        "compatibility": {
            "upgrade_from": list(compat.get("upgrade_from", [">=0.5.0,<0.7.0"])),
            "rollback_to": list(compat.get("rollback_to", [">=0.5.0,<0.7.0"])),
            "irreversible_state_migrations": bool(compat.get("irreversible_state_migrations", False)),
        },
    }
    validate_release_manifest(manifest)
    return dict(manifest)
