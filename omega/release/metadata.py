"""Single runtime source of version and contract metadata."""

from __future__ import annotations

import importlib.resources as resources
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ReleaseInfo:
    engine_version: str
    api_version: str
    config_schema_version: int
    runtime_state_schema_version: int
    control_plane_schema_version: int
    license_schema_version: int
    policy_pack_version: str
    vision_pack_version: str
    edition: str
    release_channel: str
    git_commit: str = "unknown"
    build_id: str = "local"
    build_timestamp: str = "unknown"

    @property
    def major_version(self) -> int:
        return int(self.engine_version.split(".", 1)[0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_packaged_metadata() -> Dict[str, Any]:
    path = resources.files("omega.release").joinpath("version.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("release version metadata must be a JSON object")
    return payload


def get_release_info() -> ReleaseInfo:
    payload = _load_packaged_metadata()
    return ReleaseInfo(
        engine_version=str(payload["engine_version"]),
        api_version=str(payload["api_version"]),
        config_schema_version=int(payload["config_schema_version"]),
        runtime_state_schema_version=int(payload["runtime_state_schema_version"]),
        control_plane_schema_version=int(payload["control_plane_schema_version"]),
        license_schema_version=int(payload["license_schema_version"]),
        policy_pack_version=str(payload["policy_pack_version"]),
        vision_pack_version=str(payload["vision_pack_version"]),
        edition=str(os.environ.get("OMEGA_EDITION", payload.get("edition", "community"))),
        release_channel=str(os.environ.get("OMEGA_RELEASE_CHANNEL", payload.get("release_channel", "candidate"))),
        git_commit=str(os.environ.get("OMEGA_GIT_COMMIT", "unknown")),
        build_id=str(os.environ.get("OMEGA_BUILD_ID", "local")),
        build_timestamp=str(os.environ.get("OMEGA_BUILD_TIMESTAMP", "unknown")),
    )
