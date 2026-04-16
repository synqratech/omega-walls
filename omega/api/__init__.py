from __future__ import annotations

from typing import Any


def create_app(*, resolved_config: dict[str, Any] | None = None, profile: str = "dev"):
    from omega.api.server import create_app as _create_app

    return _create_app(resolved_config=resolved_config, profile=profile)


__all__ = ["create_app"]
