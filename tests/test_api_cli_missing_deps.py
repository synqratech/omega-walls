from __future__ import annotations

import sys

import pytest

from omega.api import cli


def test_omega_walls_api_missing_optional_deps_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["omega-walls-api", "--profile", "quickstart"])

    def _raise_missing(*, cfg, profile):  # type: ignore[no-untyped-def]
        _ = (cfg, profile)
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    monkeypatch.setattr(cli, "_create_api_app", _raise_missing)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    message = str(exc_info.value)
    assert "omega-walls[api]" in message
    assert "fastapi" in message
