from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from omega.api import server as api_server
from omega.rag import harness as harness_module


def test_api_create_app_calls_startup_notifications(monkeypatch) -> None:
    called: Dict[str, Any] = {}

    def _fake_make_runtime(resolved_config: Dict[str, Any]) -> Any:
        return SimpleNamespace(
            config=resolved_config,
            projector=SimpleNamespace(semantic_active=True),
            notification_dispatcher=None,
        )

    def _fake_run_startup_notifications(**kwargs: Any) -> Dict[str, Any]:
        called.update(kwargs)
        return {"ok": True, "surface": kwargs.get("surface")}

    monkeypatch.setattr(api_server, "_make_runtime", _fake_make_runtime)
    monkeypatch.setattr(api_server, "run_startup_notifications", _fake_run_startup_notifications)

    app = api_server.create_app(
        resolved_config={"api": {"enabled": True}, "notifications": {"enabled": False}},
        profile="dev",
    )
    assert app.state.startup_summary["ok"] is True
    assert called.get("surface") == "api"
    assert called.get("profile") == "dev"


def test_harness_init_calls_startup_notifications(monkeypatch) -> None:
    called: Dict[str, Any] = {}

    def _fake_run_startup_notifications(**kwargs: Any) -> Dict[str, Any]:
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(harness_module, "run_startup_notifications", _fake_run_startup_notifications)
    monkeypatch.setattr(
        harness_module.EnforcementStateManager,
        "from_config",
        staticmethod(lambda cfg: SimpleNamespace(reset=lambda: None)),
    )
    monkeypatch.setattr(
        harness_module.CrossSessionStateManager,
        "from_config",
        staticmethod(lambda cfg: SimpleNamespace(fallback_actor_to_session=True)),
    )
    monkeypatch.setattr(harness_module, "build_monitor_collector_from_config", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(harness_module, "build_dispatcher_from_config", lambda **kwargs: SimpleNamespace(metrics_snapshot=lambda: {}))
    monkeypatch.setattr(harness_module, "build_structured_emitter_from_config", lambda **kwargs: SimpleNamespace())

    projector = SimpleNamespace(semantic_active=True)
    omega_core = SimpleNamespace()
    off_policy = SimpleNamespace()
    tool_gateway = SimpleNamespace(ensure_tool_coverage=lambda tools: None)
    tool_registry = SimpleNamespace(list_tools=lambda: [])
    cfg = {"profiles": {"env": "dev"}, "notifications": {"enabled": False}}

    obj = harness_module.OmegaRAGHarness(
        projector=projector,
        omega_core=omega_core,
        off_policy=off_policy,
        tool_gateway=tool_gateway,
        config=cfg,
        tool_registry=tool_registry,
    )
    assert obj.startup_summary["ok"] is True
    assert called.get("surface") == "runtime"
    assert called.get("profile") == "dev"
