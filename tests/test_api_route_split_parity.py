from __future__ import annotations

from fastapi.testclient import TestClient

from omega.api import server as api_server


def _route_exists(app, path: str, method: str) -> bool:
    method_up = method.upper()
    for route in app.routes:
        if getattr(route, "path", "") == path and method_up in set(getattr(route, "methods", set())):
            return True
    return False


def test_route_split_registers_incident_replay_and_session_paths() -> None:
    app = api_server.create_app(resolved_config=api_server.load_resolved_config(profile="dev").resolved, profile="dev")
    assert _route_exists(app, "/v1/health", "GET")
    assert _route_exists(app, "/v1/incidents", "GET")
    assert _route_exists(app, "/v1/incidents/{incident_id}", "GET")
    assert _route_exists(app, "/v1/incidents/{incident_id}/replay/generate", "POST")
    assert _route_exists(app, "/v1/replay/jobs/{job_id}", "GET")
    assert _route_exists(app, "/v1/replay/downloads/{token}", "GET")
    assert _route_exists(app, "/v1/session/reset", "POST")


def test_problem_payload_passthrough_for_incident_paths() -> None:
    app = api_server.create_app(resolved_config=api_server.load_resolved_config(profile="dev").resolved, profile="dev")
    client = TestClient(app)
    resp = client.get("/v1/incidents")
    assert resp.status_code in {400, 401, 404}
    body = resp.json()
    assert set(["type", "title", "status", "detail", "instance"]).issubset(set(body.keys()))
    assert body["instance"] == "/v1/incidents"
