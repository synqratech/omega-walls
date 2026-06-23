from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import socket
import threading
import zipfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from omega.api.auth import _request_is_https_proxy_mode
from omega.api.middleware import RequestDeadlineMiddleware, StreamingBodyLimitMiddleware
from omega.api.runtime_factory import ApiSecurity
from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import (
    ContentItem,
    OffAction,
    OmegaOffReasons,
    OmegaState,
    OmegaStepResult,
    ProjectionEvidence,
    ProjectionResult,
    ToolRequest,
)
from omega.notifications.models import ApprovalRecord, RiskEvent, utc_now_iso
from omega.notifications.providers import WebhookNotifier
from omega.notifications.store import InMemoryApprovalStore, SQLiteApprovalStore
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.rag.attachment_ingestion import extract_attachment
from omega.security.paths import PathContainmentError, resolve_contained_path
from omega.tools.approval import tool_args_sha256, tool_intent_id
from omega.tools.arg_validation import ToolArgValidationConfig, validate_tool_args
from omega.tools.tool_gateway import ToolGatewayV1


def _cfg() -> dict:
    return load_resolved_config(profile="dev").resolved


def _future_iso(minutes: int = 10) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_server_side_approval_is_exact_and_single_use() -> None:
    cfg = _cfg()
    store = InMemoryApprovalStore()
    gateway = ToolGatewayV1(config=cfg, approval_store=store)
    action = OffAction(type="REQUIRE_APPROVAL", target="TOOLS", allowlist=["summarize"])
    request = ToolRequest(
        tool_name="summarize",
        args={"text": "approved exact payload", "human_approved": True},
        session_id="sess-1",
        tenant_id="tenant-1",
        actor_id="actor-1",
        step=7,
    )
    first = gateway.enforce(request, [action])
    assert first.allowed is False
    assert first.reason == "REQUIRE_APPROVAL_PENDING"
    intent = tool_intent_id(request)
    approval = ApprovalRecord(
        approval_id="apr_exact",
        status="approved",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        expires_at=_future_iso(),
        required_action="EXECUTE_TOOL",
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        actor_id=request.actor_id,
        trace_id="trc",
        decision_id="dec",
        control_outcome="REQUIRE_APPROVAL",
        approval_scope="tool_intent",
        tool_name=request.tool_name,
        tool_args_sha256=tool_args_sha256(request.args),
        tool_intent_id=intent,
        single_use=True,
    )
    store.create(approval)
    request.approval_id = approval.approval_id
    allowed = gateway.enforce(request, [action])
    assert allowed.allowed is True
    replay = gateway.enforce(request, [action])
    assert replay.allowed is False
    assert replay.reason == "APPROVAL_INVALID_EXPIRED_OR_CONSUMED"

    changed = replace(request, args={"text": "changed payload"})
    changed.approval_id = approval.approval_id
    assert gateway.enforce(changed, [action]).allowed is False



def test_sqlite_approval_consumption_is_atomic_across_store_instances(tmp_path: Path) -> None:
    db = tmp_path / "approvals.sqlite3"
    store_a = SQLiteApprovalStore(sqlite_path=db)
    store_b = SQLiteApprovalStore(sqlite_path=db)
    request = ToolRequest(
        tool_name="summarize",
        args={"text": "exact payload"},
        session_id="sess-race",
        tenant_id="tenant-race",
        actor_id="actor-race",
        step=9,
    )
    intent = tool_intent_id(request)
    args_hash = tool_args_sha256(request.args)
    approval = ApprovalRecord(
        approval_id="apr_race",
        status="approved",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        expires_at=_future_iso(),
        required_action="EXECUTE_TOOL",
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        actor_id=request.actor_id,
        trace_id="trc",
        decision_id="dec",
        control_outcome="REQUIRE_APPROVAL",
        approval_scope="tool_intent",
        tool_name=request.tool_name,
        tool_args_sha256=args_hash,
        tool_intent_id=intent,
        single_use=True,
    )
    store_a.create(approval)
    barrier = threading.Barrier(2)
    results: list[ApprovalRecord | None] = []
    lock = threading.Lock()

    def consume(store: SQLiteApprovalStore, step: int) -> None:
        barrier.wait(timeout=5)
        result = store.consume_tool_approval(
            approval_id=approval.approval_id,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            tool_name=request.tool_name,
            tool_args_sha256=args_hash,
            tool_intent_id=intent,
            step=step,
            now_iso=utc_now_iso(),
        )
        with lock:
            results.append(result)

    threads = [
        threading.Thread(target=consume, args=(store_a, 10)),
        threading.Thread(target=consume, args=(store_b, 11)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sum(result is not None for result in results) == 1
    consumed = store_a.get(approval.approval_id)
    assert consumed is not None
    assert consumed.consumed_at
    assert consumed.consumed_by_step in {10, 11}

def test_containment_blocks_traversal_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PathContainmentError):
        resolve_contained_path(root, "..", "escape.txt")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PathContainmentError):
        resolve_contained_path(root, "link", "escape.txt")
    safe = resolve_contained_path(root, "tenant", "session", "result.txt", create_parent=True)
    assert safe.is_relative_to(root.resolve())


def test_state_only_off_has_no_current_document_attribution() -> None:
    cfg = _cfg()
    params = omega_params_from_config(cfg)
    core = OmegaCoreV1(params)
    state = OmegaState(session_id="s", m=np.array([(params.off_Theta + 0.1) / params.lam, 0, 0, 0]), step=0)
    item = ContentItem(doc_id="benign", source_id="src", source_type="web", trust="untrusted", text="hello")
    projection = ProjectionResult(
        doc_id="benign",
        v=np.zeros(4),
        evidence=ProjectionEvidence(polarity=[0, 0, 0, 0], debug_scores_raw=[0, 0, 0, 0], matches={}),
    )
    result = core.step(state, [item], [projection])
    assert result.off is True
    assert result.top_docs == []
    assert result.attribution_mode == "state_only"
    decision = OffPolicyV1(cfg).select_actions(result, [item])
    soft = next(a for a in decision.actions if a.type == "SOFT_BLOCK")
    assert soft.target == "SESSION"
    assert not soft.doc_ids


def test_decayed_trace_below_participation_threshold_does_not_escalate() -> None:
    cfg = _cfg()
    policy = OffPolicyV1(cfg)
    result = OmegaStepResult(
        session_id="s",
        step=5,
        v_total=np.zeros(4),
        p=np.zeros(4),
        m_prev=np.array([0.8, 1e-12, 0, 0]),
        m_next=np.array([0.8, 1e-12, 0, 0]),
        off=True,
        reasons=OmegaOffReasons(False, True, False, False),
        top_docs=[],
        contribs=[],
        attribution_mode="state_only",
    )
    walls = policy._participating_walls(result)
    assert "secret_exfiltration" not in walls
    decision = policy.select_actions(result, [])
    assert not any(a.type == "HUMAN_ESCALATE" for a in decision.actions)


def test_strict_projection_alignment_rejects_reordering() -> None:
    core = OmegaCoreV1(omega_params_from_config(_cfg()))
    items = [
        ContentItem(doc_id="a", source_id="a", source_type="web", trust="untrusted", text="a"),
        ContentItem(doc_id="b", source_id="b", source_type="web", trust="untrusted", text="b"),
    ]
    ev = ProjectionEvidence(polarity=[0, 0, 0, 0], debug_scores_raw=[0, 0, 0, 0], matches={})
    projections = [ProjectionResult(doc_id="b", v=np.zeros(4), evidence=ev), ProjectionResult(doc_id="a", v=np.zeros(4), evidence=ev)]
    with pytest.raises(ValueError, match="identical doc_id order"):
        core.step(OmegaState(session_id="s", m=np.zeros(4)), items, projections)


def test_ssrf_policy_blocks_private_resolution_and_allows_public_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ToolArgValidationConfig.from_tools_config({
        "enabled": True,
        "fail_mode": "deny",
        "network_post": {
            "allowed_hosts": ["api.example.com"],
            "allowed_ports": [443],
            "allowed_schemes": ["https"],
            "resolve_dns": True,
        },
    })
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    denied = validate_tool_args("network_post", {"url": "https://api.example.com/v1", "payload": "x"}, cfg)
    assert denied.allowed is False
    assert "non-public" in str(denied.reason)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))])
    allowed = validate_tool_args("network_post", {"url": "https://api.example.com/v1", "payload": "x"}, cfg)
    assert allowed.allowed is True
    assert validate_tool_args("network_post", {"url": "https://evil.example/v1", "payload": "x"}, cfg).allowed is False


def test_streaming_body_limit_rejects_before_route_buffering() -> None:
    app = FastAPI()
    app.add_middleware(RequestDeadlineMiddleware, timeout_sec=2)
    app.add_middleware(StreamingBodyLimitMiddleware, max_body_bytes=8)

    @app.post("/echo")
    async def echo(request: Request):
        return {"size": len(await request.body())}

    with TestClient(app) as client:
        assert client.post("/echo", content=b"12345678").status_code == 200
        assert client.post("/echo", content=b"123456789").status_code == 413


def _request(peer: str, forwarded_proto: str) -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "query_string": b"",
        "headers": [(b"x-forwarded-proto", forwarded_proto.encode())],
        "client": (peer, 1234),
        "server": ("app", 80),
    })


def test_forwarded_https_only_from_trusted_proxy() -> None:
    security = ApiSecurity(transport_mode="proxy_tls", require_https=True, trusted_proxy_cidrs=["127.0.0.1/32"])
    assert _request_is_https_proxy_mode(_request("127.0.0.1", "https"), security) is True
    assert _request_is_https_proxy_mode(_request("203.0.113.10", "https"), security) is False


def test_attachment_strict_magic_and_archive_limits() -> None:
    hardened = {
        "enabled": True,
        "strict_magic": True,
        "sandbox": {"enabled": False},
        "parser_limits": {"max_docx_entries": 4, "max_docx_uncompressed_bytes": 32},
    }
    with pytest.raises(ValueError, match="format mismatch"):
        extract_attachment(content_bytes=b"<html><body>x</body></html>", filename="fake.pdf", cfg=hardened)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "x")
        zf.writestr("word/document.xml", "A" * 128)
    with pytest.raises(ValueError, match="uncompressed size"):
        extract_attachment(content_bytes=buf.getvalue(), filename="x.docx", cfg=hardened)


def test_attachment_html_runs_in_bounded_process() -> None:
    hardened = {
        "enabled": True,
        "strict_magic": True,
        "sandbox": {"enabled": True, "timeout_sec": 8, "max_memory_mb": 512, "max_cpu_sec": 6},
        "parser_limits": {"max_html_nodes": 1000},
    }
    out = extract_attachment(
        content_bytes=b"<html><body><p>visible</p><div hidden>hidden</div></body></html>",
        filename="x.html",
        cfg=hardened,
    )
    assert "visible" in out.text
    assert "hidden_text_present" in out.warnings


def test_webhook_action_request_uses_webhook_not_telegram_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    import omega.notifications.providers as providers
    monkeypatch.setattr(providers, "_http_post_json", lambda **kwargs: {"id": "ok"})
    notifier = WebhookNotifier(url="https://hooks.example.com/omega", allowed_types=[], allowed_hosts=["hooks.example.com"])
    risk = RiskEvent(
        event_id="e", timestamp=utc_now_iso(), surface="runtime", control_outcome="REQUIRE_APPROVAL",
        triggers=[], reasons=[], action_types=[], trace_id="t", decision_id="d", tenant_id="tenant", session_id="sess",
    )
    from omega.notifications.models import ActionRequestEvent
    event = ActionRequestEvent(
        approval_id="apr", risk_event=risk, required_action="EXECUTE_TOOL", timeout_sec=30,
        approval_scope="tool_intent", tool_name="summarize", tool_args_sha256="abc", tool_intent_id="intent",
    )
    assert asyncio.run(notifier.send_action_request(event)) == "ok"
