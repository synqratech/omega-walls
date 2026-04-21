from __future__ import annotations

from copy import deepcopy

from omega.interfaces.contracts_v1 import OffAction, ToolRequest
from omega.tools.tool_gateway import ToolGatewayV1


def test_gateway_freeze_disabled_mode(gateway):
    req = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=20)
    dec = gateway.enforce(req, [action])
    assert dec.allowed is False
    assert dec.reason == "TOOL_FREEZE_ACTIVE"


def test_gateway_allowlist(gateway):
    req_allowed = ToolRequest(tool_name="summarize", args={}, session_id="s", step=1)
    req_blocked = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_ALLOWLIST", allowlist=["summarize"])

    dec_allowed = gateway.enforce(req_allowed, [action])
    dec_blocked = gateway.enforce(req_blocked, [action])

    assert dec_allowed.allowed is True
    assert dec_blocked.allowed is False
    assert dec_blocked.reason == "NOT_IN_ALLOWLIST"


def test_gateway_requires_human_approval_for_dangerous_tools(gateway):
    req_no_approval = ToolRequest(tool_name="network_post", args={}, session_id="s", step=1)
    req_with_approval = ToolRequest(
        tool_name="network_post",
        args={"url": "https://example.com", "payload": "probe", "human_approved": True},
        session_id="s",
        step=1,
    )

    dec_no = gateway.enforce(req_no_approval, [])
    dec_yes = gateway.enforce(req_with_approval, [])

    assert dec_no.allowed is False
    assert dec_no.reason == "HUMAN_APPROVAL_REQUIRED"
    assert dec_yes.allowed is True


def test_gateway_freeze_allows_read_only_exception(gateway):
    action = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=5)
    req = ToolRequest(tool_name="summarize", args={}, session_id="s", step=1)
    dec = gateway.enforce(req, [action])
    assert dec.allowed is True


def test_gateway_requires_deny_unknown_policy(resolved_config):
    cfg = deepcopy(resolved_config)
    cfg["tools"]["unknown_tool_policy"] = "ALLOW"
    try:
        ToolGatewayV1(cfg)
    except ValueError as exc:
        assert "unknown_tool_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-DENY unknown_tool_policy")


def test_gateway_requires_human_approval_for_write_dangerous_capabilities(resolved_config):
    cfg = deepcopy(resolved_config)
    cfg["tools"]["capabilities"]["write_file"]["requires_human_approval"] = False
    try:
        ToolGatewayV1(cfg)
    except ValueError as exc:
        assert "must require human approval" in str(exc)
    else:
        raise AssertionError("Expected ValueError for write capability without approval")


def test_gateway_require_approval_action_blocks_without_human_approval(gateway):
    req = ToolRequest(tool_name="summarize", args={}, session_id="s", step=1)
    action = OffAction(
        type="REQUIRE_APPROVAL",
        target="TOOLS",
        allowlist=["summarize"],
        horizon_steps=8,
    )
    dec = gateway.enforce(req, [action])
    assert dec.allowed is False
    assert dec.reason == "REQUIRE_APPROVAL_PENDING"


def test_gateway_freeze_has_precedence_over_require_approval(gateway):
    req = ToolRequest(
        tool_name="network_post",
        args={"url": "https://example.com", "payload": "probe", "human_approved": True},
        session_id="s",
        step=1,
    )
    freeze = OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED", horizon_steps=4)
    require_approval = OffAction(type="REQUIRE_APPROVAL", target="TOOLS", allowlist=["network_post"], horizon_steps=8)
    dec = gateway.enforce(req, [require_approval, freeze])
    assert dec.allowed is False
    assert dec.reason == "TOOL_FREEZE_ACTIVE"


def test_gateway_denies_network_post_missing_payload_after_approval(gateway):
    req = ToolRequest(
        tool_name="network_post",
        args={"url": "https://example.com", "human_approved": True},
        session_id="s",
        step=1,
    )
    dec = gateway.enforce(req, [])
    assert dec.allowed is False
    assert dec.reason == "INVALID_TOOL_ARGS_SCHEMA"
    assert dec.validation_status == "failed"


def test_gateway_denies_network_post_with_invalid_scheme(gateway):
    req = ToolRequest(
        tool_name="network_post",
        args={"url": "ftp://example.com", "payload": "x", "human_approved": True},
        session_id="s",
        step=1,
    )
    dec = gateway.enforce(req, [])
    assert dec.allowed is False
    assert dec.reason == "INVALID_TOOL_ARGS_SECURITY"
    assert dec.validation_status == "failed"


def test_gateway_denies_write_file_traversal(gateway):
    req = ToolRequest(
        tool_name="write_file",
        args={"filename": "../secrets.txt", "content": "x", "human_approved": True},
        session_id="s",
        step=1,
    )
    dec = gateway.enforce(req, [])
    assert dec.allowed is False
    assert dec.reason == "INVALID_TOOL_ARGS_SECURITY"
    assert dec.validation_status == "failed"


def test_gateway_denies_shell_like_destructive_command(resolved_config):
    cfg = deepcopy(resolved_config)
    cfg["tools"]["capabilities"]["powershell_exec"] = {
        "mode": "dangerous",
        "allowed_when": ["NO_OFF"],
        "requires_human_approval": True,
    }
    gateway = ToolGatewayV1(cfg)
    req = ToolRequest(
        tool_name="powershell_exec",
        args={"command": "rm -rf /", "human_approved": True},
        session_id="s",
        step=1,
    )
    dec = gateway.enforce(req, [])
    assert dec.allowed is False
    assert dec.reason == "INVALID_TOOL_ARGS_SHELLLIKE"
    assert dec.validation_status == "failed"
