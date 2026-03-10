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
        args={"human_approved": True},
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
